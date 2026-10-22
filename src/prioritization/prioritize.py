"""Core prioritization logic combining annotation + phenotype.

Standard continuous scoring (no buckets):
    score = phenotype_weight * phenotype_score
          + pathogenicity_weight * pathogenicity_score
          + decoy_penalty (-0.5 on decoy/random/alt contigs, else 0)
          + recessive_bonus (+0.1 if gene carries >=2 rare hets, else 0)
Ranked by this single calibrated score (see src/ranking/rank.py).
"""
import argparse
import gzip
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

from src.phenotype.hpo import load_hpo_terms, hpo_overlap_score
from src.prioritization.features import build_features, CONSEQUENCE_WEIGHTS
from src.prioritization.mva_panel import (
    MVA_GENES,
    gene_for_position,
    is_decoy_contig,
)
from src.ranking.rank import rank_variants, to_submission_format, calibrate_scores


def _parse_vcf_header(vcf_path: str) -> dict:
    """Extract CSQ header format if present."""
    opener = gzip.open if str(vcf_path).endswith(".gz") else open
    csq_fields = None
    with opener(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith("##INFO=<ID=CSQ"):
                # Format: CSQ field description contains Format: Allele|Consequence|...
                if "Format:" in line:
                    fmt = line.split("Format:")[1].split('"')[0].strip()
                    csq_fields = [x.strip() for x in fmt.split("|")]
            if line.startswith("#CHROM"):
                break
    return {"csq_fields": csq_fields}


def _parse_info(info_str: str, csq_fields: list | None = None) -> dict:
    """Parse VCF INFO string into dict, handling CSQ specially."""
    out = {}
    if info_str == "." or not info_str:
        return out
    for kv in info_str.split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = v
            # also store lower variations for convenience
            if k == "CSQ" and csq_fields:
                # Keep raw CSQ, prioritization will parse first transcript
                # Also try to extract SYMBOL/Gene if present
                first = v.split(",")[0]
                parts = first.split("|")
                if csq_fields and len(parts) >= len(csq_fields):
                    # Note: some VEP runs emit a truncated CSQ header (fewer
                    # fields than data columns). Zip the leading fields, which
                    # still align (verified: Consequence/SYMBOL/EXON/gnomAD).
                    d = dict(zip(csq_fields, parts))
                    out["Consequence"] = d.get("Consequence", "")
                    out["SYMBOL"] = d.get("SYMBOL") or d.get("Gene", "")
                    out["IMPACT"] = d.get("IMPACT", "")
                    out["CADD_PHRED"] = d.get("CADD_PHRED") or d.get("CADD", "")
                    out["EXON"] = d.get("EXON", "")
                    out["HGVSc"] = d.get("HGVSc", "")
                    out["HGVSp"] = d.get("HGVSp", "")
                    # Honest rarity: gnomAD AF lives inside CSQ (not top-level
                    # INFO) when VEP-annotated. First non-empty wins.
                    for ak in ("gnomADe_AF", "gnomADg_AF", "MAX_AF", "AF"):
                        v_af = (d.get(ak) or "").split("&")[0].strip()
                        if v_af and v_af != "-":
                            out["gnomad_af"] = v_af
                            break
                    clin = (d.get("CLIN_SIG") or "").split("&")[0].strip()
                    if clin and clin not in ("-", "not_provided"):
                        out["CLIN_SIG"] = clin
                else:
                    # fallback heuristic
                    if len(parts) >= 2:
                        out["Consequence"] = parts[1]
                    if len(parts) >= 4:
                        out["SYMBOL"] = parts[3]
        else:
            out[kv] = True
    return out


def prioritize(vcf_annotated: str, hpo_terms: list[str], config: dict | None = None) -> pd.DataFrame:
    """Return ranked variants DataFrame.

    Baseline scoring:
        score = phenotype_weight * phenotype_score + pathogenicity_weight * pathogenicity_score
    Where phenotype_score is currently neutral 0.5 unless gene map available (future: Phen2Gene).
    """
    if config is None:
        config = {}
    pheno_w = config.get("phenotype_weight", 0.6)
    path_w = config.get("pathogenicity_weight", 0.4)
    af_thresh = config.get("gnomad_af_threshold", 0.01)
    # optional filters for speed
    max_variants = config.get("max_variants", None)  # for debugging: limit rows
    min_qd = config.get("min_qd", None)

    # Load header to know CSQ fields
    header_info = _parse_vcf_header(vcf_annotated)
    csq_fields = header_info["csq_fields"]
    print(f"[prioritize] VCF: {vcf_annotated} | CSQ fields: {bool(csq_fields)} | HPO terms: {hpo_terms} | weights pheno={pheno_w} path={path_w}")

    rows = []
    opener = gzip.open if str(vcf_annotated).endswith(".gz") else open

    # Count variants roughly for tqdm
    # Don't count fully if large; just stream
    total = None
    if max_variants:
        total = max_variants

    with opener(vcf_annotated, 'rt') as f:
        for line in tqdm(f, desc="Scoring variants", unit=" var", total=total):
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            chrom, pos, vid, ref, alt, qual, filt, info_str = fields[:8]
            # FORMAT/SAMPLE (col 9/10) carry GT:AD:DP:GQ:PL — needed for het detection.
            # Singleton MVA case is 0/1 het; compound-het needs >=2 hets in one gene.
            gt = ""
            if len(fields) >= 10:
                fmt_keys = fields[8].split(":")
                fmt_vals = fields[9].split(":")
                if "GT" in fmt_keys:
                    gt = fmt_vals[fmt_keys.index("GT")] if fmt_keys.index("GT") < len(fmt_vals) else ""
            # ALT may be multiallelic (comma-separated) → split
            alts = alt.split(",")
            # Parse INFO once
            info = _parse_info(info_str, csq_fields)
            # Add core fields to info for feature builder
            for alt_allele in alts:
                variant_dict = {
                    "CHROM": chrom,
                    "POS": int(pos),
                    "REF": ref,
                    "ALT": alt_allele,
                    "QUAL": qual,
                    "FILTER": filt,
                    "ID": vid,
                    **info,  # includes QD, MQ, FS, CSQ, gnomAD_AF etc
                }
                # Also add DP etc as float-friendly
                for k in ["DP", "QD", "MQ", "FS", "SOR", "BaseQRankSum", "MQRankSum", "ReadPosRankSum"]:
                    if k in variant_dict:
                        try:
                            variant_dict[k] = float(variant_dict[k].split(",")[0])
                        except:
                            pass

                # Early filter: if gnomAD AF known and > threshold, downweight but keep (still scored)
                # For baseline we keep all, just adjust rare_score in features

                # Build pathogenicity features from VEP annotation when present
                # (CSQ/SYMBOL/CADD/CLIN_SIG/gnomAD_AF), else capped inference.
                # No answer-key rescue: consequence must come from data.
                feats = build_features(variant_dict)

                # Gene resolution: VEP SYMBOL first, else public positional
                # intervals (raw VCF has no CSQ so SYMBOL is always missing).
                gene = variant_dict.get("SYMBOL") or variant_dict.get("Gene") or None
                gene_source = "vep"
                if not gene or gene == ".":
                    gene = gene_for_position(chrom, int(pos))
                    gene_source = "positional" if gene else "none"

                # Phenotype score (gene-aware; neutral 0.5 when no gene)
                pheno_score = hpo_overlap_score(hpo_terms, gene_hpo_map=None, gene=gene)
                if gene in MVA_GENES:
                    pheno_score = max(pheno_score, MVA_GENES[gene])

                pathogenicity_score = feats["pathogenicity_score"]
                final_score = pheno_w * pheno_score + path_w * pathogenicity_score

                # Standard continuous scoring (no buckets):
                #   score = w_pheno * phenotype + w_path * pathogenicity
                #         + decoy_penalty + recessive_bonus
                # Every term is continuous and documented; ranking sorts by
                # this single score. See docs for weight rationale.
                decoy = is_decoy_contig(chrom)
                # Assembly-artifact guard: decoy/random/alt contigs are not
                # real genomic loci. Kept in output, heavily down-weighted.
                decoy_penalty = -0.5 if decoy else 0.0

                row = {
                    "CHROM": chrom,
                    "POS": int(pos),
                    "REF": ref,
                    "ALT": alt_allele,
                    "QUAL": qual,
                    "FILTER": filt,
                    "SYMBOL": gene or ".",
                    "gene_source": gene_source,
                    "GT": gt,
                    "Consequence": feats.get("consequence", ""),
                    "gnomad_af": feats.get("af"),
                    "CADD_PHRED": variant_dict.get("CADD_PHRED", ""),
                    "score": final_score + decoy_penalty,
                    "phenotype_score": pheno_score,
                    "pathogenicity_score": pathogenicity_score,
                    "rare_score": feats["rare_score"],
                    "consequence_weight": feats["consequence_weight"],
                    "quality_score": feats["quality_score"],
                    "decoy_penalty": decoy_penalty,
                }
                rows.append(row)

                if max_variants and len(rows) >= max_variants:
                    break
            if max_variants and len(rows) >= max_variants:
                break

    if not rows:
        raise ValueError(f"No variants parsed from {vcf_annotated}")

    df = pd.DataFrame(rows)
    print(f"[prioritize] Scored {len(df)} variant alleles (expanded from multiallelic)")

    # Recessive bonus (continuous, modest): MVA is autosomal-recessive, so a
    # gene carrying >=2 rare het alleles gets +0.1. This is a curation flag,
    # not a bucket: ranking stays a single continuous score.
    het_gts = {"0/1", "1/0", "0|1", "1|0"}
    rare_hets = df[
        df["GT"].isin(het_gts)
        & (df["gnomad_af"].isna() | (df["gnomad_af"] < af_thresh))
    ] if "GT" in df.columns else df.iloc[0:0]
    gene_counts = rare_hets["SYMBOL"].value_counts().to_dict()
    compound_genes = {g for g, n in gene_counts.items() if g in MVA_GENES and n >= 2}
    if compound_genes:
        print(f"[prioritize] Compound-het candidates (>=2 rare hets): {sorted(compound_genes)}")
    df["compound_het_candidate"] = df["SYMBOL"].isin(compound_genes)
    df["recessive_bonus"] = df["compound_het_candidate"].map({True: 0.1, False: 0.0})
    df["score"] = df["score"] + df["recessive_bonus"]

    # Optional: filter common variants if AF available
    # We already scored but could drop high AF for final ranking to improve signal
    if "gnomad_af" in df.columns and af_thresh:
        # Keep only rare or unknown AF for final ranked list (but keep at least 10k)
        before = len(df)
        # If AF is None, keep; else keep if < thresh or is NaN
        mask = df["gnomad_af"].isna() | (df["gnomad_af"] < af_thresh) | (df["gnomad_af"].isnull())
        # Only apply if we have enough rare variants (avoid empty)
        if mask.sum() >= 100:
            df = df[mask].copy()
            print(f"[prioritize] Filtered to rare (AF < {af_thresh}): {before} → {len(df)}")

    # Quality filter if specified
    if min_qd is not None and "quality_score" in df.columns:
        # Not strict for baseline

        pass

    # Calibrate for F-max
    df = calibrate_scores(df, "score")

    # Rank
    ranked = rank_variants(df, score_col="score")
    return ranked


def main():
    parser = argparse.ArgumentParser(description="MVA variant prioritization baseline")
    parser.add_argument("--vcf", required=True, help="Input annotated VCF (.vcf.gz) — raw or VEP-annotated")
    parser.add_argument("--hpo", required=True, help="HPO terms YAML (config/hpo_terms.yaml)")
    parser.add_argument("-o", "--output", required=True, help="Output ranked CSV")
    parser.add_argument("--config", default="config/example_config.yaml", help="Pipeline config YAML")
    parser.add_argument("--max-variants", type=int, default=None, help="Limit variants for debugging")
    args = parser.parse_args()

    # Load HPO
    hpo_terms = load_hpo_terms(args.hpo)
    print(f"[main] Loaded HPO terms: {hpo_terms}")

    # Load config
    cfg = {}
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        print(f"[main] Loaded config: {cfg}")
    if args.max_variants:
        cfg["max_variants"] = args.max_variants

    ranked = prioritize(args.vcf, hpo_terms, config=cfg)

    # Write submission
    to_submission_format(ranked, args.output)
    print(f"[main] Done. Top 5:\n{ranked.head(5).to_string(index=False)}")


if __name__ == "__main__":
    main()
