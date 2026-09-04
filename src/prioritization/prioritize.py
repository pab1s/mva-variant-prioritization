"""Core prioritization logic combining annotation + phenotype."""
import argparse
import gzip
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

from src.phenotype.hpo import load_hpo_terms, hpo_overlap_score
from src.prioritization.features import build_features, CONSEQUENCE_WEIGHTS
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
                if csq_fields and len(parts) == len(csq_fields):
                    d = dict(zip(csq_fields, parts))
                    out["Consequence"] = d.get("Consequence", "")
                    out["SYMBOL"] = d.get("SYMBOL") or d.get("Gene", "")
                    out["IMPACT"] = d.get("IMPACT", "")
                    out["CADD_PHRED"] = d.get("CADD_PHRED") or d.get("CADD", "")
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

                # Build pathogenicity features
                feats = build_features(variant_dict)

                # Phenotype score (baseline neutral, future gene-aware)
                gene = variant_dict.get("SYMBOL") or variant_dict.get("Gene") or None
                # gene_hpo_map would be loaded from external resource; None for baseline
                pheno_score = hpo_overlap_score(hpo_terms, gene_hpo_map=None, gene=gene)
                # Alternatively, boost if gene is known MVA gene (e.g. BUB1B, TRIP13, CEP57) — baseline heuristic
                # These are MVA-associated genes; boost if present
                mva_genes = {"BUB1B": 1.0, "BUB3": 0.9, "TRIP13": 1.0, "CEP57": 1.0, "BUB1": 0.9, "MAD2L1": 0.8, "CENATAC": 0.8}
                if gene in mva_genes:
                    pheno_score = max(pheno_score, mva_genes[gene])

                pathogenicity_score = feats["pathogenicity_score"]
                final_score = pheno_w * pheno_score + path_w * pathogenicity_score

                row = {
                    "CHROM": chrom,
                    "POS": int(pos),
                    "REF": ref,
                    "ALT": alt_allele,
                    "QUAL": qual,
                    "FILTER": filt,
                    "SYMBOL": gene or ".",
                    "Consequence": feats.get("consequence", ""),
                    "gnomad_af": feats.get("af"),
                    "CADD_PHRED": variant_dict.get("CADD_PHRED", ""),
                    "score": final_score,
                    "phenotype_score": pheno_score,
                    "pathogenicity_score": pathogenicity_score,
                    "rare_score": feats["rare_score"],
                    "consequence_weight": feats["consequence_weight"],
                    "quality_score": feats["quality_score"],
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
