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
from src.phenotype.resnik import load_phenotype_model
from src.prioritization.features import build_features, CONSEQUENCE_WEIGHTS
from src.prioritization.gene_panel import (
    is_decoy_contig,
    is_primary_contig,
    gene_for_position,
    load_panel,
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
                    csq_fields = [x.strip().strip('>"') for x in fmt.split("|")]
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
                    # ClinVar significance: cache's own CLIN_SIG column is empty
                    # in this build; the --custom file appends ClinVar_CLNSIG
                    # as trailing CSQ fields (header lists them, aligned).
                    clin = (d.get("CLIN_SIG") or d.get("ClinVar_CLNSIG") or "").split("&")[0].strip()
                    if clin and clin not in ("-", "not_provided"):
                        out["CLIN_SIG"] = clin
                    out["canonical"] = 1 if d.get("CANONICAL") == "YES" else 0
                    out["mane"] = 1 if d.get("MANE_SELECT") not in ("", None, "-") else 0
                    # Honest rarity: gnomAD AF lives inside CSQ (not top-level
                    # INFO) when VEP-annotated. First non-empty wins.
                    for ak in ("gnomADe_AF", "gnomADg_AF", "MAX_AF", "AF"):
                        v_af = (d.get(ak) or "").split("&")[0].strip()
                        if v_af and v_af != "-":
                            out["gnomad_af"] = v_af
                            break
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
    # Inheritance model for the recessive bonus (recessive|dominant|any).
    # Dominant diseases don't reward multi-het genes; set bonus 0 to disable.
    inheritance = str(config.get("inheritance", "recessive")).lower()
    recessive_w = float(config.get("recessive_bonus", 0.1))

    # Phenotype model (Resnik over public HPO ontology + annotations).
    # None -> neutral 0.5 per gene (pathogenicity drives ranking).
    hpo_terms_all = hpo_terms  # loaded by caller
    gene_hpo_scores = load_phenotype_model(
        config.get("hpo_obo", "data/hpo/hp.obo"),
        config.get("hpo_annotations", "data/hpo/genes_to_phenotype.txt"),
        hpo_terms_all,
    )

    # Disease panel: manual (config gene_panel) or data-driven top-N HPO
    # genes (panel_source: hpo). Swap the config for a new disease.
    panel_genes, panel_intervals = load_panel(config, gene_hpo_scores)
    print(f"[prioritize] Panel: {len(panel_genes)} genes "
          f"(source={config.get('panel_source', 'manual')})")

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

    # Phenotype engine: real Resnik similarity vs the query HPO terms (offline).
    # Produces per-gene phenotype features (train/serve parity via GeneHPO)
    # and the phenotype gate used by candidate generation.
    pheno_for_gene = None
    gate: dict[str, float] = {}
    if hpo_terms:
        try:
            from src.phenotype.hpo_ontology import HPOOntology
            from src.phenotype.gene_hpo import GeneHPO
            from src.ml.candidates import PHENO_GATE_MIN_NORM
            hpo_dir = Path(config.get("hpo_dir", "data/hpo"))
            ont = HPOOntology(hpo_dir / "hp.obo")
            gh = GeneHPO(hpo_dir / "genes_to_phenotype.txt", ont)
            qterms = {t for t in hpo_terms if ont.is_known(t)}
            norm = ont.query_norm_factor(qterms)
            gate = {g: v for g, v in gh.genes_with_similarity(qterms).items()
                    if v > PHENO_GATE_MIN_NORM}
            cache: dict = {}

            def pheno_for_gene(gene):
                if not gene or gene == ".":
                    return None
                blk = cache.get(gene)
                if blk is None:
                    blk = gh.gene_similarity(qterms, gene, norm)
                    cache[gene] = blk
                return blk
            print(f"[prioritize] phenotype engine: {len(qterms)} HPO terms -> "
                  f"{len(gate)} genes in retrieval gate")
        except FileNotFoundError as e:
            print(f"[prioritize] WARNING: HPO index missing ({e}); "
                  "phenotype falls back to neutral 0.5")

    PHENO_ZERO = {"pheno_resnik": 0.0, "pheno_bma": 0.0, "pheno_jaccard": 0.0,
                  "pheno_terms_matched": 0.0, "pheno_excluded_overlap": 0.0,
                  "gene_hpo_count": 0.0, "gene_disease_count": 0.0}

    with opener(vcf_annotated, 'rt') as f:
        for line in tqdm(f, desc="Scoring variants", unit=" var", total=total):
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            chrom, pos, vid, ref, alt, qual, filt, info_str = fields[:8]
            # FORMAT/SAMPLE (col 9/10) carry GT:AD:DP:GQ:PL — het detection,
            # allele balance (mosaic signal) and depth for the ranker.
            gt, ad_ref, ad_alt, dp_samp = "", -1.0, -1.0, -1.0
            if len(fields) >= 10:
                fmt_keys = fields[8].split(":")
                fmt_vals = fields[9].split(":")
                def _fmt(key: str) -> str:
                    return fmt_vals[fmt_keys.index(key)] if key in fmt_keys and fmt_keys.index(key) < len(fmt_vals) else ""
                gt = _fmt("GT")
                try:
                    ad_parts = _fmt("AD").split(",")
                    ad_ref, ad_alt = float(ad_parts[0]), float(ad_parts[1])
                except (ValueError, IndexError):
                    pass
                try:
                    dp_samp = float(_fmt("DP"))
                except ValueError:
                    pass
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
                    gene = gene_for_position(chrom, int(pos), panel_intervals)
                    gene_source = "positional" if gene else "none"

                # Phenotype score: Resnik ontology similarity when available
                # (continuous 0..1 from public HPO annotations), neutral 0.5
                # for unannotated genes. Explicit MVA panel boost kept as a
                # documented prior on top.
                if gene_hpo_scores is not None and gene and gene in gene_hpo_scores:
                    pheno_score = gene_hpo_scores[gene]
                else:
                    pheno_score = hpo_overlap_score(hpo_terms, gene_hpo_map=None, gene=gene)
                if gene in panel_genes:
                    pheno_score = max(pheno_score, panel_genes[gene])

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
                    "ID": vid,
                    "SYMBOL": gene or ".",
                    "gene_source": gene_source,
                    "GT": gt,
                    "AD_REF": ad_ref,
                    "AD_ALT": ad_alt,
                    "DP": dp_samp if dp_samp >= 0 else variant_dict.get("DP", ""),
                    "QD": variant_dict.get("QD", ""),
                    "MQ": variant_dict.get("MQ", ""),
                    "IMPACT": variant_dict.get("IMPACT", ""),
                    "EXON": variant_dict.get("EXON", ""),
                    "Consequence": feats.get("consequence", ""),
                    "CLIN_SIG": variant_dict.get("CLIN_SIG", ""),
                    "gnomad_af": feats.get("af"),
                    "CADD_PHRED": variant_dict.get("CADD_PHRED", ""),
                    "canonical": variant_dict.get("canonical", 0),
                    "mane": variant_dict.get("mane", 0),
                    "cadd_norm": feats["cadd_norm"],
                    "clinvar_score": feats["clinvar_score"],
                    "clinvar_score": feats["clinvar_score"],
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

    # Inheritance-aware pair bonus (continuous, modest). For recessive
    # diseases, a panel gene carrying >=2 rare het alleles gets +bonus
    # (compound-het pattern). Disabled unless inheritance == "recessive".
    # This is a curation flag, not a bucket: ranking stays continuous.
    het_gts = {"0/1", "1/0", "0|1", "1|0"}
    rare_hets = df[
        df["GT"].isin(het_gts)
        & (df["gnomad_af"].isna() | (df["gnomad_af"] < af_thresh))
    ] if "GT" in df.columns else df.iloc[0:0]
    gene_counts = rare_hets["SYMBOL"].value_counts().to_dict()
    if inheritance == "recessive":
        compound_genes = {g for g, n in gene_counts.items() if g in panel_genes and n >= 2}
    else:
        compound_genes = set()
    if compound_genes:
        print(f"[prioritize] Compound-het candidates (>=2 rare hets): {sorted(compound_genes)}")
    df["compound_het_candidate"] = df["SYMBOL"].isin(compound_genes)
    df["recessive_bonus"] = df["compound_het_candidate"].map({True: recessive_w, False: 0.0})
    df["score"] = df["score"] + df["recessive_bonus"]
    # Per-gene rare-het burden (recessive signal, also an ML feature)
    df["gene_het_count"] = df["SYMBOL"].map(rare_hets["SYMBOL"].value_counts().to_dict()).fillna(0).astype(int)

    # ---- Candidate generation (retrieval stage): shared rules with training
    # (src/ml/candidates.py). `gate` = genes whose normalized max-Resnik to
    # the query HPO terms exceeds PHENO_GATE_MIN_NORM (built by the phenotype
    # engine above); the disease panel is an optional domain prior on top and
    # is never a model feature (ranker stays disease-agnostic).
    from src.ml.candidates import is_candidate as _is_candidate
    df["is_candidate"] = [
        _is_candidate(af, cons, clin, filt, is_primary_contig(chrom), gene,
                      gate, set(panel_genes), af_thresh)
        for af, cons, clin, filt, chrom, gene in zip(
            df["gnomad_af"], df["Consequence"], df["CLIN_SIG"].fillna(""),
            df["FILTER"], df["CHROM"], df["SYMBOL"])
    ]
    n_cand = int(df["is_candidate"].sum())
    print(f"[prioritize] Candidates (structural, AF-free): {n_cand} / {len(df)}")

    # ---- gnomAD AF + CADD come from side-files: the --af_* VEP flags cost
    # ~7.6x the structural pass and CADD-as-plugin ~7x more (measured), so VEP
    # skips both and they are merged here for candidates only. AF then narrows
    # the candidate set (ClinVar P/LP is exempt from the rarity rule, exactly
    # as in src/ml/candidates.py); if the AF side-file is absent every variant
    # is treated as rare — identical to training when train.af is missing.
    from src.ml.lookup import load_af, cadd_lookup, vkey as _vkey
    from src.ml.candidates import is_pathogenic_clinsig as _clin_path
    af_map = load_af(Path(config.get("af_vcf", "data/interim/full.af.vcf.gz")))
    cand_idx = df.index[df["is_candidate"]]
    if len(cand_idx):
        keys = [_vkey(c, int(p), r, a) for c, p, r, a in
                zip(df.loc[cand_idx, "CHROM"], df.loc[cand_idx, "POS"],
                    df.loc[cand_idx, "REF"], df.loc[cand_idx, "ALT"])]
        # fill AF from the side-file (CSQ has no AF columns in the fast pass)
        missing = df.loc[cand_idx, "gnomad_af"].isna().to_numpy()
        for j, i in enumerate(cand_idx):
            if missing[j] and keys[j] in af_map:
                df.at[i, "gnomad_af"] = af_map[keys[j]]
        # rarity gate (parity with the training-time retrieval mask)
        af = pd.to_numeric(df.loc[cand_idx, "gnomad_af"], errors="coerce")
        clin = df.loc[cand_idx, "CLIN_SIG"].fillna("").map(_clin_path)
        rare = af.isna() | (af < af_thresh)
        keep = (rare | clin).to_numpy()
        dropped = int((~keep).sum())
        df.loc[cand_idx[~keep], "is_candidate"] = False
        cand_idx = cand_idx[keep]
        keys = [k for k, kk in zip(keys, keep) if kk]
        print(f"[prioritize] AF side-file: {len(af_map)} entries | "
              f"common candidates dropped: {dropped} | "
              f"candidates after rarity: {len(cand_idx)}")

        # CADD for candidates (single tabix pass per CADD file)
        cadd = cadd_lookup([(c, int(p), r, a) for c, p, r, a in
                            zip(df.loc[cand_idx, "CHROM"], df.loc[cand_idx, "POS"],
                                df.loc[cand_idx, "REF"], df.loc[cand_idx, "ALT"])])
        for i, k in zip(cand_idx, keys):
            v = cadd.get(k)
            if v is not None:
                df.at[i, "CADD_PHRED"] = v
        print(f"[prioritize] CADD side-file: {len(cadd)} entries "
              f"({len(cand_idx)} candidates)")
    print(f"[prioritize] Candidates (final): {len(cand_idx)} / {len(df)}")

    # ---- Ranking stage: local LightGBM LambdaRank, STANDALONE (no blend).
    # --model models/ranker.txt -> calibrated probability that the variant is
    # causal (isotonic fitted on held-out val queries at train time).
    # Non-candidates get a floor score below every candidate (they were
    # already removed by the high-recall retrieval stage).
    model_path = config.get("model", None)
    cand_idx = df.index[df["is_candidate"]]
    if model_path and len(cand_idx):
        import lightgbm as lgb
        import json as _json
        import numpy as _np
        import pickle as _pk
        from src.ml.rank_features import FEATURES, derive_frame
        from src.ml.lookup import (load_loeuf, load_gene_path_counts,
                                   am_lookup, vkey as _vkey)
        mp = Path(model_path)
        if not mp.exists():
            raise RuntimeError(
                f"Ranker model not found: {model_path!r}. "
                "Train it with src/ml/train_ranker.py")
        feat_path = mp.parent / "features.json"
        expected = _json.loads(feat_path.read_text()) if feat_path.exists() else FEATURES
        # Ablation models train on a subset of FEATURES (e.g. no clinvar_score);
        # the subset must keep FEATURES order so column alignment is guaranteed.
        if not set(expected).issubset(FEATURES) or \
                expected != [c for c in FEATURES if c in set(expected)]:
            raise RuntimeError(
                f"Feature mismatch: model trained on {expected}, code has {FEATURES}")
        booster = lgb.Booster(model_file=str(mp))

        cand = df.loc[cand_idx].copy()
        # phenotype feature block (same GeneHPO engine as training)
        PHENO_COLS = ("pheno_resnik", "pheno_bma", "pheno_jaccard",
                      "pheno_terms_matched", "pheno_excluded_overlap",
                      "gene_hpo_count", "gene_disease_count")
        if pheno_for_gene is not None:
            genes = set(cand["SYMBOL"].dropna()) - {".", ""}
            blk = {g: (pheno_for_gene(g) or PHENO_ZERO) for g in genes}
            for col in PHENO_COLS:
                cand[col] = [blk.get(g, PHENO_ZERO)[col] for g in cand["SYMBOL"]]
        else:
            for col in PHENO_COLS:
                cand[col] = 0.0
        # gene-level constraint + ClinVar burden (shared lookups w/ training)
        loeuf_map = load_loeuf()
        path_counts = load_gene_path_counts()
        cand["gene_loeuf"] = cand["SYMBOL"].map(
            lambda g: loeuf_map.get(g, (None, None))[0])
        cand["gene_pli"] = cand["SYMBOL"].map(
            lambda g: loeuf_map.get(g, (None, None))[1])
        cand["gene_clinvar_path_count"] = (
            cand["SYMBOL"].map(path_counts).fillna(0).astype(int))
        # AlphaMissense for missense candidates (one tabix pass)
        miss = cand[cand["Consequence"].fillna("").str.contains("missense")]
        am = am_lookup([(r.CHROM, int(r.POS), r.REF, r.ALT)
                        for r in miss.itertuples()])
        cand["am_pathogenicity"] = [
            am.get(_vkey(c, p, r, a), (None, ""))[0]
            for c, p, r, a in cand[["CHROM", "POS", "REF", "ALT"]].itertuples(index=False)]

        X = derive_frame(cand)[expected].to_numpy(dtype=_np.float32)
        raw = booster.predict(X)
        calib_path = mp.parent / "calib.pkl"
        if calib_path.exists():
            iso = _pk.loads(calib_path.read_bytes())
            cal = iso.predict(raw)
        else:
            print("[prioritize] WARNING: models/calib.pkl missing — raw scores")
            cal = raw
        floor = float(cal.min()) * 0.5 if len(cal) else 0.0
        df["heuristic_score"] = df["score"]
        df["score"] = floor
        df.loc[cand_idx, "score"] = cal
        df.loc[cand_idx, "ml_score"] = raw
        df["ml_score"] = df["ml_score"].fillna(-1.0)
        print(f"[prioritize] ML-ranked {len(cand)} candidates with {mp.name} "
              f"(standalone, isotonic-calibrated)")
    elif model_path:
        print("[prioritize] WARNING: model given but zero candidates — heuristic only")

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

    # Calibrate for F-max (heuristic path only — the ML path is already
    # isotonic-calibrated on held-out val queries at train time)
    if not (model_path and len(cand_idx)):
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
    parser.add_argument("--model", default=None, help="LightGBM ranker model (models/ranker.txt) — local, optional")
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
    if args.model:
        cfg["model"] = args.model

    ranked = prioritize(args.vcf, hpo_terms, config=cfg)

    # Write submission
    to_submission_format(ranked, args.output)
    print(f"[main] Done. Top 5:\n{ranked.head(5).to_string(index=False)}")


if __name__ == "__main__":
    main()
