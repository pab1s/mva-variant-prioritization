"""Ranking and submission formatting."""
import pandas as pd
from pathlib import Path


def rank_variants(scored_df: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """Sort by single calibrated score, descending (1-based ranks).

    Standard continuous ranking: no buckets. Ties broken by
    pathogenicity_score, then quality_score, then rare_score.
    Returns DataFrame with RANK column added, sorted descending.
    """
    cols = scored_df.columns.tolist()
    sort_by = [score_col]
    ascending = [False]
    for tie in ("pathogenicity_score", "quality_score", "rare_score"):
        if tie in cols:
            sort_by.append(tie)
            ascending.append(False)
    # pandas sort
    ranked = scored_df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
    ranked["RANK"] = ranked.index + 1
    # ensure SCORE col is score_col
    if score_col != "SCORE":
        ranked["SCORE"] = ranked[score_col]
    return ranked


def to_submission_format(ranked_df: pd.DataFrame, out_path: str) -> None:
    """Write Track 1 submission CSV (CHROM, POS, REF, ALT, RANK, SCORE, optional extras).

    Required by hackathon: ranked variants with confidence scores for Rank Points + F-max.
    We output canonical 6 columns plus optional INFO for debugging.

    Args:
        ranked_df: DataFrame with at least CHROM, POS, REF, ALT, SCORE/RANK
        out_path: where to write CSV
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Ensure required cols
    required = ["CHROM", "POS", "REF", "ALT"]
    for c in required:
        if c not in ranked_df.columns:
            raise ValueError(f"Missing required column {c} in ranked_df. Have {ranked_df.columns.tolist()}")

    # Normalize SCORE 0-1
    df = ranked_df.copy()
    if "SCORE" not in df.columns:
        if "score" in df.columns:
            df["SCORE"] = df["score"]
        else:
            df["SCORE"] = 1.0 / df["RANK"]  # fallback

    # Clip and calibrate (sigmoid-ish scaling already in score, just ensure 0-1)
    df["SCORE"] = df["SCORE"].clip(0, 1)

    # Select output columns (keep extras if wanted for inspection)
    base_cols = ["CHROM", "POS", "REF", "ALT", "RANK", "SCORE"]
    extra_cols = [c for c in ["SYMBOL", "gene_source", "compound_het_candidate", "is_candidate", "ml_score", "GT", "Gene", "Consequence", "gnomad_af", "CADD_PHRED", "FILTER", "score", "pathogenicity_score", "phenotype_score", "recessive_bonus", "decoy_penalty", "rare_score", "consequence_weight", "quality_score", "cadd_norm", "clinvar_score", "QD", "MQ", "DP", "AD_REF", "AD_ALT", "IMPACT", "EXON", "ID", "gene_het_count"] if c in df.columns]
    out_cols = base_cols + extra_cols

    df_sorted = df.sort_values("RANK")
    df_sorted[out_cols].to_csv(out, index=False)
    print(f"[rank] Wrote {len(df_sorted)} variants to {out} with columns {out_cols}")


def calibrate_scores(df: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """Optional: Platt-like calibration for F-max (well-calibrated_probs).

    Baseline: scores are already 0-1, but we ensure they span range for F-max thresholding.
    """
    df = df.copy()
    if score_col in df.columns:
        # min-max scale to [0.01, 0.99] to avoid extreme 0/1 that hurts F-max
        s = df[score_col]
        s_min, s_max = s.min(), s.max()
        if s_max > s_min:
            df[score_col] = 0.01 + 0.98 * (s - s_min) / (s_max - s_min)
    return df
