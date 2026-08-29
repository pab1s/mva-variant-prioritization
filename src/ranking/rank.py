"""Ranking and submission formatting."""
import pandas as pd

def rank_variants(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Sort by final score and assign ranks."""
    return scored_df.sort_values("score", ascending=False).reset_index(drop=True)

def to_submission_format(ranked_df: pd.DataFrame, out_path: str) -> None:
    """Write Track 1 submission CSV (CHROM, POS, REF, ALT, RANK, SCORE)."""
    raise NotImplementedError
