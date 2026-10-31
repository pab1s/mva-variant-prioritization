"""Candidate generation (retrieval stage): cheap high-recall prefilter.

Single source of truth used by BOTH training (src/ml/make_dataset.py builds
each query's pool through these rules) and inference (src/prioritization/
prioritize.py). Recall over causal variants is measured explicitly during
evaluation — anything filtered here can never be recovered downstream.

Rules (union; a variant is a candidate if):

    primary contig
  & ( FILTER PASS  or  ClinVar P/LP )                    # low-qual only if known path
  & ( (rare & consequence_weight >= 0.6)                 # rare + protein-altering/splice/UTR
    | ClinVar P/LP                                        # known pathogenic
    | (rare & (phenotype-gate gene | panel gene)) )       # phenotype retrieval / panel

- The phenotype gate is a UNION arm, never an intersection: a gene with no
  HPO annotation but a severe consequence still passes (protects novel genes).
- The gene panel (e.g. MVA) is an OPTIONAL domain prior passed in as config;
  it is never a model feature, so the ranker itself stays disease-agnostic.
- For public training variants FILTER/QUAL do not exist; callers pass
  filt=PASS (documented in reports/ml_ranking_report.md).
"""
from __future__ import annotations

import pandas as pd

from src.prioritization.features import CONSEQUENCE_WEIGHTS

# normalized max-Resnik threshold for the phenotype gate (query-normalized
# similarity; verified on the MVA query: causal genes ~0.9, random mean ~0.23)
PHENO_GATE_MIN_NORM = 0.3

DEFAULT_AF_THRESH = 0.01


def consequence_weight(cons: str) -> float:
    return max((CONSEQUENCE_WEIGHTS.get(k, 0.1) for k in str(cons).split("&")),
               default=0.1)


def is_pathogenic_clinsig(clinsig: str) -> bool:
    s = str(clinsig or "").lower()
    return "pathogenic" in s and "benign" not in s


def is_candidate(af: float | None, cons: str, clinsig: str, filt: str,
                 is_primary: bool, gene: str | None,
                 pheno_gate: dict[str, float] | set[str],
                 panel: set[str], af_thresh: float = DEFAULT_AF_THRESH) -> bool:
    """Scalar per-variant rule (inference path)."""
    if not is_primary:
        return False
    rare = af is None or af != af or af < af_thresh  # NaN -> unknown -> rare
    clin_path = is_pathogenic_clinsig(clinsig)
    if filt not in ("PASS", ".", "", "None") and not clin_path:
        return False
    if rare and consequence_weight(cons) >= 0.6:
        return True
    if clin_path:
        return True
    if rare and gene and (gene in pheno_gate or gene in panel):
        return True
    return False


def retrieval_components(df: pd.DataFrame, pheno_gate: set[str], panel: set[str],
                         af_thresh: float = DEFAULT_AF_THRESH) -> dict[str, pd.Series]:
    """Per-arm retrieval signals (used to attribute failures to reasons)."""
    af = pd.to_numeric(df["gnomad_af"], errors="coerce")
    return {
        "rare": af.isna() | (af < af_thresh),
        "cw_ok": df["Consequence"].fillna("").map(consequence_weight) >= 0.6,
        "clin_path": df["CLIN_SIG"].fillna("").map(is_pathogenic_clinsig),
        "gate_ok": df["SYMBOL"].fillna("").isin(pheno_gate)
                   | df["SYMBOL"].fillna("").isin(panel),
    }


def retrieval_mask(df: pd.DataFrame, pheno_gate: set[str], panel: set[str],
                   af_thresh: float = DEFAULT_AF_THRESH) -> pd.Series:
    """Vectorized same rule for a pool DataFrame (training path).

    Required columns: gnomad_af (nullable float), Consequence, CLIN_SIG,
    SYMBOL. FILTER is treated as PASS (absent for public variants).
    """
    c = retrieval_components(df, pheno_gate, panel, af_thresh)
    # NOTE: ClinVar P/LP passes regardless of rarity (parity with the scalar
    # inference rule); everything else must be rare.
    return (c["rare"] & (c["cw_ok"] | c["gate_ok"])) | c["clin_path"]
