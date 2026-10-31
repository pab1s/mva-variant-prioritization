"""Ranker feature schema — the single source of truth for train/serve parity.

`derive_frame(df)` turns a raw annotation row (VEP CSQ fields + lookups +
per-query phenotype features) into the fixed numeric feature vector the
LightGBM ranker consumes. It is called BOTH:

  - at training time on dataset rows produced by src/ml/make_dataset.py, and
  - at inference time on prioritize() rows for candidate variants.

so any change here applies identically to training and inference. Raw keys
required per row (all optional/defensive, missing -> neutral default):

    gnomad_af, Consequence, IMPACT, CADD_PHRED, CLIN_SIG, EXON,
    canonical, mane, am_pathogenicity, gene_loeuf, gene_pli,
    gene_clinvar_path_count,
    pheno_resnik, pheno_bma, pheno_jaccard, pheno_terms_matched,
    pheno_excluded_overlap, gene_hpo_count, gene_disease_count

Deliberately EXCLUDED (no public counterpart at training time; live only in
the separate heuristic system): is_het, ab_dev, dp_norm, quality_score,
qd/mq_norm, gene_het_count, compound_het_candidate, filt_pass.
No gene lists (MVA panel etc.) are features — the model is disease-agnostic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.prioritization.features import CONSEQUENCE_WEIGHTS

FEATURES = [
    # rarity / consequence / pathogenicity (static)
    "rare_score",
    "consequence_weight",
    "is_lof",
    "is_missense",
    "is_splice",
    "impact_ord",
    "cadd_norm",
    "cadd_missing",
    "clinvar_score",
    "canonical",
    "mane",
    "am_norm",
    "loeuf_norm",
    "pli_norm",
    "gene_path_count",
    "exon_frac",
    # phenotype / gene-level (query-dependent)
    "pheno_resnik",
    "pheno_bma",
    "pheno_jaccard",
    "pheno_terms_matched",
    "pheno_excluded_overlap",
    "gene_hpo_count",
    "gene_disease_count",
]

LOF_CONSEQUENCES = frozenset({
    "stop_gained", "frameshift_variant", "splice_acceptor_variant",
    "splice_donor_variant", "transcript_ablation", "start_lost",
})
IMPACT_ORD = {"HIGH": 3.0, "MODERATE": 2.0, "LOW": 1.0, "MODIFIER": 0.0}


def _f(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        return default if v != v else v  # NaN -> default
    except (TypeError, ValueError):
        return default


def derive_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Raw annotation columns -> FEATURES columns (vectorized, defensive)."""
    out = pd.DataFrame(index=df.index)

    af = pd.to_numeric(df.get("gnomad_af"), errors="coerce")
    out["rare_score"] = np.where(
        af.isna(), 1.0,
        np.where(af < 0.001, 1.0,
                 np.where(af < 0.01, 0.9,
                          np.where(af < 0.05, 0.6,
                                   np.where(af < 0.5, 0.3, 0.05)))))

    cons = df.get("Consequence", pd.Series("", index=df.index)).fillna("").astype(str)
    cw = cons.map(lambda c: max((CONSEQUENCE_WEIGHTS.get(k, 0.1)
                                 for k in c.split("&")), default=0.1))
    out["consequence_weight"] = cw
    out["is_lof"] = cons.map(
        lambda c: float(any(k in LOF_CONSEQUENCES for k in c.split("&"))))
    out["is_missense"] = cons.str.contains("missense_variant", na=False).astype(float)
    out["is_splice"] = cons.str.contains("splice", na=False).astype(float)

    impact = df.get("IMPACT", pd.Series("", index=df.index)).fillna("").str.upper()
    out["impact_ord"] = impact.map(lambda i: IMPACT_ORD.get(i, 0.0))

    cadd = pd.to_numeric(df.get("CADD_PHRED"), errors="coerce")
    out["cadd_norm"] = (cadd / 30.0).clip(0, 1).fillna(0.3)
    out["cadd_missing"] = cadd.isna().astype(float)

    clin = df.get("CLIN_SIG", pd.Series("", index=df.index)).fillna("").str.lower()
    out["clinvar_score"] = np.where(
        clin.str.contains("pathogenic") & ~clin.str.contains("benign"), 1.0,
        np.where(clin.str.contains("benign"), 0.1, 0.5))

    out["canonical"] = pd.to_numeric(df.get("canonical"), errors="coerce").fillna(0.0)
    out["mane"] = pd.to_numeric(df.get("mane"), errors="coerce").fillna(0.0)

    am = pd.to_numeric(df.get("am_pathogenicity"), errors="coerce")
    out["am_norm"] = am.fillna(0.5)

    loeuf = pd.to_numeric(df.get("gene_loeuf"), errors="coerce")
    out["loeuf_norm"] = (1.0 - (loeuf / 2.0).clip(0, 1)).fillna(0.5)
    pli = pd.to_numeric(df.get("gene_pli"), errors="coerce")
    out["pli_norm"] = pli.fillna(0.5)

    gpc = pd.to_numeric(df.get("gene_clinvar_path_count"), errors="coerce").fillna(0)
    out["gene_path_count"] = (np.log1p(gpc) / np.log1p(1000)).clip(0, 1)

    exon = df.get("EXON", pd.Series("", index=df.index)).fillna("").astype(str)
    def _exon_frac(e: str) -> float:
        try:
            n, m = e.split("/")
            return float(n) / float(m)
        except (ValueError, ZeroDivisionError, AttributeError):
            return -1.0
    out["exon_frac"] = exon.map(_exon_frac)

    for col in ("pheno_resnik", "pheno_bma", "pheno_jaccard",
                "pheno_terms_matched", "pheno_excluded_overlap",
                "gene_hpo_count", "gene_disease_count"):
        out[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)

    return out[FEATURES]


def row_to_vector(row) -> list[float]:
    """Single-row convenience wrapper (dict or Series) -> feature vector."""
    df = derive_frame(pd.DataFrame([dict(row)]))
    return df.iloc[0].to_numpy(dtype=float).tolist()
