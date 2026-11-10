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
    gene_clinvar_path_count, gene_oe_mis, gene_mis_z, gene_syn_z,
    gene_oe_lof, gene_cds_length, gene_num_exons, gene_segdup_prop,
    gene_lcr_prop, gene_inheritance_ar, gene_inheritance_ad,
    gene_inheritance_xl, gene_hpo_ic_mean, gene_hpo_ic_max,
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

# Experimental gene-intrinsic block (not in the deployed schema).
# The 2026-11 ablation showed that adding any of these groups degrades the
# held-out test ranking (see reports/gene_feature_ablation.md). They remain
# available for reproducible experiments via `include_gene_features=True`.
GENE_INTRINSIC_FEATURES = [
    "gene_oe_mis",
    "gene_mis_z",
    "gene_syn_z",
    "gene_oe_lof",
    "gene_cds_length_log",
    "gene_num_exons_log",
    "gene_segdup_prop",
    "gene_lcr_prop",
    "gene_inheritance_ar",
    "gene_inheritance_ad",
    "gene_inheritance_xl",
    "gene_hpo_ic_mean",
    "gene_hpo_ic_max",
    "is_lof_x_loeuf",
    "is_missense_x_mis_z",
    "cadd_x_loeuf",
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


def _numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    """Column -> numeric Series with a neutral default when absent/NaN."""
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def derive_frame(df: pd.DataFrame,
                 include_gene_features: bool = False) -> pd.DataFrame:
    """Raw annotation columns -> FEATURES columns (vectorized, defensive).

    The deployed schema is `FEATURES` (23 columns). Experimental gene-intrinsic
    features are appended only when `include_gene_features=True`; they are not
    part of the deployed model for the reason documented in
    reports/gene_feature_ablation.md.
    """
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

    # ---- gene-level intrinsic features (gnomAD constraint + structure)
    out["gene_oe_mis"] = _numeric(df, "gene_oe_mis", 0.5)
    out["gene_mis_z"] = _numeric(df, "gene_mis_z", 0.0).clip(-10.0, 10.0)
    out["gene_syn_z"] = _numeric(df, "gene_syn_z", 0.0).clip(-10.0, 10.0)
    out["gene_oe_lof"] = _numeric(df, "gene_oe_lof", 0.5)
    out["gene_cds_length_log"] = np.log1p(
        _numeric(df, "gene_cds_length", 0.0).clip(lower=0.0))
    out["gene_num_exons_log"] = np.log1p(
        _numeric(df, "gene_num_exons", 0.0).clip(lower=0.0))
    out["gene_segdup_prop"] = _numeric(
        df, "gene_segdup_prop", 0.0).clip(0.0, 1.0)
    out["gene_lcr_prop"] = _numeric(
        df, "gene_lcr_prop", 0.0).clip(0.0, 1.0)

    # ---- gene-level inheritance and HPO specificity (HPOA)
    for col in ("gene_inheritance_ar", "gene_inheritance_ad",
                "gene_inheritance_xl"):
        out[col] = _numeric(df, col, 0.0).clip(0.0, 1.0)
    for col in ("gene_hpo_ic_mean", "gene_hpo_ic_max"):
        out[col] = _numeric(df, col, 0.0).clip(lower=0.0)

    # ---- interactions: variant consequence x gene constraint
    out["is_lof_x_loeuf"] = out["is_lof"] * out["loeuf_norm"]
    out["is_missense_x_mis_z"] = out["is_missense"] * out["gene_mis_z"]
    out["cadd_x_loeuf"] = out["cadd_norm"] * out["loeuf_norm"]

    cols = FEATURES + (GENE_INTRINSIC_FEATURES if include_gene_features else [])
    return out[cols]


def row_to_vector(row) -> list[float]:
    """Single-row convenience wrapper (dict or Series) -> feature vector."""
    df = derive_frame(pd.DataFrame([dict(row)]))
    return df.iloc[0].to_numpy(dtype=float).tolist()
