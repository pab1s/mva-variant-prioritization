"""Tests for feature derivation parity + retrieval rules + LTR metrics."""
import numpy as np
import pandas as pd

from src.ml.candidates import (PHENO_GATE_MIN_NORM, is_candidate,
                               retrieval_mask)
from src.ml.eval_metrics import (calibration_report, fmax_score, query_metrics,
                                 summarize)
from src.ml.rank_features import FEATURES, derive_frame, row_to_vector


def _raw_row(**over):
    base = dict(gnomad_af=None, Consequence="missense_variant", IMPACT="MODERATE",
                CADD_PHRED=25.0, CLIN_SIG="", EXON="3/10", canonical=1, mane=1,
                am_pathogenicity=0.9, gene_loeuf=0.3, gene_pli=0.95,
                gene_clinvar_path_count=50,
                pheno_resnik=0.8, pheno_bma=0.6, pheno_jaccard=0.2,
                pheno_terms_matched=0.4, pheno_excluded_overlap=0.0,
                gene_hpo_count=0.3, gene_disease_count=0.1)
    base.update(over)
    return base


def test_derive_frame_features_and_order():
    df = pd.DataFrame([_raw_row(), _raw_row(gnomad_af=0.5, CADD_PHRED=None,
                                            Consequence="intron_variant")])
    X = derive_frame(df)
    assert list(X.columns) == FEATURES
    # row 0: rare, high CADD, missense, constrained gene, phenotype match
    assert X.loc[0, "rare_score"] == 1.0
    assert X.loc[0, "cadd_norm"] == 25.0 / 30.0
    assert X.loc[0, "is_missense"] == 1.0
    assert X.loc[0, "loeuf_norm"] == 1.0 - 0.3 / 2.0
    # row 1: common variant, missing CADD, intronic
    assert X.loc[1, "rare_score"] == 0.05
    assert X.loc[1, "cadd_missing"] == 1.0
    assert X.loc[1, "is_missense"] == 0.0


def test_row_to_vector_matches_frame():
    row = _raw_row()
    v = row_to_vector(row)
    X = derive_frame(pd.DataFrame([row])).iloc[0].to_numpy()
    assert np.allclose(np.array(v, dtype=float), X)


def test_clinvar_score_polarity():
    df = pd.DataFrame([
        _raw_row(CLIN_SIG="pathogenic"),
        _raw_row(CLIN_SIG="benign"),
        _raw_row(CLIN_SIG=""),
    ])
    X = derive_frame(df)
    assert list(X["clinvar_score"]) == [1.0, 0.1, 0.5]


def test_retrieval_rules_scalar_and_vector_agree():
    # (af, cons, clin, filt, gene, primary, expected_scalar, expected_vector)
    # vector path has no contig info (public variants) — decoy handled by
    # is_primary_contig in the scalar/inference path only.
    rows = [
        (0.001, "missense_variant", "", "PASS", "G1", True, True, True),
        (0.30, "intron_variant", "", "PASS", "G2", True, False, False),
        (0.30, "intron_variant", "pathogenic", "PASS", "G2", True, True, True),
        (0.0005, "intron_variant", "", "PASS", "GATE", True, True, True),
        (0.0005, "intron_variant", "", "PASS", "G3", True, False, False),
        (0.0005, "frameshift_variant", "", "PASS", "G1", False, False, True),
        (0.001, "missense_variant", "", "LowQual", "G1", True, False, True),  # vector ignores FILTER
        (0.001, "missense_variant", "pathogenic", "LowQual", "G1", True, True, True),
    ]
    df = pd.DataFrame([(r[0], r[1], r[2], r[3], r[4]) for r in rows],
                      columns=["gnomad_af", "Consequence", "CLIN_SIG", "FILTER", "SYMBOL"])
    mask = retrieval_mask(df, pheno_gate={"GATE"}, panel=set())
    assert list(mask) == [r[7] for r in rows]

    for af, cons, clin, filt, gene, primary, expected, _ in rows:
        scalar = is_candidate(af, cons, clin, filt, primary, gene, {"GATE"}, set())
        assert scalar == expected, f"scalar mismatch for {(af, cons, clin, filt, gene)}"


def test_gate_constant():
    assert 0.0 < PHENO_GATE_MIN_NORM < 1.0


def test_query_metrics_perfect_ranking():
    df = pd.DataFrame({"label": [3, 1, 0, 0], "score": [0.9, 0.5, 0.2, 0.1]})
    m = query_metrics(df)
    assert m["ndcg@10"] == 1.0 and m["recall@1"] == 1.0
    assert m["mrr"] == 1.0 and m["rank_points"] == 100.0


def test_query_metrics_bad_ranking():
    df = pd.DataFrame({"label": [0, 0, 3, 0], "score": [0.9, 0.5, 0.2, 0.1]})
    m = query_metrics(df)
    assert m["recall@1"] == 0.0 and m["mrr"] == 1 / 3
    assert m["rank_points"] < 100  # partial credit at rank 3


def test_summarize_two_queries():
    df = pd.DataFrame({
        "query_id": ["q1", "q1", "q2", "q2"],
        "label": [3, 0, 0, 3],
        "score": [0.9, 0.1, 0.9, 0.8],
    })
    s = summarize(df)
    assert s["n_queries"] == 2
    assert s["recall@1"] == 0.5  # q2's causal ranked second


def test_fmax_and_calibration():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1, 0.05])
    f1, thr = fmax_score(y, p)
    assert f1 == 1.0
    rep = calibration_report(y, p)
    assert rep["fmax"] == 1.0 and 0 <= rep["ece"] <= 1
