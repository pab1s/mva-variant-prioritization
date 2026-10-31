"""Learning-to-rank evaluation metrics (offline, no deps beyond numpy).

Shared by src/ml/train_ranker.py (train-time val metrics) and
src/ml/evaluate.py (test + baselines + ablations).

Metric definitions
------------------
- NDCG@k: graded gains (2^y - 1 with y = label 0..3).
- Recall@k: 1 if any *causal-ish* variant (label >= 2) lands in top k.
- MRR: reciprocal rank of the highest-graded variant.
- Rank Points (approx): src/ranking/metrics.rank_points applied to the best
  grade-3 (phenopacket causal) variant; 100 at rank 1, partial to rank 10.
- ECE / Brier / F-max: calibration on pooled (query, variant) probabilities
  after isotonic calibration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _dcg(relevances: np.ndarray) -> float:
    gains = (2.0 ** relevances) - 1.0
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return float((gains / discounts).sum())


def query_metrics(group: pd.DataFrame, ks=(1, 3, 10)) -> dict[str, float]:
    """Metrics for one query. `group` has cols [label, score], any order."""
    g = group.sort_values("score", ascending=False)
    y = g["label"].to_numpy(dtype=float)
    ideal = np.sort(y)[::-1]
    out: dict[str, float] = {}
    idcg = _dcg(ideal)
    for k in ks:
        out[f"ndcg@{k}"] = _dcg(y[:k]) / idcg if idcg > 0 else 0.0
        out[f"recall@{k}"] = float((y[:k] >= 2).any())
    best_pos = np.argmax(y >= 2) if (y >= 2).any() else None
    out["mrr"] = 1.0 / (best_pos + 1) if best_pos is not None else 0.0
    # approx Rank Points on the best causal (grade-3) variant
    if (y >= 3).any():
        r = int(np.argmax(y >= 3)) + 1
        out["rank_points"] = 100.0 if r == 1 else (100.0 * (11 - r) / 10 if r <= 10 else 0.0)
    else:
        out["rank_points"] = np.nan
    return out


def summarize(df: pd.DataFrame, group_col: str = "query_id") -> dict[str, float]:
    """Macro-average per-query metrics over all queries."""
    rows = [query_metrics(g) for _, g in df.groupby(group_col, sort=False)]
    m = pd.DataFrame(rows)
    out = {c: float(m[c].mean()) for c in m.columns}
    out["n_queries"] = float(len(m))
    return out


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray,
                               n_bins: int = 10) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (proba >= lo) & (proba < hi if hi < 1 else proba <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - proba[mask].mean())
    return float(ece)


def fmax_score(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    """Best F1 over score thresholds (micro over all rows). Returns (fmax, thr)."""
    y = (y_true > 0).astype(int)
    thresholds = np.unique(np.quantile(proba, np.linspace(0.5, 1.0, 201)))
    best_f1, best_thr = 0.0, 0.5
    for thr in thresholds:
        pred = proba >= thr
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        if tp == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(thr)
    return best_f1, best_thr


def calibration_report(y_true_binary: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    brier = float(np.mean((proba - y_true_binary) ** 2))
    ece = expected_calibration_error(y_true_binary, proba)
    f1, thr = fmax_score(y_true_binary, proba)
    return {"brier": brier, "ece": ece, "fmax": f1, "fmax_threshold": thr}
