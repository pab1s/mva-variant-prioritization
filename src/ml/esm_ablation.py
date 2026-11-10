"""LightGBM ablation for the ESM-2 feature block (fast screening).

Runs several block sizes on the same gene-disjoint split and reports test
metrics plus a paired bootstrap by query against the no-ESM baseline.

Run:
  .venv/bin/python -m src.ml.esm_ablation
"""
from __future__ import annotations

import json
import time
from argparse import Namespace
from pathlib import Path

import pandas as pd

from src.ml.eval_metrics import calibration_report, summarize
from src.ml.train_esm_ranker import (
    _bootstrap,
    build_esm_features,
    fit_predict,
    load_split,
    merge_esm,
    vkey,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = [
    ("base", "none", 0),
    ("score", "score", 0),
    ("delta16", "delta", 16),
    ("delta48", "delta", 48),
    ("delta96", "delta", 96),
    ("both48", "both", 48),
]


def main() -> None:
    tr, Xtr0, ytr, qtr = load_split("train")
    va, Xva0, yva, qva = load_split("val")
    te, Xte0, yte, qte = load_split("test")
    print(f"[data] train={len(tr)} val={len(va)} test={len(te)}")

    from src.ml.esm_features import DEFAULT_OUT
    all_vkeys = set(pd.read_csv(f"{DEFAULT_OUT}_index.csv.gz",
                                usecols=["vkey"])["vkey"].astype(str))
    train_vkeys = set(vkey(tr)) & all_vkeys
    args = Namespace(rounds=2000, lr=0.05, l2=3.0, leaves=63)

    results = {}
    baseline_scores = None
    for name, block, n_comp in CONFIGS:
        t0 = time.time()
        esm = (build_esm_features(block, n_comp, train_vkeys)
               if block != "none" else None)
        Xtr = merge_esm(Xtr0, tr, esm)
        Xva = merge_esm(Xva0, va, esm)
        Xte = merge_esm(Xte0, te, esm)
        from src.ml.train_esm_ranker import BASE_FEATS
        feat_names = BASE_FEATS + (list(esm.columns) if esm is not None else [])
        va_raw, te_raw, best_iter, secs, importance, _ = fit_predict(
            "lightgbm", Xtr, ytr, qtr, Xva, yva, qva, Xte, args, feat_names)
        metrics = summarize(pd.DataFrame({
            "query_id": te["query_id"].astype(str), "label": yte,
            "score": te_raw}))
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            va_raw, (yva > 0).astype(int))
        calib = calibration_report((yte > 0).astype(int), iso.predict(te_raw))
        if baseline_scores is None:
            baseline_scores = te_raw
        results[name] = {
            "esm": block, "n_components": n_comp,
            "n_features": int(Xtr.shape[1]), "best_iteration": best_iter,
            "seconds": round(secs, 1), "importance": importance,
            "test": {**metrics, **{f"calib_{k}": v for k, v in calib.items()}},
            "bootstrap_vs_base": _bootstrap(
                baseline_scores, te_raw, yte, qte.astype(str)),
        }
        print(f"[{name:8s}] ndcg@10={metrics['ndcg@10']:.4f} "
              f"r@1={metrics['recall@1']:.4f} "
              f"rp={metrics['rank_points']:.2f} "
              f"brier={calib['brier']:.4f} ({time.time() - t0:.0f}s)",
              flush=True)

    out = Path("reports/esm_ablation.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
