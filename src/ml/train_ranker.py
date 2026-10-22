"""Train the LightGBM LambdaRank ensemble — fully offline.

Input 1: scored CSV from `prioritize` (must contain FEATURES columns +
         CHROM/POS/REF/ALT/SYMBOL). Produce with:
           python -m src.prioritization.prioritize --vcf ... --hpo ... -o scored.csv
Input 2: labeled CSV with CHROM,POS,REF,ALT,label (1 = causal, 0 = benign)
         [+ optional query column, default SYMBOL]. This is YOUR df.

Output: models/ranker.txt (booster) + models/features.json (schema guard)
        + models/calib.pkl (isotonic calibrator for F-max) + models/metrics.json

Usage:
  python -m src.ml.train_ranker --scored scored.csv --labels labels.csv \
      --outdir models/ --query SYMBOL
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.ml.rank_features import FEATURES, row_to_vector


def load_labeled(scored_path: str, labels_path: str, query_col: str = "SYMBOL"):
    scored = pd.read_csv(scored_path)
    labels = pd.read_csv(labels_path)
    key = ["CHROM", "POS", "REF", "ALT"]
    for k in key + ["label"]:
        if k not in labels.columns:
            raise RuntimeError(f"labels CSV missing required column {k!r} (has {list(labels.columns)})")
    merged = labels.merge(scored, on=key, how="left", suffixes=("", "_scored"))
    missing = merged["SYMBOL"].isna() if "SYMBOL" in merged.columns else None
    if missing is not None and missing.any():
        n = int(missing.sum())
        print(f"[train] WARNING: {n}/{len(merged)} labeled variants not found in scored CSV — dropped")
        merged = merged[~missing].copy()
    if query_col not in merged.columns:
        raise RuntimeError(f"query column {query_col!r} not in merged data")
    mva = set()
    try:
        from src.prioritization.mva_panel import MVA_GENES
        mva = set(MVA_GENES)
    except ImportError:
        pass
    X = np.array([row_to_vector(r, mva) for r in merged.to_dict("records")], dtype=float)
    y = merged["label"].to_numpy(dtype=int)
    q = merged[query_col].astype(str)
    groups = q.groupby(q).size().loc[q.unique()].to_numpy() if len(q) else np.array([len(q)])
    # lightgbm wants group sizes in row order: sort by query
    order = np.argsort(q.to_numpy(), kind="stable")
    return X[order], y[order], q.to_numpy()[order], groups, merged


def main():
    p = argparse.ArgumentParser(description="Train LightGBM LambdaRank (offline)")
    p.add_argument("--scored", required=True, help="prioritize output CSV")
    p.add_argument("--labels", required=True, help="labeled CSV (CHROM,POS,REF,ALT,label)")
    p.add_argument("--outdir", default="models", help="where to write ranker.txt etc.")
    p.add_argument("--query", default="SYMBOL", help="grouping column for lambdarank")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--num-leaves", type=int, default=31)
    p.add_argument("--min-data-in-leaf", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--rounds", type=int, default=200)
    a = p.parse_args()

    X, y, qids, groups, merged = load_labeled(a.scored, a.labels, a.query)
    print(f"[train] {len(y)} labeled rows, {y.sum()} positives, {len(np.unique(qids))} queries, {X.shape[1]} features")
    if y.sum() == 0:
        raise RuntimeError("No positive labels — ranker has nothing to learn")
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 10],
        "learning_rate": a.lr,
        "num_leaves": a.num_leaves,
        "min_data_in_leaf": a.min_data_in_leaf,
        "verbose": -1,
        "seed": a.seed,
    }
    ds = lgb.Dataset(X, label=y, group=groups.tolist())
    booster = lgb.train(params, ds, num_boost_round=a.rounds)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(outdir / "ranker.txt"))
    (outdir / "features.json").write_text(json.dumps(FEATURES))
    # Isotonic calibration on train scores (for F-max thresholds).
    # Small-data honest caveat: refit on your full labeled set when big.
    train_scores = booster.predict(X)
    iso = IsotonicRegression(out_of_bounds="clip").fit(train_scores, y)
    import pickle as _pk
    (outdir / "calib.pkl").write_bytes(_pk.dumps(iso))
    # Report in-sample NDCG (optimistic) + top-k hit rate
    order = np.argsort(-train_scores, kind="stable")
    ranked_y = y[order]
    hits = {k: bool(ranked_y[:k].sum()) for k in (1, 3, 10)}
    (outdir / "metrics.json").write_text(json.dumps(
        {"n": len(y), "positives": int(y.sum()), "in_sample_topk_hit": hits,
         "features": FEATURES}, indent=2))
    print(f"[train] saved ranker.txt + features.json + calib.pkl to {outdir}/")
    print(f"[train] in-sample top-k hit (optimistic): {hits}")


if __name__ == "__main__":
    main()
