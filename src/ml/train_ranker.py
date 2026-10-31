"""Train the LightGBM LambdaRank variant ranker (standalone, no blend).

Input: dataset CSVs from `src/ml/make_dataset.py queries` (dataset_train/val
CSV.gz) — rows carry raw annotation columns + per-query phenotype features +
graded label (0 benign, 1 P/LP 1-star, 2 P/LP >=2 stars, 3 case causal) +
query_id.

Output (models/):
  ranker.txt     booster (predict -> uncalibrated ranking score)
  features.json  schema guard (FEATURES list; prioritize checks it)
  calib.pkl      isotonic calibrator fitted on VAL predictions (for F-max)
  metrics.json   val metrics: NDCG@1/3/10, recall@1/3/10, MRR, Rank Points,
                 calibration (ECE/Brier/F-max)

Usage:
  python -m src.ml.train_ranker --train data/ml/dataset_train.csv.gz \
      --val data/ml/dataset_val.csv.gz --outdir models/
"""
from __future__ import annotations

import argparse
import json
import pickle as pkl
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.ml.eval_metrics import calibration_report, summarize
from src.ml.rank_features import FEATURES, derive_frame

# graded-label gains: benign 0 < P/LP 1* (0.25) < P/LP 2* (1.0) < causal (3.0)
LABEL_GAIN = [0.0, 0.25, 1.0, 3.0]


def load_xy(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """CSV.gz -> X (features), y (graded int), query_id array, raw frame."""
    df = pd.read_csv(path)
    X = derive_frame(df).to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=int)
    qid = df["query_id"].astype(str).to_numpy()
    return X, y, qid, df


def groups_for(qid: np.ndarray) -> list[int]:
    """Contiguous group sizes (rows must be sorted by query_id)."""
    _, counts = np.unique(qid, return_counts=True)
    return counts.tolist()


def sort_by_query(X, y, qid, seed: int = 7):
    """Group rows by query (contiguous) AND shuffle within each group.

    The shuffle matters: dataset pools list positives first, so with tied
    scores both LightGBM's eval and ours would see a lucky high NDCG.
    Shuffling makes tied-score metrics honest (random baseline).
    """
    codes, _ = pd.factorize(qid, sort=True)
    rng = np.random.default_rng(seed)
    rand = rng.random(len(qid))
    order = np.lexsort((rand, codes))
    return X[order], y[order], qid[order], order


def main() -> None:
    p = argparse.ArgumentParser(description="Train LightGBM LambdaRank (standalone)")
    p.add_argument("--train", default="data/ml/dataset_train.csv.gz")
    p.add_argument("--val", default="data/ml/dataset_val.csv.gz")
    p.add_argument("--outdir", default="models")
    p.add_argument("--rounds", type=int, default=3000)
    p.add_argument("--es", type=int, default=100, help="early stopping rounds")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--leaves", type=int, default=63)
    p.add_argument("--min-data-in-leaf", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--drop-features", default="",
                   help="comma-separated FEATURES to drop (ablations)")
    a = p.parse_args()

    Xtr, ytr, qtr, _ = load_xy(a.train)
    Xva, yva, qva, dva = load_xy(a.val)
    keep = [f for f in FEATURES if f not in set(a.drop_features.split(","))]
    idx = [FEATURES.index(f) for f in keep]
    Xtr, Xva = Xtr[:, idx], Xva[:, idx]
    print(f"[train] {len(ytr)} train rows ({len(np.unique(qtr))} queries, "
          f"{int((ytr > 0).sum())} pos), {len(yva)} val rows "
          f"({len(np.unique(qva))} queries), {Xtr.shape[1]} features "
          f"(dropped: {a.drop_features or 'none'})")

    Xtr, ytr, qtr, _ = sort_by_query(Xtr, ytr, qtr, seed=a.seed)
    Xva_s, yva_s, qva_s, va_order = sort_by_query(Xva, yva, qva, seed=a.seed + 1)
    dva = dva.iloc[va_order].reset_index(drop=True)  # keep metrics row-aligned

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "label_gain": LABEL_GAIN,
        "learning_rate": a.lr,
        "num_leaves": a.leaves,
        "min_data_in_leaf": a.min_data_in_leaf,
        "feature_fraction": 0.8,
        "lambdarank_truncation_level": 10,
        "verbose": -1,
        "seed": a.seed,
    }
    dtrain = lgb.Dataset(Xtr, label=ytr, group=groups_for(qtr), free_raw_data=False)
    dval = lgb.Dataset(Xva_s, label=yva_s, group=groups_for(qva_s),
                       reference=dtrain, free_raw_data=False)
    booster = lgb.train(
        params, dtrain, num_boost_round=a.rounds, valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(a.es, verbose=True),
                   lgb.log_evaluation(200)])

    # ---- val metrics (raw ranking + calibrated probabilities)
    raw_va = booster.predict(Xva_s)
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_va, (yva_s > 0).astype(int))
    cal_va = iso.predict(raw_va)
    dva = dva.assign(score=raw_va)
    metrics = summarize(dva)
    metrics.update({f"calib_{k}": v for k, v in
                    calibration_report((yva_s > 0).astype(int), cal_va).items()})
    metrics["best_iteration"] = int(booster.best_iteration or a.rounds)
    metrics["label_gain"] = LABEL_GAIN
    metrics["features"] = keep

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(outdir / "ranker.txt"))
    (outdir / "features.json").write_text(json.dumps(keep))
    (outdir / "calib.pkl").write_bytes(pkl.dumps(iso))
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("[train] val metrics:",
          json.dumps({k: round(v, 4) for k, v in metrics.items()
                      if isinstance(v, float)}, indent=1))
    print(f"[train] saved ranker.txt (iter={metrics['best_iteration']}), "
          f"features.json, calib.pkl, metrics.json -> {outdir}/")


if __name__ == "__main__":
    main()
