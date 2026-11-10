"""Train a ranker with ESM-2 variant embeddings (experimental).

Feature blocks
--------------
- base  : the deployed 21-feature schema (`FEATURES` minus the two train-only
          columns `clinvar_score`, `pheno_excluded_overlap`).
- score : ESM-2 masked-marginal zero-shot deleteriousness (`esm_score`).
- delta : PCA of the mutant-minus-wildtype hidden state at the mutated residue
          (`--n-components`, fit on TRAIN variants only).

Every block is merged by `vkey`; non-missense variants get zeros plus an
`esm_present` flag. The split is the existing gene-disjoint train/val/test, so
the test set never shares a causal gene with train.

Run (screening):
  .venv/bin/python -m src.ml.train_esm_ranker --backend lightgbm --esm both

Run (final):
  .venv/bin/python -m src.ml.train_esm_ranker --backend catboost --esm both
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

from src.ml.eval_metrics import calibration_report, query_metrics, summarize
from src.ml.rank_features import FEATURES, derive_frame
from src.ml.train_ranker import sort_by_query

ROOT = Path(__file__).resolve().parents[2]
ML_DIR = ROOT / "data/ml"
BASE_FEATS = [f for f in FEATURES
              if f not in ("clinvar_score", "pheno_excluded_overlap")]
ESM_STEM = ML_DIR / "esm2_35m"


def vkey(df: pd.DataFrame) -> pd.Series:
    if "vkey" in df.columns:
        return df["vkey"].astype(str)
    return (df.CHROM.astype(str) + ":" + df.POS.astype(str)
            + ":" + df.REF + ":" + df.ALT)


def load_split(split: str):
    df = pd.read_csv(ML_DIR / f"dataset_{split}.csv.gz", low_memory=False)
    X = derive_frame(df)[BASE_FEATS].to_numpy(np.float32)
    y = df["label"].to_numpy(np.int64)
    qid = df["query_id"].astype(str).to_numpy()
    return df, X, y, qid


def build_esm_features(block: str, n_components: int, train_vkeys: set[str]
                       ) -> pd.DataFrame:
    """vkey-indexed ESM columns. PCA is fit on train variants only."""
    idx = pd.read_csv(f"{ESM_STEM}_index.csv.gz")
    idx["vkey"] = idx["vkey"].astype(str)
    delta = np.load(f"{ESM_STEM}_delta.npy")
    assert len(delta) == len(idx), "delta/index length mismatch"

    cols: dict[str, np.ndarray] = {"esm_present": np.ones(len(idx))}
    if block in ("score", "both"):
        cols["esm_score"] = idx["esm_score"].to_numpy(np.float32)
    if block in ("delta", "both"):
        fit_mask = idx["vkey"].isin(train_vkeys).to_numpy()
        raw = delta[fit_mask].astype(np.float32)
        scaler = StandardScaler().fit(raw)
        pca = PCA(n_components=n_components, svd_solver="randomized",
                  random_state=7).fit(scaler.transform(raw))
        comp = pca.transform(scaler.transform(delta.astype(np.float32)))
        for i in range(n_components):
            cols[f"esm_delta_{i:02d}"] = comp[:, i]
        print(f"[esm] PCA fit on {fit_mask.sum()} train variants, "
              f"{n_components} comps, explained={pca.explained_variance_ratio_.sum():.3f}")
    out = pd.DataFrame(cols, index=idx["vkey"].to_numpy())
    out.index.name = "vkey"
    return out


def merge_esm(X: np.ndarray, df: pd.DataFrame, esm: pd.DataFrame | None
              ) -> np.ndarray:
    if esm is None:
        return X
    rows = esm.reindex(vkey(df).to_numpy())
    present = rows["esm_present"].notna().to_numpy()
    rows = rows.fillna(0.0)
    extra = rows.to_numpy(np.float32)
    extra[~present, :] = 0.0
    if esm.shape[1] == 1:  # only the flag column
        return np.hstack([X, extra])
    return np.hstack([X, extra])


def _bootstrap(base: np.ndarray, new: np.ndarray, labels: np.ndarray,
               qids: np.ndarray, seed: int = 7, n_boot: int = 5000) -> dict:
    rows = []
    for q in pd.unique(qids):
        m = qids == q
        a = query_metrics(pd.DataFrame({"label": labels[m], "score": base[m]}))
        b = query_metrics(pd.DataFrame({"label": labels[m], "score": new[m]}))
        rows.append({"ndcg10": b["ndcg@10"] - a["ndcg@10"],
                     "r1": b["recall@1"] - a["recall@1"],
                     "rp": b["rank_points"] - a["rank_points"]})
    m = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    out = {}
    for metric in ("ndcg10", "r1", "rp"):
        diff = m[metric].to_numpy()
        boots = np.array([np.nanmean(diff[rng.integers(0, len(diff), len(diff))])
                          for _ in range(n_boot)])
        out[metric] = {
            "mean_diff": float(np.nanmean(diff)),
            "ci95": [float(np.nanquantile(boots, 0.025)),
                     float(np.nanquantile(boots, 0.975))],
            "win": int((diff > 1e-9).sum()),
            "loss": int((diff < -1e-9).sum()),
            "tie": int((np.abs(diff) <= 1e-9).sum()),
        }
    return out


def fit_predict(backend: str, Xtr, ytr, qtr, Xva, yva, qva, Xte, args,
                feature_names: list[str] | None = None):
    Xtr_s, ytr_s, qtr_s, _ = sort_by_query(Xtr, ytr, qtr, seed=7)
    Xva_s, yva_s, qva_s, va_order = sort_by_query(Xva, yva, qva, seed=8)
    t0 = time.time()
    if backend == "catboost":
        from catboost import CatBoostRanker, Pool
        model = CatBoostRanker(
            loss_function="YetiRank", iterations=args.rounds,
            learning_rate=args.lr, depth=6, l2_leaf_reg=args.l2,
            random_seed=7, thread_count=16, task_type="CPU",
            verbose=100, allow_writing_files=False,
            eval_metric="NDCG:top=10")
        model.fit(Pool(Xtr_s, ytr_s, group_id=qtr_s),
                  eval_set=Pool(Xva_s, yva_s, group_id=qva_s),
                  use_best_model=True, early_stopping_rounds=50)
        va_raw_s = model.predict(Xva_s)
        va_raw = np.empty_like(va_raw_s)
        va_raw[va_order] = va_raw_s
        te_raw = model.predict(Xte)
        best_iter = int(model.get_best_iteration())
    elif backend == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMRanker(
            objective="lambdarank", n_estimators=args.rounds,
            learning_rate=args.lr, num_leaves=args.leaves,
            min_child_samples=50, label_gain=[0, 1, 3, 7],
            eval_at=[1, 3, 10], random_state=7, n_jobs=16, verbose=-1)
        _, counts_tr = np.unique(qtr_s, return_counts=True)
        _, counts_va = np.unique(qva_s, return_counts=True)
        model.fit(Xtr_s, ytr_s, group=counts_tr,
                  eval_set=[(Xva_s, yva_s)], eval_group=[counts_va],
                  callbacks=[lgb.early_stopping(100, verbose=False),
                             lgb.log_evaluation(100)])
        va_raw_s = model.predict(Xva_s, num_iteration=model.best_iteration_)
        va_raw = np.empty_like(va_raw_s)
        va_raw[va_order] = va_raw_s
        te_raw = model.predict(Xte, num_iteration=model.best_iteration_)
        best_iter = int(model.best_iteration_ or args.rounds)
    else:
        raise ValueError(backend)

    imp = np.asarray(
        model.get_feature_importance(type="PredictionValuesChange")
        if backend == "catboost" else model.feature_importances_,
        dtype=float)
    if feature_names is None or len(feature_names) != len(imp):
        feature_names = [f"f{i}" for i in range(len(imp))]
    order = np.argsort(imp)[::-1]
    importance = {
        "top15": [{"feature": feature_names[i], "importance": float(imp[i])}
                  for i in order[:15]],
        "esm_share": float(imp[[i for i, f in enumerate(feature_names)
                                if f.startswith("esm_")]].sum()
                           / max(imp.sum(), 1)),
    }
    return va_raw, te_raw, best_iter, time.time() - t0, importance, model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["catboost", "lightgbm"],
                   default="lightgbm")
    p.add_argument("--esm", choices=["none", "score", "delta", "both"],
                   default="both")
    p.add_argument("--n-components", type=int, default=48)
    p.add_argument("--rounds", type=int, default=0,
                   help="0 -> 400 CatBoost / 2000 LightGBM")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--l2", type=float, default=3.0)
    p.add_argument("--leaves", type=int, default=63)
    p.add_argument("--tag", default="")
    p.add_argument("--compare-base", default="",
                   help="tag of a previous run in /tmp/opencode/esm_ranker to "
                        "bootstrap against")
    p.add_argument("--save-dir", default="",
                   help="optional artifact dir (model + schema + calibrator)")
    a = p.parse_args()
    a.rounds = a.rounds or (400 if a.backend == "catboost" else 2000)

    tr, Xtr0, ytr, qtr = load_split("train")
    va, Xva0, yva, qva = load_split("val")
    te, Xte0, yte, qte = load_split("test")
    print(f"[data] train={len(tr)} val={len(va)} test={len(te)}")

    esm = None
    if a.esm != "none":
        all_vkeys = set(pd.read_csv(f"{ESM_STEM}_index.csv.gz",
                                    usecols=["vkey"])["vkey"].astype(str))
        train_vkeys = set(vkey(tr)) & all_vkeys
        esm = build_esm_features(a.esm, a.n_components, train_vkeys)
        print(f"[esm] block={a.esm} columns={esm.shape[1]}")
    Xtr = merge_esm(Xtr0, tr, esm)
    Xva = merge_esm(Xva0, va, esm)
    Xte = merge_esm(Xte0, te, esm)
    print(f"[feats] {Xtr.shape[1]} total")

    feat_names = BASE_FEATS + (list(esm.columns) if esm is not None else [])
    va_raw, te_raw, best_iter, secs, importance, model = fit_predict(
        a.backend, Xtr, ytr, qtr, Xva, yva, qva, Xte, a, feat_names)
    print(f"[{a.backend}] fit {secs:.1f}s best_iter={best_iter} "
          f"esm_importance_share={importance['esm_share']:.3f}")

    iso = IsotonicRegression(out_of_bounds="clip").fit(
        va_raw, (yva > 0).astype(int))
    va_cal = iso.predict(va_raw)
    te_cal = iso.predict(te_raw)

    if a.save_dir:
        d = Path(a.save_dir)
        d.mkdir(parents=True, exist_ok=True)
        if a.backend == "catboost":
            model.save_model(str(d / "model.cbm"))
            (d / "model_type.txt").write_text("catboost")
        else:
            model.booster_.save_model(str(d / "ranker.txt"))
            (d / "model_type.txt").write_text("lightgbm")
        (d / "features.json").write_text(json.dumps(feat_names))
        (d / "calib.pkl").write_bytes(pickle.dumps(iso))
        print(f"[save] artifacts -> {d}")

    def _eval(df, y, score, cal, name):
        m = summarize(pd.DataFrame({
            "query_id": df["query_id"].astype(str), "label": y,
            "score": score}))
        m.update({f"calib_{k}": v for k, v in calibration_report(
            (y > 0).astype(int), cal).items()})
        print(f"[{name}] " + json.dumps(
            {k: round(v, 4) for k, v in m.items() if isinstance(v, float)}))
        return m

    val_metrics = _eval(va, yva, va_raw, va_cal, "val")
    test_metrics = _eval(te, yte, te_raw, te_cal, "test")

    tag = a.tag or f"{a.backend}_{a.esm}"
    out = {"tag": tag, "backend": a.backend, "esm": a.esm,
           "n_components": a.n_components if a.esm in ("delta", "both") else 0,
           "n_features": int(Xtr.shape[1]), "best_iteration": best_iter,
           "seconds": round(secs, 1), "importance": importance,
           "params": {"rounds": a.rounds, "lr": a.lr, "l2": a.l2,
                      "leaves": a.leaves, "seed": 7},
           "val": val_metrics, "test": test_metrics}

    # paired bootstrap against the no-ESM model trained in this same run is
    # done by the orchestration script; store raw scores for that purpose.
    outdir = Path("/tmp/opencode/esm_ranker")
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / f"{tag}_test.npy", te_raw)
    np.save(outdir / f"{tag}_val.npy", va_raw)
    if a.compare_base:
        base_scores = np.load(outdir / f"{a.compare_base}_test.npy")
        out["bootstrap_vs"] = a.compare_base
        out["bootstrap"] = _bootstrap(
            base_scores, te_raw, yte, qte.astype(str))
        print("[bootstrap] " + json.dumps(
            {k: {"mean_diff": round(v["mean_diff"], 4),
                 "ci95": [round(x, 4) for x in v["ci95"]]}
             for k, v in out["bootstrap"].items()}))
    outdir.joinpath(f"{tag}_metrics.json").write_text(json.dumps(out, indent=2))
    if a.save_dir:
        Path(a.save_dir).joinpath("metrics.json").write_text(
            json.dumps(out, indent=2))
    print(f"[save] {outdir / (tag + '_metrics.json')}")


if __name__ == "__main__":
    main()
