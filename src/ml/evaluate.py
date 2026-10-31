"""Evaluate the trained ranker on held-out test queries + baselines + ablations.

Test = phenopacket cases whose genes were held out of training entirely
(15% of genes), so nothing about those genes or their causal variants was
seen at train time. Baselines are SEPARATE systems (the ranker is scored
standalone — no blending, per project decision).

Systems evaluated on the same test pools:
  ml_full           trained ranker (models/ranker.txt), calibrated
  heuristic_ann     annotation-only heuristic (no genotype: quality neutral)
  clinvar_only      score = clinvar_score (label-leakage upper bound)
  resnik_only       score = pheno_resnik (retrieval signal alone)
  ml_no_clinvar     ablation A2: retrain without clinvar_score
  ml_no_pheno       ablation A3: retrain without the phenotype features

Usage:
  python -m src.ml.evaluate --data-dir data/ml --model models/ranker.txt --ablations
"""
from __future__ import annotations

import argparse
import json
import pickle as pkl
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.ml.eval_metrics import calibration_report, summarize
from src.ml.rank_features import derive_frame

PHENO_COLS = ["pheno_resnik", "pheno_bma", "pheno_jaccard", "pheno_terms_matched",
              "pheno_excluded_overlap", "gene_hpo_count", "gene_disease_count"]


def heuristic_ann_score(df: pd.DataFrame) -> np.ndarray:
    """features.py pathogenicity without quality (unavailable for public
    variants) + normalized BMA phenotype, 0.4/0.6 mix — the honest baseline."""
    X = derive_frame(df)
    path = (0.3 * X["rare_score"] + 0.3 * X["consequence_weight"]
            + 0.25 * X["cadd_norm"] + 0.1 * X["clinvar_score"])
    return (0.4 * path + 0.6 * X["pheno_bma"]).to_numpy()


def eval_system(df: pd.DataFrame, scores: np.ndarray, calibrated: np.ndarray | None
                ) -> dict:
    d = df.assign(score=scores)
    out = {"ranking": summarize(d)}
    y_bin = (df["label"].to_numpy() > 0).astype(int)
    out["calib_raw"] = calibration_report(y_bin, (scores - scores.min())
                                          / max(scores.max() - scores.min(), 1e-9))
    if calibrated is not None:
        out["calib_isotonic"] = calibration_report(y_bin, calibrated)
    return out


def stratify_by_known(df: pd.DataFrame, scores: np.ndarray) -> dict:
    """Recall@10 per stratum: query's causal variant known P/LP vs novel."""
    d = df.assign(score=scores)
    known_r, novel_r = [], []
    for _, g in d.groupby("query_id", sort=False):
        causal_mask = g["label"] >= 3
        if not causal_mask.any():
            continue
        is_known = bool(g.loc[causal_mask, "CLIN_SIG"].fillna("")
                        .str.lower().str.contains("pathogenic").any())
        order = np.argsort(-g["score"].to_numpy(), kind="stable")
        ranked_idx = g.index.to_numpy()[order]
        causal_idx = set(g.index[causal_mask])
        hit = float(any(i in causal_idx for i in ranked_idx[:10]))
        (known_r if is_known else novel_r).append(hit)
    return {
        "n_known_queries": len(known_r),
        "recall10_known": float(np.mean(known_r)) if known_r else None,
        "n_novel_queries": len(novel_r),
        "recall10_novel": float(np.mean(novel_r)) if novel_r else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/ml")
    p.add_argument("--model", default="models/ranker.txt")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--ablations", action="store_true")
    p.add_argument("--out", default="reports/eval_report.json")
    a = p.parse_args()
    data = Path(a.data_dir)

    df = pd.read_csv(data / f"dataset_{a.split}.csv.gz")
    df["vkey"] = (df.CHROM.astype(str) + ":" + df.POS.astype(str)
                  + ":" + df.REF + ":" + df.ALT)
    print(f"[eval] {a.split}: {len(df)} rows, {df.query_id.nunique()} queries, "
          f"{int((df.label > 0).sum())} positives, "
          f"{int((df.label >= 3).sum())} causal")
    y_bin = (df["label"].to_numpy() > 0).astype(int)

    X = derive_frame(df).to_numpy(dtype=np.float32)
    booster = lgb.Booster(model_file=a.model)
    raw = booster.predict(X)
    iso = pkl.loads(Path(a.model).with_name("calib.pkl").read_bytes())
    cal = iso.predict(raw)

    report: dict = {"split": a.split, "n_rows": len(df),
                    "queries": int(df.query_id.nunique())}
    report["ml_full"] = eval_system(df, raw, cal)
    report["stratified_ml_full"] = stratify_by_known(df, raw)
    report["heuristic_ann"] = eval_system(df, heuristic_ann_score(df), None)
    report["clinvar_only"] = eval_system(df, derive_frame(df)["clinvar_score"].to_numpy(), None)
    report["resnik_only"] = eval_system(df, derive_frame(df)["pheno_resnik"].to_numpy(), None)
    qs = data / "query_stats.json"
    if qs.exists():
        report["retrieval"] = json.loads(qs.read_text())

    if a.ablations:
        base = derive_frame(df)
        for name, drop in (("ml_no_clinvar", ["clinvar_score"]),
                           ("ml_no_pheno", PHENO_COLS)):
            outdir = Path("models") / name
            subprocess.run(
                [sys.executable, "-m", "src.ml.train_ranker",
                 "--train", str(data / "dataset_train.csv.gz"),
                 "--val", str(data / "dataset_val.csv.gz"),
                 "--outdir", str(outdir), "--drop-features", ",".join(drop)],
                check=True)
            abl = lgb.Booster(model_file=str(outdir / "ranker.txt"))
            cols = [c for c in base.columns if c not in drop]
            abl_raw = abl.predict(base[cols].to_numpy(dtype=np.float32))
            report[name] = eval_system(df, abl_raw, None)
            report[f"{name}_stratified"] = stratify_by_known(df, abl_raw)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))

    # compact stdout table
    def nd(name: str, k: str = "ndcg@10") -> str:
        r = report.get(name, {}).get("ranking", {})
        return f"{name:14s} NDCG@10={r.get(k, float('nan')):.4f} " \
               f"R@1={r.get('recall@1', float('nan')):.3f} " \
               f"R@10={r.get('recall@10', float('nan')):.3f} " \
               f"RP={r.get('rank_points', float('nan')):.1f}"
    for name in ("ml_full", "heuristic_ann", "clinvar_only", "resnik_only",
                 "ml_no_clinvar", "ml_no_pheno"):
        if name in report:
            print(nd(name))
    print("[eval] stratified:", json.dumps(report["stratified_ml_full"]))
    print(f"[eval] wrote {a.out}")


if __name__ == "__main__":
    main()
