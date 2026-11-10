# models/ — CatBoost primary + LightGBM overlay (standalone, offline)

## Recsys mapping (two-stage)

- **Candidate generation** (`src/ml/candidates.py`): single source of truth for
  train AND inference — primary contig & (FILTER/ClinVar) & (rare &
  protein-altering | ClinVar P/LP | rare & (phenotype-gate gene | panel)).
  The phenotype gate = genes with normalized max-Resnik > `PHENO_GATE_MIN_NORM`
  vs the query HPO terms (same GeneHPO engine as feature computation).
- **Ranking** (`src/ml/rank_features.py`): 23-feature schema (annotation +
  phenotype + gene constraint), `derive_frame()` guarantees train/serve parity.
  The deployed primary (`models/ml_catboost/`) is a CatBoost `YetiRank`
  model with the 21 features available at serving: `pheno_excluded_overlap`
  (train-only) and `clinvar_score` (no ClinVar prior in the primary) are
  dropped. The LightGBM 21-feature model (`models/ml_no_excluded/`) is the
  previous primary; the full 23-feature LightGBM model (`models/ranker.txt`)
  is the documented clinical overlay.
  The model is **disease-agnostic** (no gene lists in features) and scores
  **standalone — no blending with the heuristic** (project decision; the
  heuristic is evaluated as a separate baseline system).

## Workflow (dataset → train → evaluate → infer)

```bash
# 1. dataset (select -> VEP -> build -> queries); see docs/PLAN_ml_ranker.md
python -m src.ml.make_dataset select
python -m src.annotation.vep -i data/ml/variants.vcf -o data/ml/variants.vep.vcf ...
python -m src.ml.make_dataset build
python -m src.ml.make_dataset queries

# 2. train (graded labels 0..3, early stopping, isotonic on VAL)
# deployed primary: CatBoost YetiRank, 21 features
python bin/train_catboost.py
# previous LightGBM primary (21 features) for comparison
python -m src.ml.train_ranker --train data/ml/dataset_train.csv.gz \
    --val data/ml/dataset_val.csv.gz --outdir models/ml_no_excluded \
    --drop-features clinvar_score,pheno_excluded_overlap
# full clinical overlay (LightGBM, 23 features)
python -m src.ml.train_ranker --train data/ml/dataset_train.csv.gz \
    --val data/ml/dataset_val.csv.gz --outdir models/
# ablations:
python -m src.ml.train_ranker --drop-features clinvar_score --outdir models/ml_no_clinvar

# 3. evaluate on held-out genes (test = phenopacket cases, never seen)
python -m src.ml.evaluate --ablations        # -> reports/eval_report.json

# 4. inference (ML score standalone, isotonic-calibrated; fail-fast if model missing)
python -m src.prioritization.prioritize --vcf data/interim/full.vep.vcf.gz \
    --hpo config/hpo_terms.yaml --model models/ml_catboost/model.cbm \
    -o results/ranked_variants.csv
```

## Artifacts (gitignored — regenerate with the commands above)

- `ranker.txt` — full 23-feature clinical overlay booster (LightGBM)
- `features.json` — schema guard (subset of `rank_features.FEATURES`, order-preserved)
- `calib.pkl` — isotonic calibrator fitted on **val** predictions (F-max)
- `metrics.json` — val metrics (NDCG@1/3/10, recall, MRR, Rank Points, calibration)
- `ml_catboost/` — **deployed primary** (CatBoost YetiRank, 21 features)
- `ml_no_excluded/` — previous LightGBM primary (21 features)
- `ml_no_excluded_af/` — experimental AF model (includes `rare_score`; worse test ranking, not deployed)
- `ml_no_excluded_clinvar/` — alternative 22-feature model (drops only `pheno_excluded_overlap`; not deployed)
- `ml_no_clinvar/`, `ml_no_pheno/` — ablation models

## Notes

- Query groups for lambdarank = `query_id` (one case/disease per query).
- Labels are GRADED: 3 = phenopacket case causal, 2 = ClinVar P/LP ≥2★,
  1 = ClinVar P/LP 1★, 0 = benign (ClinVar B/LB or VKGL LB). VUS excluded.
- P1 (label leakage via `clinvar_score`) is settled empirically on the novel
  stratum: see `reports/ml_ranking_report.md`.
- **Experimental ESM-2 embeddings:** variant-level ESM-2 35M delta embeddings
  and a masked-marginal score were tested on top of the 21-feature schema. They
  improve LightGBM (+0.0055 NDCG@10) but not CatBoost (+0.0009, CI includes 0);
  not deployed. See `reports/esm_ranker.md`.
- **Experimental gene-intrinsic block:** 16 features (gnomAD constraint,
  structure, HPOA inheritance/IC, interactions) are gated behind
  `--include-gene-features` in `train_ranker.py`. They are **not** part of the
  deployed schema: the 2026-11 ablation showed every group degrades the
  held-out test ranking. See `reports/gene_feature_ablation.md`. Do not pass
  the flag for the deployed primary.
