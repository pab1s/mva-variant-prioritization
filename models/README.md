# models/ — LightGBM LambdaRank ranker (standalone, offline)

## Recsys mapping (two-stage)

- **Candidate generation** (`src/ml/candidates.py`): single source of truth for
  train AND inference — primary contig & (FILTER/ClinVar) & (rare &
  protein-altering | ClinVar P/LP | rare & (phenotype-gate gene | panel)).
  The phenotype gate = genes with normalized max-Resnik > `PHENO_GATE_MIN_NORM`
  vs the query HPO terms (same GeneHPO engine as feature computation).
- **Ranking** (`src/ml/rank_features.py`): 23-feature schema (annotation +
  phenotype + gene constraint), `derive_frame()` guarantees train/serve parity.
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

# 2. train (graded labels 0..3, label_gain, early stopping, isotonic on VAL)
python -m src.ml.train_ranker --train data/ml/dataset_train.csv.gz \
    --val data/ml/dataset_val.csv.gz --outdir models/
# ablations:
python -m src.ml.train_ranker --drop-features clinvar_score --outdir models/ml_no_clinvar

# 3. evaluate on held-out genes (test = phenopacket cases, never seen)
python -m src.ml.evaluate --ablations        # -> reports/eval_report.json

# 4. inference (ML score standalone, isotonic-calibrated; fail-fast if model missing)
python -m src.prioritization.prioritize --vcf data/interim/full.vep.vcf.gz \
    --hpo config/hpo_terms.yaml --model models/ranker.txt -o results/ranked_variants.csv
```

## Artifacts (gitignored — regenerate with the commands above)

- `ranker.txt` — booster
- `features.json` — schema guard (subset of `rank_features.FEATURES`, order-preserved)
- `calib.pkl` — isotonic calibrator fitted on **val** predictions (F-max)
- `metrics.json` — val metrics (NDCG@1/3/10, recall, MRR, Rank Points, calibration)
- `ml_no_clinvar/`, `ml_no_pheno/` — ablation models

## Notes

- Query groups for lambdarank = `query_id` (one case/disease per query).
- Labels are GRADED: 3 = phenopacket case causal, 2 = ClinVar P/LP ≥2★,
  1 = ClinVar P/LP 1★, 0 = benign (ClinVar B/LB or VKGL LB). VUS excluded.
- P1 (label leakage via `clinvar_score`) is settled empirically on the novel
  stratum: see `reports/ml_ranking_report.md`.
