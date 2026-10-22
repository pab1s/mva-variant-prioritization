# models/ — LightGBM LambdaRank ensemble (local, offline)

## Recsys mapping

- **Candidate generation** (`src/ml/candidates.py`): cheap high-recall rule
  (primary contig + rare + suspicious signal). 5M stream -> ~1-3% candidates.
- **Ranking** (`src/ml/rank_features.py` + `train_ranker.py`): 22 numeric
  features, all computed locally from VCF + VEP + HPO. No external services
  at train or predict time.

## Workflow (yours: download a labeled df, train, predict)

1. Score your VCF (features are recorded in the CSV):
   `python -m src.prioritization.prioritize --vcf ... --hpo ... -o scored.csv`
2. Prepare `labels.csv` with `CHROM,POS,REF,ALT,label` (1 = causal).
3. Train:
   `python -m src.ml.train_ranker --scored scored.csv --labels labels.csv --outdir models/`
   Writes `ranker.txt` + `features.json` (schema guard) + `calib.pkl`
   (isotonic, for F-max thresholds) + `metrics.json`.
4. Predict (blend 0.7 ML / 0.3 heuristic, `model_weight` in config):
   `python -m src.prioritization.prioritize --vcf ... --model models/ranker.txt -o ranked.csv`
   Missing model file with `--model` set = hard error (fail fast).

## Notes

- Query groups for lambdarank default to `SYMBOL` (learn within-gene
  ordering); use `--query` for case-level or custom groups.
- `cadd_missing=1` flags heuristic CADD — retrain after the CADD plugin
  lands for real `CADD_PHRED`.
- Tracked here: only READMEs. Trained `ranker.txt` files are yours;
  commit them only if small and shareable (they usually are, ~KBs).
