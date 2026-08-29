# MVA Variant Prioritization

> **Mosaic Variegated Aneuploidy (MVA) — Rare Disease, Real Kid Hackathon 2026**  
> Track 1 (Variant Prediction) + Track 2 (Drug Repurposing) — reproducible, end-to-end pipeline for ranking disease-causing variants from WGS + phenotype.

[![Hackathon](https://img.shields.io/badge/hackathon-MVA%202026-orange)](https://sagebio-rare-disease-real-kid-mva-hackathon-2026.hf.space/)
[![License: CC BY 4.0](https://img.shields.io/badge/outputs-CC%20BY%204.0-green)](https://creativecommons.org/licenses/by/4.0/)

A family living with **MVA (<50 cases worldwide, no established treatment)** shared their child's real genome with the global research community. This repo builds a modular pipeline to prioritize causal variants — methods designed to generalize to other undiagnosed rare-disease cases.

**Problem statement (curated):** [`docs/problem_statement.md`](docs/problem_statement.md) · **Source of truth:** https://sagebio-rare-disease-real-kid-mva-hackathon-2026.hf.space/

---

## Project Structure

```
.
├── src/
│   ├── preprocessing/      # FASTQ QC, trimming, alignment prep
│   ├── variant_calling/    # callers (GATK/DeepVariant) + filtering
│   ├── annotation/         # VEP, gnomAD, CADD, ClinVar
│   ├── phenotype/          # HPO parsing + similarity
│   ├── prioritization/     # gene-phenotype + pathogenicity integration
│   └── ranking/            # final scoring, Rank Points / F-max, submission CSV
├── configs/                # pipeline YAMLs (reference, thresholds, HPO terms)
├── workflows/              # Snakemake (Snakefile) + Nextflow (nextflow.config)
├── notebooks/              # 01_exploratory, 02_variant_prioritization
├── tests/                  # pytest suites
├── benchmarks/             # ranking benchmarks
├── reports/                # Track 1 & Track 2 submission templates
├── docs/
│   └── problem_statement.md
└── README.md
```

---

## Quick Start

```bash
# clone
git clone https://github.com/pab1s/mva-variant-prioritization.git
cd mva-variant-prioritization

# env (pick one)
python -m venv .venv && source .venv/bin/activate
# or: conda env create -f environment.yml  # TODO

pip install -e .  # TODO: add pyproject.toml

# run QC placeholder
python -c "from src.preprocessing.qc import run_fastqc; print(run_fastqc('data/sample.fastq.gz','results/qc'))"

# run tests
pytest -q
```

### Workflows

```bash
# Snakemake (dry run)
snakemake -s workflows/Snakefile -n

# Nextflow (stub)
nextflow run workflows/main.nf -profile local  # TODO
```

### Submission

```bash
# after ranking:
python -m src.ranking.rank --input results/scored.csv --output results/ranked_variants.csv
# then format per Track 1 spec and upload via Space → Submit - Track 1
```

---

## Tracks at a Glance

| Track | Goal | Input | Output | Scoring | Limit |
|---|---|---|---|---|---|
| **1 — Variant Prediction** | Rank causal variant(s) | FASTQ+VCF+phenotype | ranked variants + GitHub + writeup | **Rank Points** (100 if #1, partial top-10; half credit for 1/2 of het pair) + **F-max** | 6 |
| **2 — Drug Repurposing** | Propose approved drug(s) for mechanism | inferred mechanism | report + GitHub + 3-min video | Expert panel: Rigor 35%, Impact 25%, Innovation 25%, Scalability 15% | 1 |

See `docs/problem_statement.md` for full rules, timeline, and prizes ($50k: $25k cash + $25k Claude credits).

---

## Pipeline Overview

```
FASTQ ──▶ QC / trimming ──▶ alignment (BWA) ──▶ variant calling ──▶ filtering
                                        │
VCF ──▶ annotation (VEP + gnomAD + CADD/REVEL + ClinVar)
                                        │
Clinical description ──▶ HPO curation ──▶ phenotype similarity (Phen2Gene / Exomiser-style)
                                        │
                              prioritization (weighted / ML) ──▶ ranking ──▶ submission
```

Scoring-aware: ranking calibrated for **Rank Points + F-max** (well-calibrated scores, not just ranks).

---

## Configs

- `configs/example_config.yaml` — reference, VEP cache, AF/CADD thresholds, phenotype vs pathogenicity weights.
- `configs/hpo_terms.yaml` — fill with curated HPO terms from clinical data (controlled access).

---

## Reports & Notebooks

- `notebooks/01_exploratory.ipynb` — QC, VCF stats, HPO overview.
- `notebooks/02_variant_prioritization.ipynb` — prototype ranking.
- `reports/track1_template.md` / `reports/track2_template.md` — submission write-up scaffolds.

---

## Timeline (from Space, subject to change)

- **Aug 24** — Hackathon launch
- **Aug 25** — Submissions open
- **Oct 24 23:59 UTC** — Submissions close / leaderboard frozen
- **Oct 24–Nov 24** — Judging
- **Nov 25** — Winners announced

---

## Acknowledgements

Gratitude to the child and family who shared their genome, and to **MVA Society, Sage Bionetworks, Hugging Face, BEACON, AWS, Anthropic, Wilhelm Foundation**. Data under **WCG IRB #20252010**; outputs **CC BY 4.0**. Not medical advice.

---

## Contributing

PRs welcome. Run `pytest` and `ruff check` before pushing. See `docs/problem_statement.md` for full context.
