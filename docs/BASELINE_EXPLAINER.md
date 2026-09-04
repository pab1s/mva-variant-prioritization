# Baseline Explainer — MVA Variant Prioritization for Data Scientists

> **Audience:** You're a data scientist who read [learngenomics.dev](https://learngenomics.dev) but aren't a genomics specialist. This doc maps genomics concepts to DS intuitions, explains every artifact in the repo, and tells you exactly what the baseline pipeline you just ran does — and how to improve it.

---

## 1. The 30-Second Story

A family with an ultra-rare disease (**Mosaic Variegated Aneuploidy, <50 cases**) shared their child's **whole genome** (WGS) and clinical description. Your job in **Track 1** is to rank the 5M variants in that genome so the true causal one(s) appear at the top. It's a **ranking / information-retrieval** problem with two metrics:

- **Rank Points:** 100 if causal is #1, partial if in top-10 (like NDCG@10 but harsher).
- **F-max:** best F1 across all score thresholds (needs *calibrated* probabilities, not just order — like PR-AUC max).

You have: raw **FASTQ** (sequencer output), a **VCF** (5M candidate variants already called), and 8 **HPO terms** (standardized phenotype).

---

## 2. Data Glossary — DS Analogies

| Genomics term | What it is | DS analogy | Where it lives |
|---|---|---|---|
| **FASTQ** | 80 GB of short reads: `ATCG` strings + quality per base (like log with confidence). 4 lanes × R1/R2 = 8 files (~10 GB each). | Raw unstructured event stream (web logs) with per-event confidence | `data/*.fastq.gz` (8 files) |
| **FASTQ quality (Phred)** | `Q30 = 99.9% accurate`. Sequencer assigns per base. | Per-row confidence score | Inside FASTQ 4th line per read |
| **Reference genome GRCh38** | 3B bases "ground truth" human genome we align against. | Canonical lookup table / master dictionary | `/data/references/GRCh38.fa` (not in repo, but cached) |
| **BAM** | Reads aligned to reference (where each read maps). Not provided, but intermediate if you re-align. | Joined table: reads × reference positions | Generated if you run BWA |
| **VCF** | Table of *differences* vs reference: 5,012,204 rows (variants). Columns: CHROM, POS, REF, ALT, QUAL, FILTER, INFO. Each row = "at chr1:13273, ref G → alt C in this child". | Diff / event table (like credit-card anomalies vs expected) | `data/WGS_EX2312012_HGWCNDSX7.vcf.gz` (301 MB, + .tbi index) |
| **QUAL / FILTER / QD / MQ / FS** | Variant quality metrics. `QD<2` or `MQ<40` etc. filtered as low-confidence. | Model diagnostics (residual, leverage) | VCF INFO + FILTER column |
| **GT:AD:DP:GQ:PL** | Genotype fields: diploid `0/1` heterozygous, depths, likelihoods. | User label + feature vector per row | VCF FORMAT column |
| **HPO (Human Phenotype Ontology)** | Standard vocabulary for symptoms: `HP:0002859 = Rhabdomyosarcoma`. 8 terms curated from the clinical doc. | One-hot encoded target tags | `config/hpo_terms.yaml` + `data/Challenge_Clinical_Phenotype_1.docx` |
| **VEP (Ensembl Variant Effect Predictor)** | Tool that annotates each VCF row with: gene (`SYMBOL`), consequence (`missense`, `frameshift`), impact, CADD, gnomAD AF, ClinVar. | Feature enrichment / lookup join (gene ontology, impact) | Runs as `workflows/modules/annotate.nf` — real VEP if cache exists, else pass-through mock |
| **gnomAD AF** | Population allele frequency: how common is this variant in 140k healthy people? Rare (<0.01) → more likely disease-causing. | Prevalence feature (rare → higher signal) | VEP adds `gnomAD_AF` INFO; baseline filters `<0.01` |
| **CADD / REVEL** | Pathogenicity scores (like model output: predicted deleteriousness). CADD PHRED 30 = top 0.1% most damaging. | Pre-trained model score feature | VEP or separate annotation |
| **ClinVar** | Database of known disease variants: Pathogenic / Benign labels. | Historical label lookup | VEP `--clin_sig` |
| **MVA genes** | Known MVA-associated genes: `BUB1B`, `BUB3`, `TRIP13`, `CEP57` (+ `BUB1`, `MAD2L1`). True causal likely hits one. | High-value target entities (like fraud keywords) | Boosted in `src/prioritization/features.py` & `prioritize.py` |

---

## 3. Pipeline Map — What Runs Where

```
                          ┌─────────────────────────────────────────────────┐
FASTQ (8×10GB) ───────────▶│  QC  (src/preprocessing/qc.py)                │  per-read QC: avg length 149, GC ~41%
       │                  │  ├─ fastqc (if installed) else stub sampler   │  sampling 10k reads/file → qc JSON
       │                  │  └─ results/qc/*.json (8 files)               │
       │                  └─────────────────────────────────────────────────┘
                                  │
                                  │ (optional trimming: fastp/cutadapt or symlink)
                                  ▼
VCF 5M rows ──────────────▶┌─────────────────────────────────────────────────┐
data/WGS...vcf.gz (301MB) │  ANNOTATE (src/annotation/vep.py)              │  tries: native VEP → docker VEP → mock pass-through
 + .tbi index              │  ├─ needs /data/vep_cache (~20GB) + vep binary│  currently MOCK (adds ##VEP_MOCK flag)
                           │  └─ work/.../WGS...vcf.vep.vcf.gz (301MB)    │
                           └─────────────────────────────────────────────────┘
                                  │
Clinical docx ──▶ HPO yaml  ┌─────────────────────────────────────────────────┐
8 HPO terms                 │  PRIORITIZE (src/prioritization/prioritize.py)│  scores each variant:
config/hpo_terms.yaml       │  ├─ features.py → pathogenicity_score        │    pathogenicity (0.3 rare +0.3 consequence+0.25 CADD+...)
+ example_config.yaml       │  ├─ hpo.py → phenotype_score (0.5 neutral)   │    phenotype (Jaccard or 0.5 if no gene map)
 (weights 0.6/0.4)          │  └─ score = 0.6*pheno + 0.4*path              │    calibration → 0.01-0.99 for F-max
                           │  └─ 5,079,064 scored rows (multiallelic split)│    5079064 = 5.0M + ~67k multiallelic allelic splits
                           └─────────────────────────────────────────────────┘
                                  │
                           ┌─────────────────────────────────────────────────┐
                           │  RANK (src/ranking/rank.py)                    │  sort by score → RANK + SCORE (0-1)
                           │  ├─ rank_variants()                            │  ties broken by pathogenicity/quality
                           │  └─ to_submission_format()                     │  write Track1 CSV header:
                           └─────────────────────────────────────────────────┘
                                  │
                                  ▼ results/ranked_variants.csv (541 MB, 5,079,065 lines incl header)
                                  CHROM,POS,REF,ALT,RANK,SCORE,SYMBOL,Consequence,gnomad_af,CADD_PHRED,FILTER,score,pathogenicity_score,phenotype_score
                                  └─ upload via HF Space → Submit - Track 1 (6 allowed, live leaderboard)
```

**Nextflow orchestration:** `workflows/main.nf` (DSL2, modular) + `workflows/nextflow.config` wire those steps. Each `process` runs in isolated `work/<hash>/` then `publishDir` copies to `results/`. View execution: `results/pipeline_info/{timeline,report,dag}.html`.

---

## 4. What You Just Executed — Baseline

### Command (both QC + VCF)
```bash
nextflow run workflows/main.nf -profile local
# with --skip_qc true to skip QC, --max_variants 5000 to debug small slice
```

### What happened (log from your run)

- **QC:** 8 parallel `QC` tasks (1 per FASTQ, stub sampler: 10k reads/file, avg_len 149, GC 41%). Completed in ~1 sec, outputs in `results/qc/*.json`.
- **ANNOTATE:** Tried native VEP (`/data/vep_cache` missing) → fell back to pass-through copy (marked `VEP_MOCK`). Completed cached after first run.
- **PRIORITIZE:** Streamed 5,012,204 VCF rows → expanded multiallelic → 5,079,064 alleles, computed features per variant, scored `0.6*phenotype +0.4*pathogenicity`, calibrated to 0.01–0.99, ranked. **~75 sec** for 5M ( ~67k var/s). Memory 8–12 GB.
- **Output:** `results/ranked_variants.csv` (541 MB, 5,079,064 ranked). Top 5 all `frameshift_variant` PASS with high QD (≈1k) due to heuristic weighting indels higher (see Section 5).

```text
CHROM,POS,REF,ALT,RANK,SCORE,...
Un_KN707885v1_decoy,528,TAAG,T,1,0.99,.,frameshift_variant,,,PASS,0.99,...
17_GL000205v2_random,54674,TGTAA,T,2,0.989...,.,frameshift_variant,,,PASS,...
21,8995520,T,TG,3,0.989...,.,frameshift_variant,,,PASS,...
```

`SYMBOL` is `.` for all because without VEP we lack gene annotation — so phenotype score is neutral 0.5 for every variant (pathogenicity drives rank). That's the biggest baseline limitation.

### Config

```yaml
# config/example_config.yaml
reference: "/data/references/GRCh38.fa"
vep_cache: "/data/vep_cache"      # auto-detected; baseline mocks if absent
gnomad_af_threshold: 0.01           # keep AF <1% (but applied post-scoring, not strict)
phenotype_weight: 0.6
pathogenicity_weight: 0.4
```

### Environment

- **Python:** managed by `uv` (`pyproject.toml`, `.venv`, `uv sync`). Deps: pandas, pyyaml, pysam, cyvcf2, sklearn, matplotlib, tqdm, python-docx.
- **Nextflow:** 26.04.6 (installed to `~/.local/bin/nextflow`, Java 26). `workflows/nextflow.config` sets `local` profile (2–4 cpus, 2–8 GB per process; pipeline_info traces).
- **Optional system tools:** `fastqc`, `fastp`, `cutadapt`, `bcftools`, `samtools`, `vep`, `tabix` — pipeline gracefully degrades if missing (python stubs).

---

## 5. How Scoring Works (the ML)

For each variant `v`:

```python
# src/prioritization/features.py
rare_score       = 1.0 if no gnomAD AF else (1.0 if AF<0.001 else 0.8/0.3/0.05)
consequence_w    = map(CSQ → weight)  # frameshift 1.0 > missense 0.8 > synonymous 0.3 > intronic 0.2 ...
cadd_norm        = CADD/30 clipped 0-1, or 0.7 if high consequence else 0.35 (heuristic)
clinvar_score    = 1.0 pathogenic, 0.5 unknown, 0.1 benign
quality_score    = 0.4*QD_norm +0.3*MQ_norm +0.2*FS_inv +0.1*FILTER (PASS=1 else 0.2)

pathogenicity = 0.3*rare +0.3*consequence +0.25*cadd +0.1*clinvar +0.05*quality  # → 0-1

# src/phenotype/hpo.py
phenotype_score = Jaccard(query_HPO ∩ gene_HPO / union)  # baseline 0.5 (no gene map)
                  or 1.0 if gene in {BUB1B, TRIP13, CEP57, BUB3}  # MVA boost

final_score = 0.6*phenotype +0.4*pathogenicity  # from config
# then min-max calibrate to 0.01-0.99 for F-max (rank.py)
```

**Why frameshift tops?** Baseline without gene info → all phenos 0.5 → pathogenicity decides → frameshift weight 1.0 + high CADD heuristic 0.7 + rare 1.0 dominates. With real VEP + Phen2Gene, missense in BUB1B would outrank random frameshift.

---

## 6. Repository Anatomy

```
.
├── config/
│   ├── example_config.yaml  # weights, thresholds, reference paths
│   └── hpo_terms.yaml       # 8 curated HPOs (from docx)
├── data/
│   ├── *.fastq.gz (8)       # 80 GB WGS reads (gitignored via .gitignore data/)
│   ├── *.vcf.gz + .tbi      # 5M variants (Sentieon Haplotyper + GATK VariantFiltration)
│   ├── Challenge_Clinical_Phenotype_1.docx  # confidential — don't redistribute
│   └── .cache/huggingface/  # HF download cache
├── src/
│   ├── preprocessing/{qc.py, trimming.py}    # FASTQ QC stubs (fastqc/fastp or sampler)
│   ├── variant_calling/{caller.py, filtering.py}  # (stub — VCF already provided)
│   ├── annotation/{vep.py, population.py}   # VEP wrapper (real/dock/pass-through), gnomAD
│   ├── phenotype/hpo.py     # HPO load + Jaccard + gene boost
│   ├── prioritization/{prioritize.py, features.py}  # core scoring + CLI
│   └── ranking/{rank.py, metrics.py}        # rank + Rank Points / F-max
├── workflows/
│   ├── main.nf              # DSL2 entry (QC → annotate → prioritize)
│   ├── nextflow.config      # params + profiles + trace
│   └── modules/{qc,annotate,prioritize,trimming}.nf  # per-process scripts
│   + Snakefile               # alternative Snakemake (qc→annotate→prioritize rules)
├── notebooks/{01_exploratory.ipynb, 02_variant_prioritization.ipynb}
├── tests/{test_annotation.py, test_prioritization.py}
├── benchmarks/benchmark_ranking.py
├── reports/{track1_template.md, track2_template.md}
├── docs/{problem_statement.md, BASELINE_EXPLAINER.md (this)}
├── .venv/ + pyproject.toml  # uv env (pandas, pysam, etc.)
└── results/                 # generated after pipeline run
    ├── ranked_variants.csv  # 541 MB, 5M ranked (Track 1 submission)
    ├── qc/*.json            # 8 QC stubs
    ├── logs/prioritize.log  # full tqdm log
    └── pipeline_info/*      # timeline/report/trace (Nextflow)
```

---

## 7. Limitations of Baseline (Honest)

1. **No gene annotation:** Without VEP cache (`/data/vep_cache`), every `SYMBOL` is `.` and phenotype is neutral 0.5 — ranking is just pathogenicity heuristic. Real improvement = install VEP 113 GRCh38 cache (~20 GB) + run `vep --cache --offline --af_gnomad --clin_sig --symbol`.
2. **Consequence inferred:** Without CSQ, we guess `frameshift` for indels, `missense` for SNVs — wrong for splicing/UTR etc. Real VEP gives precise SO terms.
3. **Population AF missing:** Provided VCF's `AF` is cohort (AN=2) not gnomAD; we treat missing as rare (1.0). Real gnomAD annotation via VEP or `bcftools annotate` would filter common variants properly.
4. **Phenotype naïve:** Jaccard 0.5 + MVA gene hard-coded boost. Real = Phen2Gene/Exomiser with HPO ontology ancestors + gene–phenotype DB.
5. **Multiallelic expansion double-counts:** We split `ALT=A,C` into two rows — scoring identical context but may duplicate top. Fine for baseline, but could normalize.
6. **Quality heuristics:** QD/MQ/FS scoring is heuristic; true VQSR would be better.

---

## 8. How to Improve (Next Steps Ordered by Impact)

1. **Install VEP + cache:** `vep_install --AUTO c --SPECIES homo_sapiens --ASSEMBLY GRCh38 --CACHE_VERSION 113` to `/data/vep_cache`, then rerun `nextflow` → gene + consequence become real, phenotype boost starts working. Validate: `grep BUB1B results/ranked_variants.csv | head`.
2. **Add gnomAD AF:** Via VEP `--af_gnomad` (already in cmd) or `bcftools annotate -a gnomad.vcf.gz`.
3. **Phenotype:** Plug Phen2Gene API or `src/phenotype/hpo.py` gene map from HPO `genes_to_phenotype.txt` + ontology similarity (Resnik).
4. **CADD/REVEL:** Add `CADD.tsv.gz` custom annotation or via `vep --plugin CADD`.
5. **Call-level features:** Parse `AD, DP, GQ, PL` for somatic mosaicism (MVA is mosaic aneuploidy — allele balance matters!).
6. **Aneuploidy:** Provided VCF is SNV/indel only; consider copy-number / mosaic aneuploidy from BAM (not in baseline).
7. **Model:** Train weighted ensemble or simple logistic regression on ClinVar labels, calibrate with Platt scaling for better F-max (currently min-max).
8. **Nephrocalcinosis + cancer cluster:** Instead of per-gene Jaccard, embed full HPO set with `phen2gene` or use Exomiser-style phenotype matching.

---

## 9. Reproducing / Tweaking

```bash
# 1. Setup (uv, python 3.12)
uv sync
uv run python -m src.prioritization.prioritize --help

# 2. Quick debug (5000 variants, ~1 sec)
uv run python -m src.prioritization.prioritize \
  --vcf data/WGS_EX2312012_HGWCNDSX7.vcf.gz \
  --hpo config/hpo_terms.yaml --config config/example_config.yaml \
  --output /tmp/ranked_small.csv --max-variants 5000
head /tmp/ranked_small.csv

# 3. Full Nextflow (80GB FASTQ + 5M VCF, ~80 sec prioritize)
nextflow run workflows/main.nf -profile local              # both QC + VCF
nextflow run workflows/main.nf -profile local --skip_qc true  # VCF-only, faster
nextflow run workflows/main.nf -profile local --max_variants 100000  # debug slice

# 4. Inspect
wc -l results/ranked_variants.csv   # 5079065
head -20 results/ranked_variants.csv
cat results/qc/*.json | head -40
cat results/logs/prioritize.log | tail -40
nextflow log .nextflow.log | tail

# 5. Adjust weights
vim config/example_config.yaml  # phenotype_weight 0.6 → 0.8
nextflow run workflows/main.nf -profile local -resume  # resumes cached QC/annotate

# 6. Snakemake alternative
snakemake -s workflows/Snakefile -n  # dry run

# 7. Evaluation (Rank Points)
uv run python -c "from src.ranking.metrics import rank_points; print(rank_points(1), rank_points(5), rank_points(11))"
# pytest
uv run pytest -q
```

---

## 10. Submission & Ethics

- **What to submit (Track 1):** `results/ranked_variants.csv` with `CHROM,POS,REF,ALT,RANK,SCORE` (+ optional extras) + GitHub link + writeup (`reports/track1_template.md`). Upload via HF Space → Submit - Track 1 (max 6).
- **Data use:** Real patient, WCG IRB #20252010. Don't re-identify. Outputs CC BY 4.0, not medical advice.
- **Secondary findings:** Welcome but don't affect automated score; report responsibly.
- **If you find causal:** Likely in `BUB1B/TRIP13/CEP57/BUB3` pathway (mitotic checkpoint). Validate with literature + ClinVar before claiming.

---

## 11. FAQ for DS

**Q: Is VCF like a CSV?** Yes, but tab-delimited + gzipped + tbi-indexed, with header metadata. Use `pysam` or `cyvcf2` (python) or `bcftools` (CLI) to parse — not `pd.read_csv` directly because of `#` headers.

**Q: Why 8 FASTQs not 1?** Sequencer ran 4 lanes × paired-end (R1/R2) = 8 files. Merge per sample after QC.

**Q: Do I need to re-align FASTQ?** Not for baseline — provided VCF is already called (Sentieon Haplotyper). Re-alignment (BWA → BAM → re-call) is optional/advanced and heavy.

**Q: What does FILTER PASS vs QD2/MQ40 mean?** PASS = good; QD2/MQ40 = low confidence flagged by GATK; baseline down-weights but keeps them (some true rare variants are low-qual).

**Q: F-max vs Accuracy?** F-max is threshold-aware: you must output well-calibrated probabilities so the best cutoff yields high precision+recall. Just ranking #1 correctly but giving it 0.51 vs 0.99 matters.

**Q: Why is `results/ranked_variants.csv` 541 MB?** 5M rows × 14 cols, each with SCORE + extra diagnostics. For submission you could strip extras to 6 cols (`cut -d, -f1-6`).

---

*Generated 2026-09-04 after baseline `nextflow run` — prioritize 5,079,064 alleles, QC 8 files. See `.nextflow.log`, `results/pipeline_info/*`, and `results/logs/prioritize.log` for provenance.*

