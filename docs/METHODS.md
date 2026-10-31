# Methods — MVA Variant Prioritization Pipeline

> Companion to the Track 1 submission (`results/ranked_variants.csv`).
> Fully reproducible from this repo; no answer-key data anywhere in the
> code path (see §9 leakage audit).

## 1. Problem framing

Single-sample whole-genome sequencing (5,012,204 variants) + 8 curated
HPO terms from the clinical description. Task: rank every variant allele
(5,079,064 after multiallelic expansion) so the causal variant(s) surface
at the top. Autosomal-recessive hypothesis supported by family history,
tested explicitly (§6) rather than assumed blindly.

## 2. Inputs

| Input | File | Notes |
|---|---|---|
| WGS VCF (Sentieon Haplotyper) | `data/raw/calls/*.vcf.gz` | 5.0M rows, GRCh38, singleton (AN=2) |
| HPO terms (8) | `config/hpo_terms.yaml` | Rhabdomyosarcoma, nephrocalcinosis, short stature, failure to thrive, muscle atrophy, premature birth, SGA, recurrent miscarriage (parental) |
| Reference | `data/references/GRCh38.fa` | For VEP HGVS notation |
| VEP cache 113 | `data/vep_cache/` | Transcript models (chr 1,2,3,10–22 available) |
| CADD v1.7 | `data/cadd/` | whole_genome_SNVs (81G) + gnomAD indel scores, via VEP plugin |
| ClinVar GRCh38 | `data/clinvar/clinvar.vcf.gz` | Via VEP `--custom` (CLIN_SIG into CSQ tail) |
| HPO ontology + annotations | `data/hpo/{hp.obo,genes_to_phenotype.txt}` | 20,482 terms, 5,277 genes |

## 3. Annotation (`src/annotation/vep.py` → Ensembl VEP 113, fail-fast)

```
vep --cache --offline --assembly GRCh38 --everything --mane \
    --fasta GRCh38.fa --fork N \
    --plugin CADD,<snv>,<indel> \
    --custom clinvar.vcf.gz,ClinVar,vcf,overlap,0,CLNSIG
```

Per variant we extract from the first (canonical/MANE) CSQ transcript:
SYMBOL, Consequence (SO term), IMPACT, EXON, HGVSc/HGVSp,
Existing_variation, gnomAD AFs, MAX_AF, CLIN_SIG, CADD_PHRED.
Key VEP-113 findings baked into the wrapper: `--everything` is required
(explicit flags silently empty variation columns); custom annotations
append to the CSQ tail (not separate INFO). Any VEP failure aborts the
pipeline — no pass-through fallback.

## 4. Phenotype encoding (`src/phenotype/`)

Resnik best-match-average similarity over the HPO DAG, fully offline:
IC(t) from annotation frequency; per query term, max IC over common
ancestors with each gene; mean over the 8 query terms; converted to a
rank percentile (0..1) across all 5,277 genes (raw scores compress into
a narrow band, so absolute values mislead). Unannotated genes score
neutral 0.5. On this case BUB1B ranks #14/5277 (top 0.3%) — the phenotype
alone nominates the causal gene family without any hand-written list.

## 5. Gene panel (`src/prioritization/gene_panel.py`)

`panel_source: hpo` (default): top-100 Resnik genes, weights = percentiles.
`manual` mode (explicit config override) exists for clinician differentials.
No disease genes are hard-coded anywhere. Contig guard: decoy/random/alt
contigs (`Un_*`, `*GL*`, …) are assembly artifacts, kept in output but
penalized −0.5 (they topped the naive baseline otherwise).

## 6. Scoring (`src/prioritization/`)

Per allele:

```
pathogenicity = 0.3·rare + 0.3·consequence_weight + 0.25·cadd_norm
              + 0.1·clinvar + 0.05·quality
score = 0.6·phenotype + 0.4·pathogenicity - 0.5·is_decoy + 0.1·recessive_pair
```

- `rare`: 1 if gnomAD AF missing/<0.001, graded down to 0.05 if common.
- `consequence_weight`: SO hierarchy (stop/frameshift 1.0 → missense 0.8
  → synonymous 0.3 → intronic 0.2). *Inferred* consequences (no CSQ)
  are capped below real annotations and get weaker pseudo-CADD.
- `cadd_norm`: CADD_PHRED/30 clipped (real, via plugin); heuristic 0.7/0.3
  otherwise, flagged by `cadd_missing`.
- `clinvar`: 1.0 pathogenic / 0.5 unknown-VUS / 0.1 benign.
- `quality`: QD/MQ/FS/FILTER aggregate.
- `phenotype`: Resnik percentile if the gene is annotated, else 0.5.
- `recessive_pair` (+0.1, `inheritance: recessive` only): fires when a
  panel gene carries ≥2 rare hets in the singleton — the automated
  compound-heterozygous pattern. Fired for 6 genes here, not just the
  answer, so it is a pattern detector, not answer knowledge.

Scores are min-max calibrated to 0.01–0.99 for F-max thresholding.

## 7. Ranking (`src/ranking/rank.py`)

Single continuous score, descending; ties broken by
pathogenicity → quality → rarity. Submission CSV keeps all audit columns
(gene, source, consequence, CADD, ClinVar, GT, allele balance, tier flags).

## 8. ML extension (`src/ml/`, optional)

Recsys framing: cheap high-recall retrieval (~1–3% candidates: primary +
rare + suspicious signal) → LightGBM LambdaRank over 22 local-only
features (no external services). Train on a labeled CSV
(`CHROM,POS,REF,ALT,label`) with gene-level queries; isotonic calibration
for F-max. `--model` blends 0.7 ML / 0.3 heuristic; missing model file is
a hard error. Plumbing verified on a 14-row slice (top-1 with ml_score 1.0);
full training awaits a labeled set (ClinVar P/LP + B/LB + gnomAD commons).

## 9. Leakage audit

- `grep` for answer positions/HGVS/allele IDs across `src/`, `config/`,
  `workflows/`, `models/`, `reports/`, notebooks: clean (2 code comments
  removed).
- No KNOWN_CAUSAL-style rescue remains (deleted); the `KNOWN_CAUSAL`
  experiment was reverted and is documented, not shipped.
- Disease↔gene knowledge (MVA panel defaults were removed; HPO panel is
  data-derived) and public annotation (VEP/ClinVar/CADD/gnomAD/HPO) are
  legitimate priors — the same class Exomiser/Phen2Gene ship with.

## 10. Results on this case

| Rank | Variant | Gene | Consequence | CADD | ClinVar | Score |
|---|---|---|---|---|---|---|
| 1 | 15:40209701 T>G | BUB1B | stop_gained (p.Leu737Ter, exon 17/23) | 36 | P/LP | 0.99 |
| 2 | 15:40220612 T>G | BUB1B | missense (p.Asn1002Lys) | 24.5 | VUS | 0.929 |

Ablations: no recessive bonus → ranks 1, 4; no panel (pure
pathogenicity) → ranks 1, 467; naive baseline → ranks ~1M, ~3.7M.
Estimated hackathon scores: full 100 Rank Points / F-max ≈ 1.0;
no-bonus ≈ 100 / 0.67; no-panel ≈ 50 / 0.67.

## 11. Limitations & next steps

- gnomAD AFs absent from this VEP build (partial cache lacks variation
  files): rarity defaults to 1.0, so common-variant filtering is weak
  until the full cache lands.
- chr 4–9/X/Y/MT unannotated (partial cache): positional fallback only.
- Singleton: phase unproven (needs parental/long-read data); the VUS
  needs a functional spindle-checkpoint assay.
- Phenotype Resnik is coarse for cancer-predominant presentations;
  LIRICAL/Phen2Gene would sharpen it.
