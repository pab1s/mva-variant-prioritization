# data/ — local-only, NEVER committed (see ../.gitignore)

## Layout

```
data/
  raw/
    fastq/       8 WGS FASTQs (4 lanes x R1/R2, ~80 GB) — controlled access
    calls/       Provided Sentieon VCF + .tbi (5,012,204 variants) — controlled access
    phenotype/   Challenge_Clinical_Phenotype_1.docx — confidential, do not redistribute
  references/    GRCh38.fa + .fai (+ no-alt .gz source)
  interim/       Expensive intermediates: full.vep.vcf.gz (5M VEP-annotated) + .tbi
  vep_cache/     Ensembl VEP 113 GRCh38 cache (extracted homo_sapiens/ + tmp/ downloads)
  cadd/          CADD v1.7 GRCh38 pre-scored files (whole_genome_SNVs + indel + .tbi)
  .cache/        Hugging Face download cache
```

## Provenance

- Raw genome + phenotype: MVA hackathon dataset (WCG IRB #20252010). Request
  access via the HF Space dataset page; governed by its Data Use Agreement.
- `references/GRCh38.fa`: standard human reference (GRCh38, no alt contigs).
- `vep_cache/`: `homo_sapiens_vep_113_GRCh38.tar.gz` from
  https://ftp.ensembl.org/pub/release-113/variation/vep/
- `cadd/`: v1.7 GRCh38 files from https://cadd.gs.washington.edu/download

## Privacy

Everything under `raw/` is real patient data. Do not copy it out of this
machine, do not commit it, do not attempt re-identification. Derived
outputs (ranked variant lists) may contain variant-level information —
treat them with the same care until de-identified for submission.
