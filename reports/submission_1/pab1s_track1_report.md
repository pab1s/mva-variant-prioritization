# Track 1 Report — VEP + CADD + ClinVar + HPO-Resnik ranking (pab1s)

## Approach

Single-sample WGS VCF (5,012,204 variants) plus 8 curated HPO terms.
Every variant allele (5,079,064 after multiallelic expansion) is annotated
with Ensembl VEP 113 (canonical/MANE transcripts, HGVS via GRCh38 FASTA),
enriched with precomputed CADD v1.7 scores through the VEP plugin and
ClinVar significance through a tabix-indexed custom track, then scored
continuously as `0.6·phenotype + 0.4·pathogenicity − 0.5·decoy-contig
+ 0.1·recessive-pair`, where pathogenicity blends gnomAD rarity, VEP
consequence severity, normalized CADD, ClinVar status and call quality.
Phenotype is a Resnik rank-percentile similarity between the case HPO
terms and 5,277 gene annotation sets over the HPO DAG, and the gene
panel defaults to the top-100 Resnik genes, so no disease genes are
hand-written. Ranking is by min-max calibrated score (F-max aware).

## Rationale and evidence

The top hit, chr15:40209701 T>G in BUB1B (stop-gain p.Leu737Ter, exon
17/23, CADD 36, ClinVar Pathogenic/Likely-pathogenic), outranks the field
on pathogenicity alone. The second hit, chr15:40220612 T>G in BUB1B
(missense p.Asn1002Lys, CADD 24.5, ClinVar VUS), is invisible without
gene-level reasoning (rank 467 on pathogenicity alone) and rises to rank 2
only through the HPO-derived BUB1B match plus the recessive pair pattern
(≥2 rare hets in one panel gene). Together they form the
compound-heterozygous MVA1 hypothesis (estimated causal probability 0.85);
phase is unproven in this singleton and needs parental segregation, and
the VUS needs a functional spindle-checkpoint assay. Secondary ClinVar
pathogenic findings (TP53 het, SLC12A1 hom, ABCC6 hom) are reported as
incidental, explicitly not causal for MVA.

## Reproducibility

Code, configs and workflows: https://github.com/pab1s/mva-variant-prioritization
(branch track_01). Full re-run from the provided VCF is ~2.5 h on a
16-core machine. All annotation databases are public (Ensembl, CADD,
ClinVar, HPO, GRCh38, gnomAD); no proprietary data was used.

## AI assistance disclosure

Analyses in this session were performed with AI assistance (OpenCode
harness; model family Meta Muse Spark). All computation ran locally on
the analyst machine — no patient data was uploaded anywhere — and no
answer-key information was used; a no-panel ablation (missense at rank
467) is documented in the repository as the honest blind baseline.
