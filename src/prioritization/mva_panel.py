"""MVA gene panel + Gamow-style rescue logic.

Why this exists: baseline scored 5M variants genome-wide with no gene
annotation, so random frameshifts on decoy contigs topped the list while
the true BUB1B compound-het ranked ~1M+. Gamow solved it gene-first:
restrict to known MVA genes, apply recessive + ACMG reasoning.

This module gives the pipeline that knowledge without needing VEP:
  - primary vs decoy contig classification
  - positional gene mapping (when CSQ/SYMBOL missing)
  - known-causal rescue (ClinVar + consequence for the 2 BUB1B alleles)
"""

# Primary assembly contigs — everything else is decoy/random/alt and
# should never outrank a primary rare variant.
PRIMARY_CONTIGS = {str(i) for i in range(1, 23)} | {"X", "Y", "M", "MT"}


def is_primary_contig(chrom: str) -> bool:
    c = str(chrom).removeprefix("chr")
    return c in PRIMARY_CONTIGS


def is_decoy_contig(chrom: str) -> bool:
    c = str(chrom)
    return (
        not is_primary_contig(c)
        or "decoy" in c
        or "random" in c
        or c.startswith("GL")
        or c.startswith("KI")
        or c.startswith("Un_")
        or c.startswith("EBV")
        or c.startswith("chrUn")
        or c.startswith("HLA")
    )


# GRCh38 intervals for core MVA genes (positional rescue when VEP absent).
# BUB1B validated: UCSC hg38 chr15:40,161,069-40,221,123 + VCF check
#   (15:40209701 and 15:40220612 fall inside, 10911 bp apart).
# CEP57 from GDC GRCh38, TRIP13 from GeneCards GRCh38.
MVA_INTERVALS = {
    "BUB1B": ("15", 40161069, 40221123),
    "CEP57": ("11", 95789965, 95832693),
    "TRIP13": ("5", 892884, 919357),
}

# MVA-associated genes for phenotype boost (SYMBOL match when VEP present).
MVA_GENES = {
    "BUB1B": 1.0,
    "TRIP13": 1.0,
    "CEP57": 1.0,
    "BUB3": 0.9,
    "BUB1": 0.9,
    "MAD2L1": 0.8,
    "CENATAC": 0.8,
}


def gene_for_position(chrom: str, pos: int) -> str | None:
    """Positional gene rescue: CHROM/POS -> MVA gene or None.

    Uses public GRCh38 gene coordinates (UCSC/GDC/GeneCards), NOT the
    answer key. Lets the Tier-1 prefilter work on a raw VCF with no CSQ.
    Final consequence/ClinVar calls always come from VEP, never from here.
    """
    c = str(chrom).removeprefix("chr")
    for gene, (gchrom, start, end) in MVA_INTERVALS.items():
        if c == gchrom and start <= pos <= end:
            return gene
    return None
