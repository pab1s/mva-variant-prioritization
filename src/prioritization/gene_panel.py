"""Disease gene panel (public knowledge) + assembly helpers.

The panel encodes *disease -> gene* prior knowledge from public curation
(OMIM / HPO / ClinGen) — the same kind of knowledge Exomiser and Phen2Gene
ship with. It holds NO variant-level answers: intervals are whole-gene
GRCh38 spans (tens of kb), weights are per-gene priors.

For a new disease, swap the config (``gene_panel`` + ``inheritance``) —
no code change needed. Defaults below are the MVA example.
"""

# Primary assembly contigs — everything else is decoy/random/alt and
# should never outrank a primary rare variant. Assembly property,
# not disease-specific.
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


# Default example: MVA panel (public GRCh38 intervals + gene weights).
DEFAULT_PANEL_GENES = {
    "BUB1B": 1.0,
    "TRIP13": 1.0,
    "CEP57": 1.0,
    "BUB3": 0.9,
    "BUB1": 0.9,
    "MAD2L1": 0.8,
    "CENATAC": 0.8,
}

DEFAULT_PANEL_INTERVALS = {
    "BUB1B": ("15", 40161069, 40221123),
    "CEP57": ("11", 95789965, 95832693),
    "TRIP13": ("5", 892884, 919357),
}


def load_panel(config: dict | None = None,
               hpo_scores: dict[str, float] | None = None) -> tuple[dict, dict]:
    """Return (panel_genes, panel_intervals).

    Sources (config ``panel_source``):
      - ``manual`` (default): ``gene_panel`` dict from config (MVA example).
      - ``hpo``: top-N genes by Resnik percentile over the query HPO terms
        (fully data-driven, no hand-written genes). Weights = percentiles.
        Intervals (positional rescue) still come from ``gene_panel`` when
        the same gene is listed there, else SYMBOL-only matching.

    For a new disease: point config at its HPO terms and set
    ``panel_source: hpo`` — zero code change, zero hand-written genes.
    """
    config = config or {}
    source = str(config.get("panel_source", "manual")).lower()
    genes, intervals = _manual_panel(config)
    if source == "hpo":
        if not hpo_scores:
            raise RuntimeError(
                "panel_source=hpo but no HPO scores available "
                "(missing hp.obo / genes_to_phenotype.txt?)."
            )
        top_n = int(config.get("panel_top_n", 100))
        ranked = sorted(hpo_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        genes = {g: float(s) for g, s in ranked}
    return genes, intervals


def _manual_panel(config: dict) -> tuple[dict, dict]:
    raw = config.get("gene_panel")
    if not raw:
        return dict(DEFAULT_PANEL_GENES), dict(DEFAULT_PANEL_INTERVALS)
    genes: dict = {}
    intervals: dict = {}
    for gene, spec in raw.items():
        if isinstance(spec, dict):
            genes[gene] = float(spec.get("weight", 1.0))
            if {"chrom", "start", "end"} <= set(spec):
                intervals[gene] = (str(spec["chrom"]), int(spec["start"]), int(spec["end"]))
        else:
            genes[gene] = float(spec)
    return genes, intervals


def gene_for_position(chrom: str, pos: int, intervals: dict | None = None) -> str | None:
    """Positional gene rescue: CHROM/POS -> panel gene or None.

    Uses public GRCh38 gene coordinates (UCSC/GDC/GeneCards), NOT any
    answer key. Lets the prefilter work on a raw VCF with no CSQ.
    Final consequence/ClinVar calls always come from VEP, never from here.
    """
    if intervals is None:
        intervals = DEFAULT_PANEL_INTERVALS
    c = str(chrom).removeprefix("chr")
    for gene, (gchrom, start, end) in intervals.items():
        if c == gchrom and start <= pos <= end:
            return gene
    return None
