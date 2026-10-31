"""Disease gene panel (data-driven) + assembly helpers.

The panel is a *prior*: genes whose disruption could explain the case.
It is built per case from data, never hand-written into code:

  - ``panel_source: hpo`` (default): top-N genes by Resnik similarity
    between the case HPO terms and public gene annotations
    (HPO project: hp.obo + genes_to_phenotype.txt).
  - ``panel_source: manual``: explicit ``gene_panel`` dict in config, for
    when a clinician already has a differential (documented, auditable).

Contig classification (primary vs decoy) is assembly knowledge, not
disease knowledge, and stays hardcoded: decoys must never outrank loci.
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


def load_panel(config: dict | None = None,
               hpo_scores: dict[str, float] | None = None) -> tuple[dict, dict]:
    """Return (panel_genes, panel_intervals). No hard-coded disease genes.

    Sources (config ``panel_source``, default ``hpo``):
      - ``hpo``: top-N genes by Resnik percentile over the query HPO terms.
        Weights = percentiles. No hand-written genes anywhere.
      - ``manual``: explicit ``gene_panel`` dict from config, e.g.
        ``GENE1: {weight: 1.0, chrom: "1", start: 1000, end: 2000}``.
        Intervals enable positional rescue on raw VCFs without CSQ;
        without them, matching is SYMBOL-only (fine for VEP output).
    """
    config = config or {}
    source = str(config.get("panel_source", "hpo")).lower()
    if source == "manual":
        return _manual_panel(config)
    if not hpo_scores:
        raise RuntimeError(
            "panel_source=hpo but no HPO scores available "
            "(missing hp.obo / genes_to_phenotype.txt?)."
        )
    top_n = int(config.get("panel_top_n", 100))
    ranked = sorted(hpo_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    genes = {g: float(s) for g, s in ranked}
    _, intervals = _manual_panel(config)
    return genes, intervals


def _manual_panel(config: dict) -> tuple[dict, dict]:
    raw = config.get("gene_panel") or {}
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

    Only fires for genes with coordinates in the manual panel config.
    Final consequence/ClinVar calls always come from VEP, never from here.
    """
    if not intervals:
        return None
    c = str(chrom).removeprefix("chr")
    for gene, (gchrom, start, end) in intervals.items():
        if c == gchrom and start <= pos <= end:
            return gene
    return None
