"""Candidate generation (retrieval stage): cheap high-recall prefilter.

Runs inline over the 5M stream. Anything passing is ML-scored; the rest
keeps the heuristic score. Recall matters more than precision here —
a causal variant filtered now can never be recovered downstream.
"""

HIGH_MODERATE = {"HIGH", "MODERATE"}


def is_candidate(feats: dict, gene: str | None, mva_genes: set[str],
                 filt: str, is_primary: bool) -> bool:
    """High-recall rule: primary contig + rare + one suspicious signal."""
    if not is_primary:
        return False
    af = feats.get("af")
    if af is not None and af >= 0.01:
        return False
    if filt not in ("PASS", ".", "", "None"):
        # keep low-qual only if ClinVar says pathogenic
        if feats.get("clinvar_score", 0.5) < 1.0:
            return False
    if gene in mva_genes:
        return True
    if feats.get("consequence_weight", 0) >= 0.6:
        return True
    if feats.get("clinvar_score", 0.5) >= 1.0:
        return True
    return False
