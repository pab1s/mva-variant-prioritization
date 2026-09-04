"""HPO term handling and phenotype similarity."""
from typing import List
import yaml
from pathlib import Path


def load_hpo_terms(hpo_file: str) -> List[str]:
    """Load HPO terms from YAML (config/hpo_terms.yaml).

    Expected format:
        hpo_terms:
          - \"HP:0002859\"
          - \"HP:0000121\"
    Returns list of HP IDs (strings). Empty list if file missing/empty.
    """
    p = Path(hpo_file)
    if not p.exists():
        return []
    with open(p) as f:
        data = yaml.safe_load(f) or {}
    terms = data.get("hpo_terms") or []
    # filter None/comments, ensure string
    return [str(t).strip() for t in terms if t and str(t).strip().startswith("HP:")]


def phenotype_similarity(query_terms: List[str], disease_terms: List[str]) -> float:
    """Compute simple Jaccard-style phenotype similarity.

    Real pipelines use Phen2Gene / Exomiser with ontology-aware similarity
    (Resnik, information content). For baseline we use set overlap:

        similarity = |query ∩ disease| / |query ∪ disease|

    Returns 0.0-1.0. If disease_terms empty → neutral 0.5.

    Example:
        query = [HP:0002859, HP:0004322]  # rhabdomyosarcoma, short stature
        disease_gene_terms = [HP:0002859]  # gene associated with rhabdomyosarcoma
        → 0.5
    """
    if not disease_terms:
        return 0.5
    q = set(query_terms)
    d = set(disease_terms)
    if not q and not d:
        return 0.5
    inter = len(q & d)
    union = len(q | d)
    return inter / union if union else 0.0


def hpo_overlap_score(query_terms: List[str], gene_hpo_map: dict | None = None, gene: str | None = None) -> float:
    """Gene-level phenotype score used in prioritization.

    If we have a gene → HPO map (e.g. from HPO annotations or Phen2Gene),
    compute similarity. For baseline without gene map, return neutral 0.5
    so pathogenicity drives ranking, but still weighted by phenotype.

    Future: plug in real Phen2Gene API or HPO GAF.
    """
    if gene and gene_hpo_map and gene in gene_hpo_map:
        return phenotype_similarity(query_terms, gene_hpo_map[gene])
    # baseline: no gene info → neutral
    return 0.5
