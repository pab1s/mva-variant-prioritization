"""HPO term handling and phenotype similarity."""
from typing import List

def load_hpo_terms(hpo_file: str) -> List[str]:
    """Load HPO terms from clinical description."""
    raise NotImplementedError

def phenotype_similarity(query_terms: List[str], disease_terms: List[str]) -> float:
    """Compute phenotype similarity (e.g., Phen2Gene, Exomiser-style)."""
    raise NotImplementedError
