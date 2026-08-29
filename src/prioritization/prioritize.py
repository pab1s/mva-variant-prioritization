"""Core prioritization logic combining annotation + phenotype."""
import pandas as pd

def prioritize(vcf_annotated: str, hpo_terms: list[str], config: dict | None = None) -> pd.DataFrame:
    """Return ranked variants DataFrame."""
    raise NotImplementedError
