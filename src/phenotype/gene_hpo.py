"""Gene/disease ↔ HPO annotations (HPOA `genes_to_phenotype.txt`) + similarity.

Loads, filters and indexes HPOA annotations so the ranker can compute
*generic* (disease-agnostic) phenotype features:

    pheno_resnik            normalized max-Resnik(query HPO, gene HPO)
    pheno_bma               frequency-weighted best-match average (normalized)
    pheno_jaccard           plain Jaccard on raw term sets
    pheno_terms_matched     #query terms with a non-zero match (clipped)
    pheno_excluded_overlap  fraction of gene terms present in query EXCLUDED set
    gene_hpo_count          #phenotype terms annotated to the gene (log-scaled)
    gene_disease_count      #diseases annotated to the gene (log-scaled)

MOI (mode of inheritance) is derived from the inheritance HPO terms present in
the annotation file itself (HP:0000006=AD, HP:0000007=AR, ... verified) and is
NOT part of the phenotype-similarity term sets.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

from src.phenotype.hpo_ontology import (
    EXCLUDED_TERM,
    FREQ_TERMS,
    HPOOntology,
)

# Inheritance HPO terms seen in HPOA (verified from the local file).
MOI_TERMS: dict[str, str] = {
    "HP:0000006": "AD",   # Autosomal dominant inheritance
    "HP:0000007": "AR",   # Autosomal recessive inheritance
    "HP:0001417": "XL",   # X-linked inheritance
    "HP:0001419": "XLR",  # X-linked recessive
    "HP:0001423": "XLD",  # X-linked dominant
    "HP:0001427": "MT",   # Mitochondrial
    "HP:0001450": "YL",   # Y-linked
    "HP:0010984": "DIG",  # Digenic
    "HP:0010983": "OLI",  # Oligogenic
    "HP:0001426": "NON",  # Non-Mendelian
    "HP:0012275": "AD",   # AD with maternal imprinting
    "HP:0034341": "AR",   # Pseudoautosomal recessive
}


def _freq_weight(value: str) -> float:
    """HPOA frequency column -> weight in [0, 1].

    Accepts frequency HPO terms (HP:0040280..284), ratios ("1/2"), or "-".
    """
    v = (value or "").strip()
    if v in FREQ_TERMS:
        return FREQ_TERMS[v]
    if v == EXCLUDED_TERM:
        return 0.0
    if "/" in v:
        try:
            a, b = v.split("/")[:2]
            return float(a) / float(b) if float(b) > 0 else 0.5
        except (ValueError, ZeroDivisionError):
            return 0.5
    return 0.5  # unknown ("-") -> neutral


class GeneHPO:
    """Indexed HPOA annotations + query-aware gene similarity."""

    def __init__(self, g2p_path: str | Path, ontology: HPOOntology):
        self.ontology = ontology
        self.gene_terms: dict[str, set[str]] = {}
        self.gene_diseases: dict[str, set[str]] = defaultdict(set)
        self.disease_terms: dict[str, set[str]] = defaultdict(set)
        self.disease_genes: dict[str, set[str]] = defaultdict(set)
        self.disease_freq: dict[tuple[str, str], float] = {}
        self.gene_moi: dict[str, set[str]] = defaultdict(set)
        self.disease_moi: dict[str, set[str]] = defaultdict(set)
        self._load(g2p_path)
        # IC from gene annotations (propagated inside compute_ic).
        ontology.compute_ic(self.gene_terms)
        self._gene_closure: dict[str, frozenset[str]] = {}

    # ------------------------------------------------------------------ load
    def _load(self, path: str | Path) -> None:
        with open(path) as fh:
            header = fh.readline().rstrip("\n").split("\t")
            col = {name: i for i, name in enumerate(header)}
            need = {"gene_symbol", "hpo_id", "hpo_name", "frequency", "disease_id"}
            if not need.issubset(col):
                raise RuntimeError(
                    f"genes_to_phenotype header mismatch: {header} (need {need})"
                )
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < len(header):
                    continue
                gene = parts[col["gene_symbol"]].strip()
                term = parts[col["hpo_id"]].strip()
                name = parts[col["hpo_name"]].strip()
                freq = parts[col["frequency"]].strip()
                disease = parts[col["disease_id"]].strip()
                if not gene or not term.startswith("HP:"):
                    continue
                if term in MOI_TERMS:
                    self.gene_moi[gene].add(MOI_TERMS[term])
                    if disease:
                        self.disease_moi[disease].add(MOI_TERMS[term])
                    continue
                if term == EXCLUDED_TERM or term in FREQ_TERMS:
                    continue
                if "inheritance" in name.lower():
                    continue
                if not self.ontology.is_known(term):
                    continue  # obsolete/unknown term — drop from similarity sets
                self.gene_terms.setdefault(gene, set()).add(term)
                if disease:
                    self.gene_diseases[gene].add(disease)
                    self.disease_terms[disease].add(term)
                    self.disease_genes[disease].add(gene)
                    key = (disease, term)
                    w = _freq_weight(freq)
                    if w > self.disease_freq.get(key, 0.0):
                        self.disease_freq[key] = w
        if not self.gene_terms:
            raise RuntimeError(f"No usable gene-HPO annotations in {path!r}")

    # ------------------------------------------------------------- accessors
    def query_weights(self, disease: str | None) -> dict[str, float]:
        """Per-term frequency weights for a disease-derived query."""
        if not disease:
            return {}
        return {
            t: self.disease_freq.get((disease, t), 0.5)
            for t in self.disease_terms.get(disease, set())
        }

    def _closure(self, gene: str) -> frozenset[str]:
        cached = self._gene_closure.get(gene)
        if cached is not None:
            return cached
        closure: set[str] = set()
        for t in self.gene_terms.get(gene, ()):
            closure |= self.ontology.ancestors(t)
        out = frozenset(closure)
        self._gene_closure[gene] = out
        return out

    # ------------------------------------------------------------- similarity
    def gene_similarity(
        self,
        query_terms: set[str],
        gene: str,
        norm: float,
        query_weights: dict[str, float] | None = None,
        excluded_terms: set[str] | None = None,
    ) -> dict[str, float]:
        """Phenotype feature block for (query, gene), normalized to ~[0,1].

        `norm` = ontology.query_norm_factor(query_terms) — pass the SAME value
        for every gene of a query so scores are comparable.
        """
        terms = self.gene_terms.get(gene)
        zero = {
            "pheno_resnik": 0.0,
            "pheno_bma": 0.0,
            "pheno_jaccard": 0.0,
            "pheno_terms_matched": 0.0,
            "pheno_excluded_overlap": 0.0,
            "gene_hpo_count": 0.0,
            "gene_disease_count": 0.0,
        }
        if not terms:
            return zero
        sim = self.ontology.similarity(query_terms, terms, query_weights)
        if excluded_terms:
            inter_ex = len(terms & excluded_terms)
            excl = inter_ex / len(terms)
        else:
            excl = 0.0
        jaccard = (
            len(query_terms & terms) / len(query_terms | terms)
            if query_terms
            else 0.0
        )
        return {
            "pheno_resnik": min(sim["resnik_max"] / norm, 1.0) if norm else 0.0,
            "pheno_bma": min(sim["resnik_bma"] / norm, 1.0) if norm else 0.0,
            "pheno_jaccard": jaccard,
            "pheno_terms_matched": min(sim["terms_matched"] / 10.0, 1.0),
            "pheno_excluded_overlap": excl,
            "gene_hpo_count": min(math.log1p(len(terms)) / math.log1p(500), 1.0),
            "gene_disease_count": min(
                math.log1p(len(self.gene_diseases.get(gene, ()))) / math.log1p(100), 1.0
            ),
        }

    def genes_with_similarity(
        self, query_terms: set[str], min_norm_resnik: float = 0.0
    ) -> dict[str, float]:
        """All annotated genes with normalized max-Resnik > threshold.

        Used by candidate generation's phenotype gate (union rule) and by
        per-query normalization checks. Returns gene -> normalized resnik.
        """
        norm = self.ontology.query_norm_factor(query_terms)
        out: dict[str, float] = {}
        for gene, terms in self.gene_terms.items():
            sim = self.ontology.similarity(query_terms, terms)
            v = sim["resnik_max"] / norm if norm else 0.0
            if v > min_norm_resnik:
                out[gene] = v
        return out
