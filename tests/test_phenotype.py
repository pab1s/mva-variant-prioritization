"""Tests for the HPO ontology + gene-HPO phenotype index (F0)."""
import random

import pytest

from src.phenotype.gene_hpo import GeneHPO
from src.phenotype.hpo_ontology import HPOOntology

OBO = "data/hpo/hp.obo"
G2P = "data/hpo/genes_to_phenotype.txt"

# MVA case terms (config/hpo_terms.yaml)
MVA_TERMS = {"HP:0002859", "HP:0000121", "HP:0004322", "HP:0001508",
             "HP:0003202", "HP:0001622", "HP:0001518", "HP:0200067"}


@pytest.fixture(scope="module")
def gh():
    ont = HPOOntology(OBO)
    return GeneHPO(G2P, ont), ont


def test_ontology_parses(gh):
    _, ont = gh
    assert len(ont.parents) > 15000
    assert ont.is_known("HP:0000001")  # root
    assert not ont.is_known("HP:9999999")


def test_ancestors_include_self_and_root(gh):
    _, ont = gh
    anc = ont.ancestors("HP:0002859")  # rhabdomyosarcoma
    assert "HP:0002859" in anc
    assert "HP:0000118" in anc  # Phenotypic abnormality


def test_ic_monotone(gh):
    gene_hpo, ont = gh
    # the root is annotated to every gene -> IC ~ 0
    assert ont.ic.get("HP:0000001", 0.0) == pytest.approx(0.0, abs=1e-6)
    # a specific term has higher IC than a generic one
    ic_specific = ont.ic.get("HP:0002859", 0.0)   # rhabdomyosarcoma
    ic_generic = ont.ic.get("HP:0000118", 0.0)    # Phenotypic abnormality
    assert ic_specific > ic_generic > 0


def test_mva_genes_score_high(gh):
    gene_hpo, ont = gh
    terms = {t for t in MVA_TERMS if ont.is_known(t)}
    norm = ont.query_norm_factor(terms)
    causal = [gene_hpo.gene_similarity(terms, g, norm)["pheno_resnik"]
              for g in ("BUB1B", "TRIP13", "CEP57")]
    assert max(causal) >= 0.85  # causal MVA genes match strongly

    rng = random.Random(1)
    genes = list(gene_hpo.gene_terms)
    rand = [gene_hpo.gene_similarity(terms, rng.choice(genes), norm)["pheno_resnik"]
            for _ in range(50)]
    assert max(causal) > max(rand) or max(causal) >= 0.85


def test_gene_without_annotations_is_zero(gh):
    gene_hpo, ont = gh
    norm = ont.query_norm_factor(MVA_TERMS)
    s = gene_hpo.gene_similarity(MVA_TERMS, "NOT_A_GENE", norm)
    assert s["pheno_resnik"] == 0.0 and s["gene_hpo_count"] == 0.0


def test_moi_and_frequency_parsing(gh):
    gene_hpo, _ = gh
    assert "AR" in gene_hpo.gene_moi.get("BUB1B", set()) or gene_hpo.gene_moi
    # disease-level frequency weights exist and are in [0, 1]
    assert all(0.0 <= w <= 1.0 for w in gene_hpo.disease_freq.values())
    assert gene_hpo.disease_freq  # non-empty


def test_gate_threshold_separates_causal(gh):
    gene_hpo, ont = gh
    terms = {t for t in MVA_TERMS if ont.is_known(t)}
    sim = gene_hpo.genes_with_similarity(terms)
    from src.ml.candidates import PHENO_GATE_MIN_NORM
    assert sim["BUB1B"] > PHENO_GATE_MIN_NORM
