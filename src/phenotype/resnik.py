"""Ontology-aware phenotype similarity (Resnik), fully offline.

Replaces the naive Jaccard overlap with information-content similarity over
the HPO DAG, using only local files:
  - data/hpo/hp.obo                 (ontology: terms + is_a DAG)
  - data/hpo/genes_to_phenotype.txt (gene -> HPO annotations, HPO project)

Method (standard Resnik, best-match average):
  IC(t)      = -ln(P(t)), P(t) = fraction of genes annotated to t or below
  sim(q, g)  = max IC over common ancestors of q and g  (per query term)
  score(G)   = mean_q sim(q, G) / max_IC  -> 0..1, gene G with terms

Genes without annotations score neutral 0.5 (same convention as before).
"""

import math
from pathlib import Path


def load_ontology(obo_path: str) -> dict[str, set[str]]:
    """Parse hp.obo -> {term_id: set(parent_ids)} for is_a links only."""
    parents: dict[str, set[str]] = {}
    cur: str | None = None
    with open(obo_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                cur = None
            elif line.startswith("id: HP:"):
                cur = line.split("id:")[1].strip()
                parents.setdefault(cur, set())
            elif line.startswith("is_a: HP:") and cur:
                # "is_a: HP:0001234 ! Name" -> parent id
                par = line.split("is_a:")[1].strip().split()[0]
                parents[cur].add(par)
                parents.setdefault(par, set())
    return parents


def _ancestors(term: str, parents: dict[str, set[str]], memo: dict[str, frozenset]) -> frozenset:
    if term in memo:
        return memo[term]
    seen, stack = {term}, [term]
    while stack:
        t = stack.pop()
        for p in parents.get(t, ()):
            if p not in seen:
                seen.add(p)
                stack.append(p)
    memo[term] = frozenset(seen)
    return memo[term]


def load_gene_annotations(path: str) -> dict[str, set[str]]:
    """Parse genes_to_phenotype.txt -> {GENE_SYMBOL: {HPO ids}}.

    Skips header (#...). Columns: ncbi_id, symbol, hpo, name, freq, disease.
    """
    genes: dict[str, set[str]] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            symbol, hpo = parts[1].strip(), parts[2].strip()
            if not symbol or symbol in ("-", "."):
                continue  # annotation rows without gene symbol
            if hpo.startswith("HP:"):
                genes.setdefault(symbol, set()).add(hpo)
    return genes


def build_ic(gene_terms: dict[str, set[str]],
             parents: dict[str, set[str]]) -> dict[str, float]:
    """IC per term from annotation frequency (terms + descendants)."""
    memo: dict[str, frozenset] = {}
    # gene -> full ancestor closure of its terms
    gene_anc: dict[str, set[str]] = {}
    for g, terms in gene_terms.items():
        s: set[str] = set()
        for t in terms:
            s |= set(_ancestors(t, parents, memo))
        gene_anc[g] = s
    n = max(len(gene_anc), 1)
    counts: dict[str, int] = {}
    for s in gene_anc.values():
        for t in s:
            counts[t] = counts.get(t, 0) + 1
    return {t: -math.log(c / n) for t, c in counts.items()}, gene_anc


def resnik_gene_scores(query_terms: list[str],
                       gene_terms: dict[str, set[str]],
                       parents: dict[str, set[str]],
                       ic: dict[str, float],
                       gene_anc: dict[str, set[str]]) -> dict[str, float]:
    """Best-match-average Resnik per gene, as a rank percentile (0..1).

    Raw BMA scores compress into a narrow band (generic query terms dilute
    everything), so absolute values mislead: an annotated-but-distant gene
    can score below the 0.5 neutral given to unannotated genes. Percentile
    rank fixes the direction: top genes -> ~1, median gene -> 0.5,
    bottom -> 0. BUB1B lands at ~0.997 on the MVA query (top 0.3%).
    """
    memo: dict[str, frozenset] = {}
    q_anc = [set(_ancestors(q, parents, memo)) for q in query_terms if q in parents]
    if not q_anc:
        return {}
    max_ic = max(ic.values()) if ic else 1.0
    raw: dict[str, float] = {}
    for g, gset in gene_anc.items():
        best = []
        for qa in q_anc:
            common = qa & gset
            best.append(max((ic.get(t, 0.0) for t in common), default=0.0))
        raw[g] = (sum(best) / len(best) / max_ic) if best else 0.0
    n = len(raw)
    if n == 0:
        return {}
    ordered = sorted(raw.values())
    import bisect
    return {g: bisect.bisect_right(ordered, s) / n for g, s in raw.items()}


def load_phenotype_model(obo_path: str, annot_path: str,
                         query_terms: list[str]) -> dict[str, float] | None:
    """Load ontology + annotations, return {gene: 0..1 resnik score}.

    Returns None (caller falls back to neutral 0.5) if files are missing.
    Warns loudly — phenotype is load-bearing for ranking.
    """
    if not Path(obo_path).exists() or not Path(annot_path).exists():
        print(f"[hpo] WARNING: ontology/annotations missing ({obo_path}, {annot_path}) — phenotype neutral 0.5")
        return None
    parents = load_ontology(obo_path)
    gene_terms = load_gene_annotations(annot_path)
    ic, gene_anc = build_ic(gene_terms, parents)
    scores = resnik_gene_scores(query_terms, gene_terms, parents, ic, gene_anc)
    print(f"[hpo] Resnik model: {len(parents)} terms, {len(gene_terms)} genes, IC max={max(ic.values()):.2f}")
    return scores
