"""Lightweight HPO ontology: parsing, information content, Resnik similarity.

Offline, stdlib-only (no new deps). Loaded once and queried many times:
one query per pseudo-case in training (~13k), one per patient at inference.

Design notes
------------
- Obsolete terms are dropped; `part_of` relationships are treated as `is_a`
  for ancestor propagation (standard practice in phenotype similarity).
- IC is computed from *gene* annotation frequency (genes annotated to a term
  or any of its ancestors), which is the HPOA-consistent choice; the
  caller injects the gene annotations via `compute_ic()`.
- Resnik(term_a, term_b) = IC of the most informative common ancestor (MICA).
  Set-level similarity = max over term pairs (max-Resnik), plus a
  frequency-weighted best-match average (BMA) à la Phenomizer/Exomiser.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

# HPO "frequency of the phenotypic feature in the disease" annotation terms
# (verified against data/hpo/hp.obo names, 2026-10).
FREQ_TERMS: dict[str, float] = {
    "HP:0040280": 1.0,  # Obligate
    "HP:0040281": 0.9,  # Very frequent
    "HP:0040282": 0.6,  # Frequent
    "HP:0040283": 0.3,  # Occasional
    "HP:0040284": 0.1,  # Very rare
}
EXCLUDED_TERM = "HP:0040285"  # "Excluded" (feature explicitly absent)

ROOT = "HP:0000001"  # "All"


class HPOOntology:
    """Parsed hp.obo DAG with memoized ancestor closures and gene-based IC."""

    def __init__(self, obo_path: str | Path):
        self.parents: dict[str, tuple[str, ...]] = {}
        self.names: dict[str, str] = {}
        self._ancestors: dict[str, frozenset[str]] = {}
        self.ic: dict[str, float] = {}
        self._ic_ready = False
        self._parse(obo_path)

    # ------------------------------------------------------------------ parse
    def _parse(self, path: str | Path) -> None:
        term_id: str | None = None
        name = ""
        parents: list[str] = []
        obsolete = False
        pending = False

        def flush() -> None:
            nonlocal term_id, name, parents, obsolete, pending
            if pending and term_id and not obsolete:
                self.parents[term_id] = tuple(parents)
                self.names[term_id] = name
            term_id, name, parents, obsolete, pending = None, "", [], False, False

        with open(path) as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith("[Term]"):
                    flush()
                    pending = True
                elif line.startswith("id: "):
                    term_id = line[4:].strip()
                elif line.startswith("name: "):
                    name = line[6:].strip()
                elif line.startswith("is_a: "):
                    parents.append(line[6:].split("!")[0].strip())
                elif line.startswith("relationship: part_of "):
                    parents.append(line.split()[2])
                elif line.startswith("is_obsolete: true"):
                    obsolete = True
                elif line == "[Typedef]" or line.startswith("["):
                    flush()
        flush()
        if not self.parents:
            raise RuntimeError(f"No HPO terms parsed from {path!r}")

    # ------------------------------------------------------------- ancestors
    def ancestors(self, term: str) -> frozenset[str]:
        """Ancestor closure including the term itself (memoized)."""
        cached = self._ancestors.get(term)
        if cached is not None:
            return cached
        seen: set[str] = set()
        stack = [term]
        while stack:
            t = stack.pop()
            if t in seen:
                continue
            seen.add(t)
            stack.extend(self.parents.get(t, ()))
        out = frozenset(seen)
        self._ancestors[term] = out
        return out

    def is_known(self, term: str) -> bool:
        return term in self.parents

    # -------------------------------------------------------------------- IC
    def compute_ic(self, gene_terms: dict[str, set[str]]) -> dict[str, float]:
        """IC(t) = -log(#genes annotated to t or descendant / #genes).

        `gene_terms`: gene symbol -> raw (non-propagated) HPO term set.
        Root gets IC 0 by construction. Terms seen in no gene keep IC 0.
        """
        if not gene_terms:
            raise RuntimeError("compute_ic needs a non-empty gene_terms map")
        total = len(gene_terms)
        counts: Counter[str] = Counter()
        for terms in gene_terms.values():
            closure: set[str] = set()
            for t in terms:
                closure |= self.ancestors(t)
            counts.update(closure)
        self.ic = {t: -math.log(c / total) for t, c in counts.items() if c}
        self._ic_ready = True
        return self.ic

    def _require_ic(self) -> None:
        if not self._ic_ready:
            raise RuntimeError("IC not computed — call compute_ic(gene_terms) first")

    # ------------------------------------------------------------ similarity
    def term_resnik(self, a: str, b: str) -> float:
        self._require_ic()
        common = self.ancestors(a) & self.ancestors(b)
        if not common:
            return 0.0
        return max(self.ic.get(t, 0.0) for t in common)

    def _best_ic(self, q_term: str, target_closure: frozenset[str]) -> float:
        """Max IC over ancestors of q_term that appear in target_closure."""
        best = 0.0
        for anc in self.ancestors(q_term):
            if anc in target_closure:
                v = self.ic.get(anc, 0.0)
                if v > best:
                    best = v
        return best

    def similarity(
        self,
        query_terms: set[str],
        target_terms: set[str],
        query_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Set-level phenotype similarity (unnormalized — normalize per query).

        Returns:
            resnik_max: max-Resnik over all term pairs
            resnik_bma: frequency-weighted best-match average over query terms
                        (each query term matched to its best target term)
            terms_matched: #query terms with a non-zero best match
        """
        self._require_ic()
        if not query_terms or not target_terms:
            return {"resnik_max": 0.0, "resnik_bma": 0.0, "terms_matched": 0}
        target_closure: set[str] = set()
        for t in target_terms:
            target_closure |= self.ancestors(t)
        target_closure_f = frozenset(target_closure)

        resnik_max = 0.0
        matched = 0
        wsum = 0.0
        wacc = 0.0
        for q in query_terms:
            best = self._best_ic(q, target_closure_f)
            if best <= 0.0:
                continue
            matched += 1
            if best > resnik_max:
                resnik_max = best
            w = (query_weights or {}).get(q, 0.5)
            wacc += w * best
            wsum += w
        resnik_bma = (wacc / wsum) if wsum > 0 else 0.0
        return {
            "resnik_max": resnik_max,
            "resnik_bma": resnik_bma,
            "terms_matched": matched,
        }

    def query_norm_factor(self, query_terms: set[str]) -> float:
        """Upper bound of achievable Resnik for this query (normalize by this).

        = max over query terms of the IC of their most informative ancestor
        that is annotated to at least one gene. Cheap (no per-gene scan) and
        monotone in the same units as `similarity()`, so features land in [0,1].
        """
        self._require_ic()
        best = 0.0
        for q in query_terms:
            for anc in self.ancestors(q):
                v = self.ic.get(anc, 0.0)
                if v > best:
                    best = v
        return best if best > 0 else 1.0
