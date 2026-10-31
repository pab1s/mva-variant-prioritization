"""Batch annotation lookups shared by dataset build and inference (parity).

Sources (all local, public):
  - AlphaMissense  data/annotations/AlphaMissense_hg38_modified.tsv.gz (tabix)
  - gnomAD v4.1 constraint (LOEUF / pLI) data/annotations/loeuf_dataset_grch38.tsv.gz
  - per-gene ClinVar P/LP counts        data/ml/gene_counts.csv.gz
"""
from __future__ import annotations

import gzip
import subprocess
from pathlib import Path

import pandas as pd

AM_TSV = Path("data/annotations/AlphaMissense_hg38_modified.tsv.gz")
LOEUF_TSV = Path("data/annotations/loeuf_dataset_grch38.tsv.gz")
GENE_COUNTS = Path("data/ml/gene_counts.csv.gz")

_LOEUF_CACHE: dict[str, tuple[float, float]] | None = None
_PATHCOUNT_CACHE: dict[str, int] | None = None


def load_loeuf(path: str | Path = LOEUF_TSV) -> dict[str, tuple[float, float]]:
    """gene symbol -> (LOEUF, pLI) from MANE/canonical transcript rows."""
    global _LOEUF_CACHE
    if _LOEUF_CACHE is not None:
        return _LOEUF_CACHE
    out: dict[str, tuple[float, float]] = {}
    with gzip.open(path, "rt") as fh:
        cols = fh.readline().rstrip("\n").lstrip("#").split("\t")
        i_gene = cols.index("gene")
        i_mane, i_can = cols.index("mane_select"), cols.index("canonical")
        i_loeuf, i_pli = cols.index("lof.oe_ci.upper"), cols.index("lof.pLI")
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) <= i_pli:
                continue
            if p[i_mane].lower() != "true" and p[i_can].lower() != "true":
                continue
            try:
                out[p[i_gene]] = (float(p[i_loeuf]), float(p[i_pli]))
            except ValueError:
                continue
    _LOEUF_CACHE = out
    return out


def load_gene_path_counts(path: str | Path = GENE_COUNTS) -> dict[str, int]:
    """gene symbol -> #ClinVar P/LP variants (log-scaled downstream)."""
    global _PATHCOUNT_CACHE
    if _PATHCOUNT_CACHE is not None:
        return _PATHCOUNT_CACHE
    if not Path(path).exists():
        _PATHCOUNT_CACHE = {}
        return {}
    df = pd.read_csv(path)
    _PATHCOUNT_CACHE = dict(zip(df["gene"], df["n_path"].astype(int)))
    return _PATHCOUNT_CACHE


def am_lookup(variants: list[tuple[str, int, str, str]]) -> dict[str, tuple[float, str]]:
    """Batch AlphaMissense lookup for (chrom, pos, ref, alt) via one tabix pass.

    Uses a temporary regions BED + `tabix -R` (single scan, no per-variant
    queries). Returns key 'chrom:pos:ref:alt' -> (score, class).
    """
    if not variants or not AM_TSV.exists():
        return {}
    regions = Path("/tmp/am_regions.bed")
    with open(regions, "w") as out:
        for chrom, pos, ref, alt in variants:
            out.write(f"chr{chrom}\t{pos - 1}\t{pos}\n")
    res = subprocess.run(["tabix", "-R", str(regions), str(AM_TSV)],
                         capture_output=True, text=True, check=True)
    am: dict[str, tuple[float, str]] = {}
    for line in res.stdout.splitlines():
        p = line.split("\t")
        if len(p) < 10:
            continue
        chrom = p[0].removeprefix("chr")
        am[f"{chrom}:{p[1]}:{p[2]}:{p[3]}"] = (float(p[8]), p[9])
    return am


def vkey(chrom, pos, ref, alt) -> str:
    return f"{chrom}:{pos}:{ref}:{alt}"


CADD_SNV = Path("data/cadd/whole_genome_SNVs.tsv.gz")
CADD_INDEL = Path("data/cadd/gnomad.genomes.r4.0.indel.tsv.gz")


def cadd_lookup(variants: list[tuple[str, int, str, str]]) -> dict[str, float]:
    """Batch CADD PHRED lookup (one `tabix -R` pass per CADD file).

    CADD is intentionally NOT run as a VEP plugin anymore: one tabix query per
    variant made VEP ~7x slower. Both files are bgzipped+tabixed TSVs with
    columns: #Chrom Pos Ref Alt RawScore PHRED.
    """
    if not variants:
        return {}
    regions = Path("/tmp/cadd_regions.bed")
    with open(regions, "w") as out:
        for chrom, pos, ref, alt in variants:
            out.write(f"{chrom}\t{pos - 1}\t{pos}\n")
    out: dict[str, float] = {}
    for f in (CADD_SNV, CADD_INDEL):
        if not f.exists() or not Path(str(f) + ".tbi").exists():
            continue
        res = subprocess.run(["tabix", "-R", str(regions), str(f)],
                             capture_output=True, text=True, check=True)
        for line in res.stdout.splitlines():
            p = line.split("\t")
            if len(p) < 6:
                continue
            try:
                out[vkey(p[0], int(p[1]), p[2], p[3])] = float(p[5])
            except ValueError:
                continue
    return out


def load_af(vcf_path: str | Path) -> dict[str, float]:
    """gnomAD AF per variant key from a side-file annotated with --af_gnomade/g.

    Reads the CSQ gnomADe_AF / gnomADg_AF columns (fallback AF) and keeps the
    max across transcripts. Returns {} when the file is absent (AF is then
    unknown -> treated as rare, consistently at train and serve time).
    """
    import gzip as _gzip
    p = Path(vcf_path)
    if not p.exists():
        return {}
    opener = _gzip.open if str(p).endswith(".gz") else open
    af: dict[str, float] = {}
    hdr: list[str] | None = None
    idx: dict[str, int] = {}
    with opener(p, "rt") as fh:
        for line in fh:
            if line.startswith("##INFO=<ID=CSQ"):
                hdr = line.split("Format:")[1].split('"')[0].split("|")
                idx = {k: i for i, k in enumerate(hdr)}
                continue
            if line.startswith("#"):
                continue
            if hdr is None:
                break
            f = line.split("\t")
            csq = None
            for kv in f[7].split(";"):
                if kv.startswith("CSQ="):
                    csq = kv[4:]
                    break
            if not csq:
                continue
            best: float | None = None
            for entry in csq.split(","):
                d = entry.split("|")
                if len(d) <= max(idx.get("gnomADe_AF", 0), idx.get("gnomADg_AF", 0), idx.get("AF", 0)):
                    continue
                for k in ("gnomADg_AF", "gnomADe_AF", "AF"):
                    i = idx.get(k, -1)
                    if i < 0:
                        continue
                    v = d[i].strip()
                    if v and v != "-":
                        try:
                            fv = float(v)
                        except ValueError:
                            continue
                        if best is None or fv > best:
                            best = fv
                        break
            if best is not None:
                af[vkey(f[0], int(f[1]), f[3], f[4])] = best
    return af
