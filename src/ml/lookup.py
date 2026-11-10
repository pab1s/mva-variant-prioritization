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

_LOEUF_CACHE: dict[str, tuple[float | None, float | None]] | None = None
_GENE_CONSTRAINT_CACHE: dict[str, dict[str, float | None]] | None = None
_PATHCOUNT_CACHE: dict[str, int] | None = None


def _num(parts: list[str], idx: int) -> float | None:
    """Parse a gnomAD constraint field; missing/invalid -> None."""
    if idx < 0 or idx >= len(parts):
        return None
    value = parts[idx].strip()
    if value in ("", "NA", "NaN", "nan", "None", "[]"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_gene_constraint(
    path: str | Path = LOEUF_TSV,
) -> dict[str, dict[str, float | None]]:
    """Gene symbol -> gnomAD constraint/structure metrics.

    One row per gene, using the MANE Select transcript when available
    (falling back to the canonical transcript). The values are the raw
    public metrics; `derive_frame` applies log/clip/neutral imputation.

    Keys: loeuf, pli, oe_lof, oe_mis, mis_z, syn_z, cds_length,
    num_exons, segdup_prop, lcr_prop.
    """
    global _GENE_CONSTRAINT_CACHE
    if _GENE_CONSTRAINT_CACHE is not None:
        return _GENE_CONSTRAINT_CACHE

    out: dict[str, dict[str, float | None]] = {}
    with gzip.open(path, "rt") as fh:
        cols = fh.readline().rstrip("\n").lstrip("#").split("\t")
        idx = {name: i for i, name in enumerate(cols)}

        def gi(name: str) -> int:
            return idx.get(name, -1)

        i_gene = gi("gene")
        i_mane = gi("mane_select")
        i_can = gi("canonical")
        i_loeuf = gi("lof.oe_ci.upper")
        i_pli = gi("lof.pLI")
        i_oe_lof = gi("lof.oe")
        i_oe_mis = gi("mis.oe")
        i_mis_z = gi("mis.z_score")
        i_syn_z = gi("syn.z_score")
        i_cds = gi("cds_length")
        i_exons = gi("num_coding_exons")
        i_segdup = gi("gene_quality_metrics.exome_prop_segdup")
        i_lcr = gi("gene_quality_metrics.exome_prop_LCR")

        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) <= max(i_gene, i_mane, i_can):
                continue
            gene = p[i_gene].strip()
            if not gene:
                continue
            is_mane = p[i_mane].strip().lower() == "true"
            is_canonical = p[i_can].strip().lower() == "true"
            if not is_mane and not is_canonical:
                continue
            # Prefer MANE Select; a later MANE row may replace a canonical one.
            if gene in out and not is_mane:
                continue
            out[gene] = {
                "loeuf": _num(p, i_loeuf),
                "pli": _num(p, i_pli),
                "oe_lof": _num(p, i_oe_lof),
                "oe_mis": _num(p, i_oe_mis),
                "mis_z": _num(p, i_mis_z),
                "syn_z": _num(p, i_syn_z),
                "cds_length": _num(p, i_cds),
                "num_exons": _num(p, i_exons),
                "segdup_prop": _num(p, i_segdup),
                "lcr_prop": _num(p, i_lcr),
            }
    _GENE_CONSTRAINT_CACHE = out
    return out


def load_loeuf(
    path: str | Path = LOEUF_TSV,
) -> dict[str, tuple[float | None, float | None]]:
    """gene symbol -> (LOEUF, pLI) from the shared constraint table."""
    global _LOEUF_CACHE
    if _LOEUF_CACHE is None:
        gc = load_gene_constraint(path)
        _LOEUF_CACHE = {g: (v.get("loeuf"), v.get("pli"))
                        for g, v in gc.items()}
    return _LOEUF_CACHE


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


def load_af_tsv(path: str | Path) -> dict[str, float]:
    """Variant key -> max population AF from a CHROM/POS/REF/ALT/AF TSV."""
    import csv as _csv
    import gzip as _gzip

    p = Path(path)
    if not p.exists():
        return {}
    opener = _gzip.open if str(p).endswith(".gz") else open
    out: dict[str, float] = {}
    with opener(p, "rt", newline="") as fh:
        sample = fh.readline()
        fh.seek(0)
        delim = "\t" if "\t" in sample else ","
        reader = _csv.DictReader(fh, delimiter=delim)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}
        need = ("chrom", "pos", "ref", "alt", "af")
        if not all(c in cols for c in need):
            raise RuntimeError(f"AF TSV {p} missing columns {need}")
        for row in reader:
            value = row[cols["af"]].strip()
            if not value:
                continue
            try:
                af = float(value)
            except ValueError:
                continue
            chrom = str(row[cols["chrom"]]).removeprefix("chr")
            out[vkey(chrom, int(float(row[cols["pos"]])),
                     row[cols["ref"]], row[cols["alt"]])] = af
    return out


def load_af(vcf_path: str | Path) -> dict[str, float]:
    """gnomAD AF per variant key from a VEP side-file or an AF TSV.

    - `.tsv/.tsv.gz/.csv/.csv.gz` -> direct AF table (fast cache lookup).
    - VCF -> read CSQ gnomADe_AF / gnomADg_AF / AF columns (VEP side-file).
    Returns {} when the file is absent (AF unknown -> treated as rare,
    consistently at train and serve time).
    """
    import gzip as _gzip
    p = Path(vcf_path)
    if not p.exists():
        return {}
    if p.suffix.lower() in (".tsv", ".csv") or \
            p.name.lower().endswith((".tsv.gz", ".csv.gz")):
        return load_af_tsv(p)
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
