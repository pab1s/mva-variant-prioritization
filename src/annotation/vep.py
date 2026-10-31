"""Ensembl VEP annotation helper — fail fast, no silent fallbacks.

Real VEP requires:
  - ensembl-vep (conda perl + vep script, see CONDA_PERL/CONDA_VEP)
  - cache (~20GB) at cache_dir (e.g. data/vep_cache)
  - reference FASTA for HGVS notation

Design: annotation is load-bearing for ranking. A silently unannotated VCF
produces a plausible-looking but wrong ranked list (all consequences
inferred, all variants "rare"). So every failure mode raises instead of
passing data through.
"""
import shlex
import shutil
import subprocess
from pathlib import Path

CONDA_PERL = Path.home() / "micromamba/envs/mva/bin/perl"
CONDA_VEP = Path.home() / "micromamba/envs/mva/bin/vep"
# VEP CADD plugin shipped inside the conda ensembl-vep dist (top-level CADD.pm)
CONDA_PLUGINS = Path.home() / "micromamba/envs/mva/share/ensembl-vep-113.4-0"


def _vep_cmd() -> list[str]:
    """Working VEP invocation: conda perl + conda vep script.

    The `vep` on PATH uses /usr/bin/perl which lacks DBI; the conda
    perl has it. Raises if VEP is not installed.
    """
    if CONDA_PERL.exists() and CONDA_VEP.exists():
        return [str(CONDA_PERL), str(CONDA_VEP)]
    vep_bin = shutil.which("vep")
    if vep_bin:
        return [vep_bin]
    raise RuntimeError(
        "VEP binary not found (checked conda env + PATH). "
        "Install ensembl-vep 113 (e.g. micromamba install -n mva ensembl-vep)."
    )


def _require_cache(cache_dir: str | None) -> str:
    if not cache_dir or not Path(cache_dir).exists():
        raise RuntimeError(
            f"VEP cache not found: {cache_dir!r}. Download "
            "homo_sapiens_vep_113_GRCh38.tar.gz from "
            "https://ftp.ensembl.org/pub/release-113/variation/vep/ "
            "and extract it into data/vep_cache/."
        )
    return cache_dir


def annotate_vcf(in_vcf: str, out_vcf: str, cache_dir: str | None = None,
                 fasta: str | None = None, fork: int = 2,
                 timeout: int = 7200,
                 cadd_snv: str | None = None,
                 cadd_indel: str | None = None,
                 clinvar: str | None = None,
                 buffer_size: int | None = None,
                 everything: bool = True,
                 extra_args: str = "") -> str:
    """Annotate VCF with VEP. Raises on any failure - never passes through.

    Args:
        in_vcf: input VCF (.vcf or .vcf.gz)
        out_vcf: output annotated VCF path
        cache_dir: VEP cache directory (required)
        fasta: reference FASTA for HGVS notation (required)
        fork: VEP parallel workers
        timeout: subprocess timeout in seconds

    Returns path to annotated VCF.

    Raises:
        RuntimeError: VEP missing, cache missing, fasta missing, or VEP
            exits non-zero / produces empty output.
    """
    in_p = Path(in_vcf)
    out_p = Path(out_vcf)
    if not in_p.exists():
        raise RuntimeError(f"Input VCF not found: {in_vcf!r}")
    out_p.parent.mkdir(parents=True, exist_ok=True)

    cmd = _vep_cmd()
    cache = _require_cache(cache_dir)
    if not fasta or not Path(fasta).exists():
        raise RuntimeError(
            f"Reference FASTA not found: {fasta!r}. HGVS notation requires it."
        )

    plugin_args: list[str] = []
    if cadd_snv or cadd_indel:
        for label, path in (("CADD SNV", cadd_snv), ("CADD indel", cadd_indel)):
            if not path or not Path(path).exists():
                raise RuntimeError(
                    f"{label} file not found: {path!r}. Expected e.g. "
                    "data/cadd/whole_genome_SNVs.tsv.gz (+ .tbi)."
                )
            tbi = str(path) + ".tbi"
            if not Path(tbi).exists():
                raise RuntimeError(f"Tabix index missing for CADD file: {tbi!r}.")
        plugin_args = [
            "--dir_plugins", str(CONDA_PLUGINS),
            "--plugin", f"CADD,{cadd_snv},{cadd_indel}",
        ]

    # ClinVar as --custom (verified: cache's own CLIN_SIG column is empty in
    # this build; VEP appends the custom fields to the CSQ tail, and
    # _parse_info reads ClinVar_CLNSIG from there).
    custom_args: list[str] = []
    if clinvar:
        if not Path(clinvar).exists():
            raise RuntimeError(f"ClinVar VCF not found: {clinvar!r}.")
        if not Path(str(clinvar) + ".tbi").exists():
            raise RuntimeError(f"ClinVar tabix index missing: {clinvar}.tbi")
        custom_args = ["--custom", f"{clinvar},ClinVar,vcf,overlap,0,CLNSIG"]

    # --everything: enables all annotation sources, including the variation
    # overlap that carries Existing_variation (ClinVar ALLELEID) and CLIN_SIG.
    # Measured on this box: --everything / --af_* cost ~7.6x the structural
    # pass (13k vs 99k rec/min), so callers needing only SYMBOL/EXON/CANONICAL/
    # MANE/CLIN_SIG pass everything=False plus an explicit --extra flag list.
    # (Note: --everything implies af_gnomade; adding --af_gnomad explicitly
    #  errors with "Can't use --af_gnomade and --af_gnomad together".)
    vep_cmd = cmd + [
        "-i", str(in_p),
        "-o", str(out_p),
        "--vcf",
        "--cache",
        "--offline",
        "--force_overwrite",
        "--assembly", "GRCh38",
        "--fork", str(fork),
        "--fasta", str(fasta),
        "--dir_cache", str(cache),
    ]
    if everything:
        vep_cmd += ["--everything", "--mane"]
    # Smaller in-memory batch on RAM-constrained boxes: systemd-oomd kills the
    # whole scope when RAM+swap pass 90%, so keep VEP's working set small.
    if buffer_size:
        vep_cmd += ["--buffer_size", str(buffer_size)]
    if extra_args:
        vep_cmd += shlex.split(extra_args)
    vep_cmd += plugin_args + custom_args
    print(f"[VEP] Running: {' '.join(vep_cmd)}")
    try:
        subprocess.run(vep_cmd, check=True, timeout=timeout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"VEP exited with code {e.returncode}. Command: {' '.join(vep_cmd)}"
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"VEP timed out after {timeout}s on {in_vcf!r}."
        ) from e

    if not out_p.exists() or out_p.stat().st_size == 0:
        raise RuntimeError(
            f"VEP produced no output at {out_p!r} (input: {in_vcf!r})."
        )
    n = sum(1 for line in open(out_p) if not line.startswith("#"))
    if n == 0 and sum(1 for line in open(in_p) if not line.startswith("#")) > 0:
        raise RuntimeError(
            f"VEP dropped all variants (0 data lines in {out_p!r}). "
            "Likely cause: input contigs missing from cache "
            "(check data/vep_cache/homo_sapiens/113_GRCh38/)."
        )
    return str(out_p)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="VEP annotate helper (fail fast)")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--cache", default=None)
    p.add_argument("--fasta", default=None)
    p.add_argument("--fork", type=int, default=2)
    p.add_argument("--cadd_snv", default=None,
                   help="CADD whole_genome_SNVs.tsv.gz (enables --plugin CADD)")
    p.add_argument("--cadd_indel", default=None,
                   help="CADD gnomad.*.indel.tsv.gz (enables --plugin CADD)")
    p.add_argument("--clinvar", default=None,
                   help="ClinVar VCF.gz + .tbi (enables --custom ClinVar CLNSIG)")
    p.add_argument("--buffer_size", type=int, default=None,
                   help="VEP --buffer_size (lower = less RAM; helps systemd-oomd boxes)")
    p.add_argument("--no-everything", action="store_true",
                   help="skip --everything/--mane (fast structural pass)")
    p.add_argument("--extra", default="",
                   help="raw VEP flags appended verbatim (quote as one string)")
    p.add_argument("--timeout", type=int, default=7200,
                   help="VEP subprocess timeout in seconds")
    a = p.parse_args()
    print(annotate_vcf(a.input, a.output, a.cache, a.fasta, a.fork,
                       timeout=a.timeout,
                       cadd_snv=a.cadd_snv, cadd_indel=a.cadd_indel,
                       clinvar=a.clinvar, buffer_size=a.buffer_size,
                       everything=not a.no_everything, extra_args=a.extra))
