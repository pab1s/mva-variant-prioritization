"""Adapter trimming and filtering."""
from pathlib import Path
import shutil
import subprocess


def trim_adapters(fastq_r1: str, fastq_r2: str, out_dir: str) -> tuple[str, str]:
    """Trim adapters using fastp / cutadapt if available, else pass-through.

    Baseline: QC-only mode often skips trimming for WGS baseline (variant calling
    from provided VCF doesn't need it). This function is wired into Nextflow
    so future realignment can enable it.

    Returns tuple of (trimmed_R1, trimmed_R2) paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    r1 = Path(fastq_r1)
    r2 = Path(fastq_r2)

    out_r1 = out / f"{r1.name.replace('.fastq.gz','')}.trimmed.fastq.gz"
    out_r2 = out / f"{r2.name.replace('.fastq.gz','')}.trimmed.fastq.gz"

    # Prefer fastp (handles both reads together, auto adapter detection)
    if shutil.which("fastp"):
        try:
            cmd = ["fastp", "-i", str(r1), "-I", str(r2), "-o", str(out_r1), "-O", str(out_r2), "--detect_adapter_for_pe", "-w", "4"]
            subprocess.run(cmd, check=True, capture_output=True, timeout=1800)
            return (str(out_r1), str(out_r2))
        except Exception as e:
            print(f"[trim] fastp failed ({e}), falling back to copy")

    # Try cutadapt
    if shutil.which("cutadapt"):
        try:
            # minimal: trim Illumina adapters, q20
            cmd1 = ["cutadapt", "-a", "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA", "-A", "AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT", "-q", "20", "-m", "20", "-o", str(out_r1), "-p", str(out_r2), str(r1), str(r2)]
            subprocess.run(cmd1, check=True, capture_output=True, timeout=1800)
            return (str(out_r1), str(out_r2))
        except Exception as e:
            print(f"[trim] cutadapt failed ({e}), falling back to copy")

    # Fallback: symlink/copy (no trimming) — pipeline continues
    # Use symlink to avoid copying 80GB
    try:
        if not out_r1.exists():
            out_r1.symlink_to(r1.resolve())
        if not out_r2.exists():
            out_r2.symlink_to(r2.resolve())
    except:
        import shutil as sh
        sh.copy2(r1, out_r1)
        sh.copy2(r2, out_r2)
    return (str(out_r1), str(out_r2))
