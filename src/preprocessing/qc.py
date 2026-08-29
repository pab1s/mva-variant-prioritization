"""Quality control for FASTQ / BAM inputs."""
from pathlib import Path

def run_fastqc(fastq_path: str, out_dir: str) -> dict:
    """Run FastQC (placeholder)."""
    # TODO: integrate fastqc / fastp
    return {"input": fastq_path, "out_dir": out_dir, "status": "not_implemented"}

def summarize_qc(qc_dir: str) -> dict:
    """Aggregate QC metrics."""
    return {}
