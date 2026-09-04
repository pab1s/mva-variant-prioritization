"""Population frequency annotation (gnomAD, ExAC).

Baseline: if VCF already has gnomAD AF from VEP (--af_gnomad), use it.
Otherwise placeholder that returns input path unchanged but logs.

Future: annotate via gnomAD VCFs, tabix, or API:
  - bcftools annotate -a gnomad.af.vcf.gz -c INFO/gnomAD_AF
  - or VEP custom annotation with gnomAD VCF
"""
from pathlib import Path
import shutil
import subprocess


def add_gnomad_af(vcf_path: str, gnomad_vcf: str | None = None) -> str:
    """Annotate VCF with gnomAD AF if resource available, else pass-through."""
    p = Path(vcf_path)
    if gnomad_vcf and Path(gnomad_vcf).exists() and shutil.which("bcftools"):
        out = p.with_suffix("").with_suffix(".gnomad.vcf.gz")
        cmd = ["bcftools", "annotate", "-a", gnomad_vcf, "-c", "INFO/gnomAD_AF", "-Oz", "-o", str(out), str(p)]
        try:
            print(f"[gnomAD] Annotating: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            subprocess.run(["tabix", "-p", "vcf", str(out)], check=True)
            return str(out)
        except Exception as e:
            print(f"[gnomAD] Annotation failed ({e}), returning original")
            return str(p)
    # baseline: no gnomAD resource, assume rare (AF handling in prioritize.py will default)
    return str(p)


def filter_by_af(df, threshold: float = 0.01, af_col: str = "gnomad_af"):
    """Helper used in notebooks — keep variants with AF < threshold or missing."""
    import pandas as pd
    if af_col not in df.columns:
        return df  # no AF info, keep all
    return df[(df[af_col].isna()) | (df[af_col] < threshold)]
