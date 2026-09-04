"""Quality control for FASTQ / BAM inputs."""
from pathlib import Path
import subprocess
import shutil
import json


def run_fastqc(fastq_path: str, out_dir: str) -> dict:
    """Run FastQC if available, else return stub.

    Baseline supports both FASTQ QC paths:
    - fastqc (per-file html/zip)
    - fastp (fast all-in-one QC + trimming) if fastqc missing

    Never fails pipeline: returns dict with status.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inp = Path(fastq_path)

    # Try fastqc
    if shutil.which("fastqc"):
        try:
            cmd = ["fastqc", str(inp), "-o", str(out)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            html = out / f"{inp.name.replace('.fastq.gz','').replace('.fq.gz','')}_fastqc.html"
            return {"input": str(inp), "out_dir": str(out), "status": "ok", "tool": "fastqc", "html": str(html), "log": result.stdout[:500]}
        except Exception as e:
            return {"input": str(inp), "out_dir": str(out), "status": "failed", "error": str(e), "tool": "fastqc"}

    # Try fastp
    if shutil.which("fastp"):
        try:
            out_json = out / f"{inp.stem}.fastp.json"
            out_html = out / f"{inp.stem}.fastp.html"
            cmd = ["fastp", "-i", str(inp), "-o", "/dev/null", "-j", str(out_json), "-h", str(out_html), "--dont_eval_duplication"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            return {"input": str(inp), "out_dir": str(out), "status": "ok", "tool": "fastp", "json": str(out_json)}
        except Exception as e:
            return {"input": str(inp), "out_dir": str(out), "status": "failed", "error": str(e), "tool": "fastp"}

    # Fallback stub: compute basic stats via python (no external tool)
    try:
        import gzip
        count = 0
        total_bases = 0
        gc = 0
        # sample first 10k reads for speed if large
        max_reads = 10000
        with gzip.open(inp, 'rt') as f:
            while count < max_reads:
                header = f.readline()
                if not header:
                    break
                seq = f.readline().strip()
                plus = f.readline()
                qual = f.readline()
                if not qual:
                    break
                count += 1
                total_bases += len(seq)
                gc += seq.count('G') + seq.count('C')
        avg_len = total_bases / count if count else 0
        gc_pct = (gc / total_bases * 100) if total_bases else 0
        summary = out / f"{inp.name}.qc.stub.json"
        with open(summary, 'w') as jf:
            json.dump({"input": str(inp), "reads_sampled": count, "avg_len": avg_len, "gc_pct": gc_pct, "tool": "stub"}, jf, indent=2)
        return {"input": str(inp), "out_dir": str(out), "status": "stub", "reads_sampled": count, "avg_len": avg_len, "gc_pct": gc_pct}
    except Exception as e:
        return {"input": str(inp), "out_dir": str(out), "status": "not_implemented", "error": str(e)}


def summarize_qc(qc_dir: str) -> dict:
    """Aggregate QC metrics from qc_dir (stub jsons or fastqc)."""
    p = Path(qc_dir)
    if not p.exists():
        return {}
    summary = {}
    for f in p.glob("*.json"):
        try:
            import json as js
            with open(f) as jf:
                summary[f.name] = js.load(jf)
        except:
            continue
    return summary
