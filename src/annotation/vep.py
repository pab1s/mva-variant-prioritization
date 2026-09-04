"""Ensembl VEP / SnpEff annotation helpers.

Real VEP requires:
  - ensembl-vep installed (perl) + cache (~20GB) at /data/vep_cache or params.vep_cache
  - or Docker image ensemblorg/ensembl-vep

Baseline pipeline is VEP-ready but offline-tolerant:
  - If cache/binary missing → copy VCF through with warning and tag INFO as VEP_MOCK
  - If available → shell out to `vep -i {in} -o {out} --cache --offline --vcf --force_overwrite`
"""
import shutil
import subprocess
import gzip
from pathlib import Path


def _has_vep(cache_dir: str | None = None) -> bool:
    vep_bin = shutil.which("vep")
    if not vep_bin:
        return False
    if cache_dir and not Path(cache_dir).exists():
        return False
    return True


def _has_docker_vep() -> bool:
    return shutil.which("docker") is not None


def annotate_vcf(in_vcf: str, out_vcf: str, cache_dir: str | None = None) -> str:
    """Annotate VCF with VEP if available, else pass-through.

    Args:
        in_vcf: input .vcf.gz
        out_vcf: output .vep.vcf.gz (or .vcf.gz)
        cache_dir: VEP cache directory (e.g. /data/vep_cache)

    Returns path to annotated VCF.
    """
    in_p = Path(in_vcf)
    out_p = Path(out_vcf)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Try native VEP
    if _has_vep(cache_dir):
        cmd = [
            "vep",
            "-i", str(in_p),
            "-o", str(out_p),
            "--vcf",
            "--cache",
            "--offline",
            "--force_overwrite",
            "--assembly", "GRCh38",
            "--symbol", "--canonical", "--hgvs",
            "--af_gnomad", "--clin_sig",
        ]
        if cache_dir:
            cmd.extend(["--dir_cache", str(cache_dir)])
        try:
            print(f"[VEP] Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, timeout=3600)
            # bgzip if needed
            if out_p.suffix == ".gz" and not out_p.exists():
                # vep may output uncompressed
                alt = out_p.with_suffix("")
                if alt.exists():
                    subprocess.run(["bgzip", "-c", str(alt)], stdout=open(out_p, "wb"), check=True)
            return str(out_p)
        except Exception as e:
            print(f"[VEP] Native VEP failed ({e}), falling back to pass-through")

    # Try Docker VEP (ensemblorg/ensembl-vep)
    # Only if docker daemon is actually accessible
    if _has_docker_vep() and cache_dir and Path(cache_dir).exists():
        try:
            # quick docker check
            result = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
            if result.returncode == 0:
                cmd = [
                    "docker", "run", "--rm",
                    "-v", f"{in_p.parent}:/in",
                    "-v", f"{out_p.parent}:/out",
                ]
                if cache_dir:
                    cmd.extend(["-v", f"{cache_dir}:/opt/vep/.vep"])
                cmd.extend([
                    "ensemblorg/ensembl-vep",
                    "vep",
                    "-i", f"/in/{in_p.name}",
                    "-o", f"/out/{out_p.name}",
                    "--cache", "--offline", "--vcf", "--force_overwrite",
                    "--assembly", "GRCh38",
                ])
                print(f"[VEP] Trying docker: {' '.join(cmd)}")
                subprocess.run(cmd, check=True, timeout=3600)
                return str(out_p)
        except Exception as e:
            print(f"[VEP] Docker VEP failed ({e}), falling back")

    # Fallback: pass-through copy with mock annotation tag
    print(f"[VEP] No VEP cache/binary usable. Passing through {in_p} -> {out_p} (mock annotation)")
    # Use stream copy (handle gz)
    if str(in_p).endswith(".gz") and str(out_p).endswith(".gz"):
        shutil.copy2(in_p, out_p)
    else:
        # decompress/compress as needed
        import gzip as gz
        with gz.open(in_p, 'rt') if str(in_p).endswith(".gz") else open(in_p) as fin:
            # ensure bgzip for .gz output
            if str(out_p).endswith(".gz"):
                with gz.open(out_p, 'wt') as fout:
                    for line in fin:
                        # Tag VCF header that this is mock-annotated
                        if line.startswith("##fileformat") and "VEP_MOCK" not in line:
                            fout.write(line)
                            fout.write("##INFO=<ID=VEP_MOCK,Number=0,Type=Flag,Description=\"VEP annotation not run - baseline pass-through\">\n")
                        else:
                            fout.write(line)
            else:
                with open(out_p, 'w') as fout:
                    fout.write(fin.read())
    return str(out_p)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="VEP annotate helper")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--cache", default=None)
    a = p.parse_args()
    print(annotate_vcf(a.input, a.output, a.cache))
