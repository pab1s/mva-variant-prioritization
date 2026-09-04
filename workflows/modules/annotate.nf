process ANNOTATE {
    tag "${vcf.baseName}"
    label 'annotation'
    cpus 2
    memory '4 GB'

    input:
    path vcf
    path vcf_index

    output:
    path "${vcf.baseName}.vep.vcf.gz", emit: annotated_vcf

    script:
    """
    set -o pipefail
    # Real VEP if cache+binary available, else mock pass-through via python helper
    if PYTHONPATH=${launchDir} uv run --project ${launchDir} python -m src.annotation.vep --input ${vcf} --output ${vcf.baseName}.vep.vcf.gz --cache ${params.vep_cache} > annotate.log 2>&1; then
        echo "[annotate] VEP annotation done" | tee -a annotate.log
    else
        echo "[annotate] python VEP helper failed (see annotate.log), copying through" | tee -a annotate.log
        cat annotate.log
        cp ${vcf} ${vcf.baseName}.vep.vcf.gz
    fi
    # Index for downstream (tabix if available)
    if command -v tabix >/dev/null 2>&1; then
        tabix -p vcf ${vcf.baseName}.vep.vcf.gz || echo "tabix indexing failed (optional)"
    fi
    ls -lh ${vcf.baseName}.vep.vcf.gz
    cat annotate.log || true
    """
}
