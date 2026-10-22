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
    # Real VEP via python helper (conda perl + cache + fasta for HGVS).
    # Fail fast: no pass-through fallback. An unannotated VCF would silently
    # produce a wrong ranking, so a VEP failure must fail the pipeline.
    PYTHONPATH=${launchDir} uv run --project ${launchDir} python -m src.annotation.vep --input ${vcf} --output ${vcf.baseName}.vep.vcf.gz --cache ${params.vep_cache} --fasta ${params.fasta} --fork ${task.cpus} | tee annotate.log
    # Index for downstream (tabix if available)
    if command -v tabix >/dev/null 2>&1; then
        tabix -p vcf ${vcf.baseName}.vep.vcf.gz
    fi
    ls -lh ${vcf.baseName}.vep.vcf.gz
    """
}
