process PRIORITIZE {
    tag "${vcf.baseName}"
    label 'prioritization'
    cpus 4
    memory '8 GB'
    publishDir "${params.outdir}", mode: 'copy', pattern: "ranked_variants.csv"
    publishDir "${params.outdir}/logs", mode: 'copy', pattern: "prioritize.log"

    input:
    path vcf
    path hpo
    path config_yaml

    output:
    path "ranked_variants.csv", emit: ranked
    path "prioritize.log", emit: log

    script:
    def maxVar = params.max_variants ? "--max-variants ${params.max_variants}" : ""
    """
    set -e
    echo "[prioritize] VCF=${vcf} HPO=${hpo} config=${config_yaml} outdir=${params.outdir} launchDir=${launchDir}" | tee prioritize.log
    # Use launchDir for src/
    PYTHONPATH=${launchDir} uv run --project ${launchDir} python -m src.prioritization.prioritize \
        --vcf ${vcf} \
        --hpo ${hpo} \
        --config ${config_yaml} \
        --output ranked_variants.csv \
        ${maxVar} 2>&1 | tee -a prioritize.log

    ls -lh ranked_variants.csv >> prioritize.log
    head -20 ranked_variants.csv >> prioritize.log
    wc -l ranked_variants.csv >> prioritize.log
    """
}
