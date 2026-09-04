process QC {
    tag "${fastq.baseName}"
    label 'qc'
    cpus 1
    memory '2 GB'
    publishDir "${params.outdir}/qc", mode: 'copy'

    input:
    path fastq

    output:
    path "qc_${fastq.baseName}/*", optional: true, emit: qc_results
    path "qc_${fastq.baseName}.json", emit: qc_json

    script:
    """
    mkdir -p qc_${fastq.baseName}
    # Use python stub if fastqc/fastp not available; never fails pipeline
    PYTHONPATH=${launchDir} uv run --project ${launchDir} python -c "
from src.preprocessing.qc import run_fastqc
import json
res = run_fastqc('${fastq}', 'qc_${fastq.baseName}')
print(json.dumps(res, indent=2))
with open('qc_${fastq.baseName}.json','w') as jf:
    json.dump(res, jf, indent=2)
" 2>&1 | tee qc.log

    # Also try native fastqc if present (optional, not required)
    if command -v fastqc >/dev/null 2>&1; then
        fastqc ${fastq} -o qc_${fastq.baseName} || echo 'fastqc failed, using python stub' | tee -a qc.log
    fi
    cat qc.log || true
    ls -R qc_${fastq.baseName} || true
    """
}
