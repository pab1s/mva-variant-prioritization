process TRIMMING {
    tag "${r1.baseName} / ${r2.baseName}"
    label 'trimming'
    cpus 4
    memory '16 GB'

    input:
    tuple path(r1), path(r2)

    output:
    tuple path("${r1.baseName}.trimmed.fastq.gz"), path("${r2.baseName}.trimmed.fastq.gz"), emit: trimmed

    script:
    """
    uv run python -c "
from src.preprocessing.trimming import trim_adapters
import sys
r1, r2 = trim_adapters('${r1}', '${r2}', '.')
print(f'Trimmed: {r1}, {r2}')
"
    # The python function creates files in current dir via symlink/copy if no fastp/cutadapt
    # Ensure expected outputs exist
    if [[ ! -f "${r1.baseName}.trimmed.fastq.gz" ]]; then ln -s ${r1} ${r1.baseName}.trimmed.fastq.gz || cp ${r1} ${r1.baseName}.trimmed.fastq.gz; fi
    if [[ ! -f "${r2.baseName}.trimmed.fastq.gz" ]]; then ln -s ${r2} ${r2.baseName}.trimmed.fastq.gz || cp ${r2} ${r2.baseName}.trimmed.fastq.gz; fi
    """
}
