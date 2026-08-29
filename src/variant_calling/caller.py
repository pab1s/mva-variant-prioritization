"""Variant calling wrappers (GATK / DeepVariant / bcftools)."""
def call_variants(bam_path: str, reference: str, out_vcf: str) -> str:
    """Call variants from BAM."""
    raise NotImplementedError
