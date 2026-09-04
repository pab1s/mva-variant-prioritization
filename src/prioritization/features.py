"""Feature engineering for variant pathogenicity.

Maps VCF row → numeric features used in scoring.
Baseline is intentionally simple but calibrated for rare disease:
  - rare AF
  - consequence severity
  - quality metrics (QD, MQ, FS)
  - CADD/REVEL if present, else heuristic

Designed to work with both VEP-annotated and raw VCFs (like the provided 5M-row Sentieon VCF).
"""
from typing import Dict, Any
import math

# Consequence severity ranking (VEP SO terms → weight)
CONSEQUENCE_WEIGHTS = {
    # high impact
    "transcript_ablation": 1.0,
    "splice_acceptor_variant": 1.0,
    "splice_donor_variant": 1.0,
    "stop_gained": 1.0,
    "frameshift_variant": 1.0,
    "stop_lost": 0.95,
    "start_lost": 0.95,
    "transcript_amplification": 0.9,
    # moderate
    "inframe_insertion": 0.85,
    "inframe_deletion": 0.85,
    "missense_variant": 0.8,
    "protein_altering_variant": 0.8,
    "splice_region_variant": 0.75,
    # low
    "incomplete_terminal_codon_variant": 0.5,
    "stop_retained_variant": 0.5,
    "synonymous_variant": 0.3,
    "coding_sequence_variant": 0.4,
    # non-coding / modifiers
    "mature_miRNA_variant": 0.4,
    "5_prime_UTR_variant": 0.3,
    "3_prime_UTR_variant": 0.3,
    "non_coding_transcript_exon_variant": 0.25,
    "intron_variant": 0.2,
    "NMD_transcript_variant": 0.6,
    "non_coding_transcript_variant": 0.2,
    "upstream_gene_variant": 0.15,
    "downstream_gene_variant": 0.15,
    "TFBS_ablation": 0.5,
    "TFBS_amplification": 0.5,
    "TF_binding_site_variant": 0.3,
    "regulatory_region_variant": 0.3,
    "intergenic_variant": 0.1,
}


def _parse_csq(csq_str: str | None) -> Dict[str, Any]:
    """Parse VEP CSQ field first transcript (format allele|Consequence|IMPACT|SYMBOL|...)."""
    if not csq_str:
        return {}
    # CSQ is comma-separated transcripts, take first
    first = csq_str.split(",")[0]
    # We don't know exact header order without VCF header, but common fields:
    # Allele|Consequence|IMPACT|SYMBOL|Gene|Feature_type|Feature|BIOTYPE|EXON|INTRON|HGVSc|HGVSp|cDNA_position|CDS_position|Protein_position|Amino_acids|Codons|Existing_variation|DISTANCE|STRAND|FLAGS|VARIANT_CLASS|MINIMISED|SYMBOL_SOURCE|HGNC_ID|CANONICAL|TSL|APPRIS|CCDS|ENSP|SWISSPROT|TREMBL|UNIPARC|UNIPROT_ISOFORM|GENE_PHENO|SIFT|PolyPhen|EXON|... etc
    parts = first.split("|")
    # Heuristic: consequence is usually 1st after allele, SYMBOL is 3rd
    d = {"raw": first}
    if len(parts) >= 2:
        d["consequence"] = parts[1]
    if len(parts) >= 4:
        d["symbol"] = parts[3]
    return d


def build_features(variant_row: dict) -> dict:
    """Convert a variant dict (from VCF) to feature dict.

    Expected variant_row keys (all optional):
      CHROM, POS, REF, ALT, QUAL, FILTER, QD, MQ, FS, SOR, DP,
      AF (internal), gnomad_af, gnomAD_AF, CADD_PHRED, REVEL, CLIN_SIG,
      CSQ, Consequence, IMPACT, SYMBOL, Gene

    Returns dict with normalized 0-1 features:
      - rare_score
      - consequence_weight
      - cadd_norm
      - clinvar_score
      - quality_score
      - pathogenicity_score (aggregated)
    """
    # --- AF / rarity ---
    # Prefer population AF (gnomAD) over internal cohort AF
    af = None
    for k in ["gnomad_af", "gnomAD_AF", "AF_gnomAD", "gnomADe_AF", "pop_af"]:
        if k in variant_row and variant_row[k] not in (None, "", "."):
            try:
                af = float(variant_row[k])
                break
            except:
                pass
    # Fallback: if only internal AF (0.5/1) and DP=2, it's not population → ignore
    # So only use AF if AN>2 or missing, but baseline just treats missing as rare
    if af is None:
        rare_score = 1.0  # assume rare if no population AF (good for rare disease)
    else:
        # common variants are less likely causal
        if af < 0.001:
            rare_score = 1.0
        elif af < 0.01:
            rare_score = 0.8
        elif af < 0.05:
            rare_score = 0.3
        else:
            rare_score = 0.05

    # --- Consequence ---
    csq = variant_row.get("CSQ") or variant_row.get("Consequence") or ""
    if not csq:
        # Infer from REF/ALT length if no annotation
        ref = str(variant_row.get("REF", ""))
        alt = str(variant_row.get("ALT", ""))
        if len(ref) != len(alt):
            # indel
            if len(ref) % 3 != 0 or len(alt) % 3 != 0:
                csq = "frameshift_variant"
            else:
                csq = "inframe_insertion" if len(alt) > len(ref) else "inframe_deletion"
        else:
            # SNV: assume missense as baseline (more sensitive than intronic)
            csq = "missense_variant"
    # CSQ may contain multiple consequences separated by &
    cons_list = str(csq).split("&")[0].split(",")[0].split("|")  # handle both
    # Take max weight among listed
    weights = [CONSEQUENCE_WEIGHTS.get(c.strip(), 0.5) for c in cons_list if c.strip()]
    consequence_weight = max(weights) if weights else 0.5

    # --- CADD / REVEL ---
    cadd = variant_row.get("CADD_PHRED") or variant_row.get("CADD") or variant_row.get("CADD_raw")
    try:
        cadd_f = float(cadd) if cadd not in (None, "", ".") else None
    except:
        cadd_f = None
    if cadd_f is not None:
        cadd_norm = min(max(cadd_f / 30.0, 0.0), 1.0)  # PHRED 30 ~ top 0.1%
    else:
        # heuristic: high consequence → pseudo-CADD 20, otherwise 10
        cadd_norm = 0.7 if consequence_weight >= 0.8 else 0.35

    # --- ClinVar ---
    clin = str(variant_row.get("CLIN_SIG") or variant_row.get("ClinVar") or "").lower()
    if "pathogenic" in clin and "benign" not in clin:
        clinvar_score = 1.0
    elif "likely_pathogenic" in clin:
        clinvar_score = 0.8
    elif "uncertain" in clin or "vus" in clin:
        clinvar_score = 0.5
    elif "benign" in clin:
        clinvar_score = 0.1
    else:
        clinvar_score = 0.5  # unknown

    # --- Quality ---
    # QD good if >2, MQ good if >40, FS good if <60
    try:
        qd = float(variant_row.get("QD", 0))
    except:
        qd = 0
    try:
        mq = float(variant_row.get("MQ", 0))
    except:
        mq = 0
    try:
        fs = float(variant_row.get("FS", 0))
    except:
        fs = 0
    filt = str(variant_row.get("FILTER", ""))
    qd_score = min(qd / 15.0, 1.0) if qd else 0.5
    mq_score = min(mq / 60.0, 1.0) if mq else 0.5
    fs_score = 1.0 - min(fs / 100.0, 1.0) if fs else 0.8
    filt_score = 0.2 if filt not in (".", "PASS", "", "None") else 1.0
    quality_score = (qd_score * 0.4 + mq_score * 0.3 + fs_score * 0.2 + filt_score * 0.1)

    # --- Aggregated pathogenicity (weighted) ---
    pathogenicity_score = (
        rare_score * 0.3 +
        consequence_weight * 0.3 +
        cadd_norm * 0.25 +
        clinvar_score * 0.1 +
        quality_score * 0.05
    )

    return {
        "rare_score": rare_score,
        "consequence_weight": consequence_weight,
        "consequence": csq,
        "cadd_norm": cadd_norm,
        "clinvar_score": clinvar_score,
        "quality_score": quality_score,
        "pathogenicity_score": pathogenicity_score,
        "af": af,
    }
