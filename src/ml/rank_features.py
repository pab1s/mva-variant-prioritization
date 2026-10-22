"""Fixed numeric feature vector for the ranker. All features are local:
VCF fields + VEP CSQ + HPO config. No external services at train or
predict time. Every accessor is defensive: missing data -> neutral default.
"""

FEATURES = [
    "rare_score",        # 0-1, 1 = absent/ultra-rare in gnomAD
    "consequence_weight",  # 0-1 VEP SO severity
    "is_lof",            # stop_gained/frameshift/splice
    "is_missense",
    "is_splice",
    "impact_ord",        # HIGH 3 / MODERATE 2 / LOW 1 / MODIFIER 0
    "cadd_norm",         # 0-1 (real CADD_PHRED/30 when plugin present)
    "cadd_missing",      # 1 if CADD was heuristic
    "clinvar_score",     # 1 pathogenic / 0.5 unknown / 0.1 benign
    "quality_score",     # QD/MQ/FS/FILTER aggregate
    "qd_norm",
    "mq_norm",
    "phenotype_score",   # 1 MVA gene / 0.5 neutral
    "is_mva_gene",
    "is_het",            # 0/1 or phased het
    "is_novel",          # no rsID
    "filt_pass",
    "ab_dev",            # |0.5 - ALT/(REF+ALT)| allele balance deviation
    "dp_norm",           # log-scaled depth
    "exon_frac",         # EXON n/N -> n/N (NMD relevance), -1 if n/a
    "gene_het_count",    # rare hets in same gene (recessive signal)
    "compound_het_candidate",
]

LOF_CONS = {"stop_gained", "frameshift_variant", "splice_acceptor_variant",
            "splice_donor_variant", "transcript_ablation", "start_lost"}
IMPACT_ORD = {"HIGH": 3.0, "MODERATE": 2.0, "LOW": 1.0, "MODIFIER": 0.0}


def _f(x, default=0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except (TypeError, ValueError):
        return default


def row_to_vector(row: dict, mva_genes: set[str]) -> list[float]:
    """Scored-row dict (from prioritize) -> fixed feature vector.

    `row` keys used: rare_score, consequence_weight, Consequence, IMPACT,
    CADD_PHRED, cadd_missing?, clinvar_score, quality_score, QD, MQ,
    phenotype_score, SYMBOL, gene_source, GT, ID, FILTER, AD_REF, AD_ALT,
    DP, EXON, gene_het_count, compound_het_candidate.
    """
    cons = str(row.get("Consequence", "") or "")
    impact = str(row.get("IMPACT", "") or "").upper()
    gene = row.get("SYMBOL") or ""
    gt = str(row.get("GT", "") or "")
    vid = str(row.get("ID", "") or "")
    ad_ref, ad_alt = _f(row.get("AD_REF"), -1), _f(row.get("AD_ALT"), -1)
    if ad_ref >= 0 and ad_alt >= 0 and (ad_ref + ad_alt) > 0:
        ab_dev = abs(0.5 - ad_alt / (ad_ref + ad_alt))
    else:
        ab_dev = 0.5  # unknown -> neutral-ish (not het-like, not hom-like)
    exon = str(row.get("EXON", "") or "")
    try:
        n, m = exon.split("/")
        exon_frac = float(n) / float(m)
    except (ValueError, ZeroDivisionError):
        exon_frac = -1.0
    cadd_raw = row.get("CADD_PHRED")
    try:
        cadd_missing = 0.0 if (cadd_raw not in (None, "", ".")) else 1.0
    except TypeError:
        cadd_missing = 1.0
    return [
        _f(row.get("rare_score"), 1.0),
        _f(row.get("consequence_weight"), 0.5),
        1.0 if cons in LOF_CONS else 0.0,
        1.0 if cons == "missense_variant" else 0.0,
        1.0 if "splice" in cons else 0.0,
        IMPACT_ORD.get(impact, 0.0),
        _f(row.get("cadd_norm"), 0.3),
        cadd_missing,
        _f(row.get("clinvar_score"), 0.5),
        _f(row.get("quality_score"), 0.5),
        min(_f(row.get("QD"), 0.0) / 15.0, 1.0),
        min(_f(row.get("MQ"), 0.0) / 60.0, 1.0),
        _f(row.get("phenotype_score"), 0.5),
        1.0 if gene in mva_genes else 0.0,
        1.0 if gt in ("0/1", "1/0", "0|1", "1|0") else 0.0,
        1.0 if vid in ("", ".", "None") or not vid.startswith("rs") else 0.0,
        1.0 if str(row.get("FILTER", "")) in ("PASS", ".", "", "None") else 0.0,
        ab_dev,
        min(_f(row.get("DP"), 0.0) / 60.0, 1.0),
        exon_frac,
        min(_f(row.get("gene_het_count"), 0.0) / 5.0, 1.0),
        1.0 if row.get("compound_het_candidate") in (True, 1, "True", "true") else 0.0,
    ]
