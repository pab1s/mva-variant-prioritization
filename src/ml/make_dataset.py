"""Build the supervised training dataset for the LightGBM ranker.

Fully offline + public data only (ClinVar, HPOA, phenopacket store, VKGL,
AlphaMissense, gnomAD constraint). No patient data is read here.

Stages (run in order):

  1. select   variants + labels + query metadata + gene split
              -> data/ml/variants.vcf, labels.csv.gz, queries.csv.gz,
                 gene_counts.csv.gz, gene_split.json
  2. (external) VEP annotate variants.vcf -> data/ml/variants.vep.vcf.gz
               (src/annotation/vep.py, same cache/CADD/ClinVar args as the
                patient run -> feature parity)
  3. build    CSQ + AlphaMissense + LOEUF lookups -> static_features.csv.gz
  4. queries  per-query pools, phenotype features, retrieval filter,
              graded rows -> dataset_{train,val,test}.csv.gz + stats.json

Query design (see docs/PLAN_ml_ranker.md):
  - ClinVar-disease queries: variant P/LP -> CLNDISDB(OMIM) -> HPOA terms
    (augmented: subsample 70% + ancestor expansion + noise terms).
  - Phenopacket queries: real patient HPO + real causal variant (hg38).
  - Negatives: ClinVar B/LB + VKGL LB; hard negatives = P/LP of other
    diseases (label 0 for this query).

Labels (graded, A1):  3 = phenopacket causal · 2 = ClinVar P/LP >=2 stars ·
  1 = ClinVar P/LP 1 star · 0 = benign (ClinVar B/LB or VKGL LB).
  VUS are excluded everywhere (unlabeled, some are causal).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from src.phenotype.gene_hpo import GeneHPO
from src.phenotype.hpo_ontology import HPOOntology

ML_DIR = Path("data/ml")
CLINVAR = Path("data/clinvar/clinvar.vcf.gz")
VKGL = Path("data/annotations/vkgl_consensus_20260930.tsv")
PP_ROOT = Path("data/phenopackets/extracted/0.1.27")
G2P = Path("data/hpo/genes_to_phenotype.txt")
OBO = Path("data/hpo/hp.obo")
AM_TSV = Path("data/annotations/AlphaMissense_hg38_modified.tsv.gz")
LOEUF_TSV = Path("data/annotations/loeuf_dataset_grch38.tsv.gz")

SEED = 7
MAX_POS_PER_DISEASE = 30      # cap ClinVar-disease positives per disease
N_NEG_RANDOM = 60             # random benign negatives per query
N_NEG_HARD = 20               # P/LP-of-other-disease hard negatives per query
N_PP_NEG = 100                # negatives per phenopacket query
HPO_SUBSAMPLE = 0.7           # A5: keep fraction of disease terms
N_NOISE_TERMS = (2, 5)        # A5: noise terms from other diseases
ANC_PROB = 0.5                # A5: expand a term to its parents (p=)

REVIEW_HIGH = ("criteria_provided,_multiple_submitters,_no_conflicts",
               "reviewed_by_expert_panel", "practice_guideline")


def _key(chrom: str, pos: int, ref: str, alt: str) -> str:
    return f"{chrom}:{pos}:{ref}:{alt}"


def _norm_chrom(c: str) -> str:
    return c.removeprefix("chr")


# --------------------------------------------------------------------- hpoa
def load_hpoa() -> tuple[dict[str, set[str]], dict[str, set[str]], set[str]]:
    """disease_id -> HPO terms, gene -> disease ids, all disease ids."""
    dis_terms: dict[str, set[str]] = defaultdict(set)
    dis_genes: dict[str, set[str]] = defaultdict(set)
    with open(G2P) as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 6:
                continue
            gene, term, name, disease = p[1], p[2], p[3], p[5]
            if not term.startswith("HP:") or "inheritance" in name.lower():
                continue
            dis_terms[disease].add(term)
            dis_genes[disease].add(gene)
    return dis_terms, dis_genes, set(dis_terms)


# ------------------------------------------------------------------- clinvar
def scan_clinvar(hpoa_diseases: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """One pass over ClinVar: P/LP positives (joinable to HPOA), B/LB
    negatives, and per-gene P/LP counts."""
    rows_pos, rows_neg = [], []
    gene_counts: Counter[str] = Counter()
    cmd = ["bcftools", "query", "-f",
           "%CHROM\t%POS\t%REF\t%ALT\t%CLNSIG\t%CLNREVSTAT\t%CLNDISDB\t%GENEINFO\n",
           str(CLINVAR)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        chrom, pos, ref, alt, clnsig, revstat, clndisdb, geneinfo = line.rstrip("\n").split("\t")
        sig = clnsig.lower()
        gene = geneinfo.split(":")[0] if geneinfo and geneinfo != "." else ""
        if "conflicting" in sig or "uncertain" in sig or "not_provided" in sig or "association" in sig or "risk_factor" in sig or "drug_response" in sig or "protective" in sig or "other" == sig:
            continue
        is_path = "pathogenic" in sig and "benign" not in sig
        is_benign = "benign" in sig and "pathogenic" not in sig
        if is_path and gene:
            gene_counts[gene] += 1
        if is_path:
            # disease join: OMIM / Orphanet ids present in HPOA
            diseases = []
            for tok in (clndisdb or "").split(","):
                tok = tok.strip()
                if tok.startswith(("OMIM:", "Orphanet:", "MONDO:")):
                    if tok in hpoa_diseases:
                        diseases.append(tok)
            if diseases:
                stars = 2 if revstat in REVIEW_HIGH else 1
                rows_pos.append((chrom, int(pos), ref, alt, stars, ";".join(diseases[:3]), gene))
        elif is_benign:
            rows_neg.append((chrom, int(pos), ref, alt))
        if proc.poll() is not None:
            break
    proc.wait()
    cols_pos = ["CHROM", "POS", "REF", "ALT", "stars", "diseases", "GENE"]
    cols_neg = ["CHROM", "POS", "REF", "ALT"]
    return (pd.DataFrame(rows_pos, columns=cols_pos),
            pd.DataFrame(rows_neg, columns=cols_neg),
            dict(gene_counts))


# --------------------------------------------------------------- phenopacket
def load_phenopackets() -> list[dict]:
    """All phenopacket cases with a simple (SNV/indel) causal variant."""
    cases = []
    for path in sorted(PP_ROOT.glob("*/*.json")):
        with open(path) as fh:
            d = json.load(fh)
        feats = [f["type"]["id"] for f in d.get("phenotypicFeatures", [])
                 if not f.get("excluded") and f.get("type", {}).get("id", "").startswith("HP:")]
        excluded = [f["type"]["id"] for f in d.get("phenotypicFeatures", [])
                    if f.get("excluded") and f.get("type", {}).get("id", "").startswith("HP:")]
        disease = ""
        diseases = d.get("diseases", [])
        if diseases:
            disease = (diseases[0].get("term") or {}).get("id", "")
        variants, gene = [], ""
        for interp in d.get("interpretations", []):
            for gi in (interp.get("diagnosis", {}) or {}).get("genomicInterpretations", []) or []:
                vd = ((gi.get("variantInterpretation") or {}).get("variationDescriptor")) or {}
                if "structuralType" in vd:
                    continue  # CNV — out of scope for the SNV/indel ranker
                vcf = vd.get("vcfRecord")
                gene = gene or (vd.get("geneContext") or {}).get("symbol", "")
                if vcf and str(vcf.get("genomeAssembly", "")).lower() in ("hg38", "grch38", ""):
                    variants.append((_norm_chrom(str(vcf["chrom"])), int(vcf["pos"]),
                                     str(vcf["ref"]), str(vcf["alt"])))
        if feats and variants:
            cases.append({"case_id": path.stem, "file": str(path), "gene": gene,
                          "disease": disease, "hpo": sorted(set(feats)),
                          "excluded": sorted(set(excluded)), "variants": variants})
    return cases


def load_vkgl_benign() -> pd.DataFrame:
    df = pd.read_csv(VKGL, sep="\t")
    df = df[df["classification"] == "LB"][["chromosome", "start", "ref", "alt"]]
    df["CHROM"] = df["chromosome"].map(_norm_chrom)
    df["POS"] = df["start"].astype(int)
    return df[["CHROM", "POS", "ref", "alt"]].rename(columns={"ref": "REF", "alt": "ALT"})


# ------------------------------------------------------------------ a5: HPO
def augment_disease_hpo(disease_terms: set[str], rng: random.Random,
                        all_dis_terms: list[set[str]], ontology: HPOOntology | None
                        ) -> tuple[list[str], dict[str, float]]:
    """A5 noise model: subsample 70% + parent expansion + noise terms."""
    terms = sorted(disease_terms)
    keep = [t for t in terms if rng.random() < HPO_SUBSAMPLE] or terms[:2]
    weights = {t: 0.5 for t in keep}
    # ancestor expansion (patients often report the generic parent term)
    expanded = []
    for t in keep:
        if ontology is not None and rng.random() < ANC_PROB:
            for anc in ontology.parents.get(t, ()):
                expanded.append(anc)
    for t in set(expanded):
        weights.setdefault(t, 0.3)
    # noise terms from other diseases
    n_noise = rng.randint(*N_NOISE_TERMS)
    for _ in range(n_noise):
        pool = rng.choice(all_dis_terms)
        if pool:
            t = rng.choice(sorted(pool))
            weights.setdefault(t, 0.1)
    return sorted(weights), weights


# -------------------------------------------------------------------- select
def stage_select(args) -> None:
    ML_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    dis_terms, dis_genes, hpoa_diseases = load_hpoa()
    print(f"[select] HPOA: {len(dis_terms)} diseases with HPO terms")

    pos, neg, gene_counts = scan_clinvar(hpoa_diseases)
    print(f"[select] ClinVar: {len(pos)} P/LP joinable to HPOA, {len(neg)} B/LB, "
          f"{len(gene_counts)} genes with P/LP")

    pp_cases = load_phenopackets()
    pp_variants = {v for c in pp_cases for v in c["variants"]}
    print(f"[select] phenopackets: {len(pp_cases)} cases, {len(pp_variants)} causal variants")

    # ---- gene split (15% test / 15% val / 70% train) on phenopacket genes
    pp_genes = sorted({c["gene"] for c in pp_cases if c["gene"]})
    rng_split = random.Random(SEED + 1)
    rng_split.shuffle(pp_genes)
    n = len(pp_genes)
    split_of_gene = {}
    for i, g in enumerate(pp_genes):
        split_of_gene[g] = "test" if i < n * 0.15 else ("val" if i < n * 0.30 else "train")
    holdout_genes = {g for g, s in split_of_gene.items() if s != "train"}
    json.dump(split_of_gene, open(ML_DIR / "gene_split.json", "w"), indent=1)

    # anti-leakage: causal variants of val/test cases never appear in train
    heldout_causal = {v for c in pp_cases
                      if split_of_gene.get(c["gene"], "train") != "train"
                      for v in c["variants"]}

    # ---- positives B: cap per disease, drop held-out diseases (gene overlap)
    pos = pos[~pos.apply(lambda r: _key(r.CHROM, r.POS, r.REF, r.ALT) in heldout_causal, axis=1)]
    dis_of_var: dict[str, list[str]] = {}
    rows = []
    per_dis: Counter[str] = Counter()
    dis_list = pos["diseases"].str.split(";").tolist()
    order = list(range(len(pos)))
    rng.shuffle(order)  # deterministic via seeded rng
    heldout_dis: set[str] = set()
    for i in order:
        diseases = dis_list[i]
        d0 = diseases[0]
        if any(dis_genes.get(d, set()) & holdout_genes for d in diseases):
            heldout_dis.add(d0)
            continue
        if per_dis[d0] >= MAX_POS_PER_DISEASE:
            continue
        per_dis[d0] += 1
        r = pos.iloc[i]
        rows.append((r.CHROM, r.POS, r.REF, r.ALT, int(r.stars), ";".join(diseases)))
    pos = pd.DataFrame(rows, columns=["CHROM", "POS", "REF", "ALT", "stars", "diseases"])
    print(f"[select] positives B after caps/leakage: {len(pos)} variants, "
          f"{len(per_dis)} diseases, excluded {len(heldout_dis)} held-out diseases")

    # ---- negatives: ClinVar B/LB sample + VKGL LB (minus causal variants)
    neg = neg[~neg.apply(lambda r: _key(r.CHROM, r.POS, r.REF, r.ALT) in heldout_causal, axis=1)]
    neg = neg.drop_duplicates(subset=["CHROM", "POS", "REF", "ALT"])
    vkgl = load_vkgl_benign()
    neg_keys = set(_key(*v) for v in neg[["CHROM", "POS", "REF", "ALT"]].itertuples(index=False))
    vkgl = vkgl[~vkgl.apply(lambda r: _key(r.CHROM, r.POS, r.REF, r.ALT) in neg_keys, axis=1)]
    vkgl = vkgl[~vkgl.apply(lambda r: _key(r.CHROM, r.POS, r.REF, r.ALT) in heldout_causal, axis=1)]
    neg = pd.concat([neg, vkgl], ignore_index=True).drop_duplicates(
        subset=["CHROM", "POS", "REF", "ALT"])
    neg = neg.sample(n=min(args.neg_pool, len(neg)), random_state=SEED)
    print(f"[select] negative pool: {len(neg)} (ClinVar B/LB + VKGL LB)")

    # ---- labels table (grade: 3 causal, 2 P>=2star, 1 P 1star, 0 benign)
    # Aggregated PER VARIANT across sources so no information is lost:
    # a variant can be causal for several phenopacket cases (shared families)
    # AND carry ClinVar disease annotations; grade = max of all sources.
    agg: dict[str, dict] = {}

    def _slot(k, chrom, pos, ref, alt):
        s = agg.get(k)
        if s is None:
            s = agg[k] = {"CHROM": chrom, "POS": int(pos), "REF": ref, "ALT": alt,
                          "label": 0, "case_ids": set(), "diseases": set(),
                          "sources": set()}
        return s

    for c in pp_cases:
        for v in c["variants"]:
            s = _slot(_key(*v), *v)
            s["label"] = max(s["label"], 3)
            s["case_ids"].add(c["case_id"])
            s["sources"].add("phenopacket")
    for r in pos.itertuples(index=False):
        s = _slot(_key(r.CHROM, r.POS, r.REF, r.ALT), r.CHROM, r.POS, r.REF, r.ALT)
        s["label"] = max(s["label"], 2 if r.stars >= 2 else 1)
        s["diseases"].update(r.diseases.split(";"))
        s["sources"].add("clinvar")
    for r in neg.itertuples(index=False):
        k = _key(r.CHROM, r.POS, r.REF, r.ALT)
        if k in agg:
            continue  # positives win over benign rows (VKGL/ClinVar conflicts)
        s = _slot(k, r.CHROM, r.POS, r.REF, r.ALT)
        s["label"] = 0
        s["sources"].add("benign")

    labels = pd.DataFrame([
        {"CHROM": s["CHROM"], "POS": s["POS"], "REF": s["REF"], "ALT": s["ALT"],
         "label": s["label"],
         "case_ids": ";".join(sorted(s["case_ids"])),
         "diseases": ";".join(sorted(s["diseases"])),
         "sources": ";".join(sorted(s["sources"]))}
        for s in agg.values()])
    labels.to_csv(ML_DIR / "labels.csv.gz", index=False, compression="gzip")
    print(f"[select] labels: {len(labels)} variants "
          f"(gr3={(labels.label == 3).sum()}, gr>=1={(labels.label >= 1).sum()}, "
          f"gr0={(labels.label == 0).sum()})")

    # case_id -> set of its causal variant keys (for pp query positives)
    case_pos: dict[str, list[str]] = defaultdict(list)
    for c in pp_cases:
        for v in c["variants"]:
            case_pos[c["case_id"]].append(_key(*v))

    # ---- variants VCF (single-allelic, sites-only, coordinate-sorted) for VEP
    chrom_order = {str(i): i for i in range(1, 23)}
    chrom_order.update({"X": 23, "Y": 24, "MT": 25, "M": 25})
    labels["_co"] = labels.CHROM.map(lambda c: chrom_order.get(str(c), 99))
    labels = labels.sort_values(["_co", "POS"]).drop(columns="_co").reset_index(drop=True)
    with open(ML_DIR / "variants.vcf", "w") as out:
        out.write("##fileformat=VCFv4.2\n")
        for c in [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]:
            out.write(f"##contig=<ID={c}>\n")
        out.write('##INFO=<ID=SRC,Number=1,Type=String,Description="label source">\n')
        out.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for r in labels.itertuples(index=False):
            out.write(f"{r.CHROM}\t{r.POS}\t{r.REF}_{r.ALT}\t{r.REF}\t{r.ALT}\t.\t.\tSRC={r.sources}\n")
    print(f"[select] wrote {ML_DIR/'variants.vcf'} "
          f"({len(labels)} records) — run VEP next")

    # ---- queries metadata
    q_rows = []
    all_dis_term_sets = list(dis_terms.values())
    for diseases in sorted(set(pos["diseases"])):
        d0 = diseases.split(";")[0]
        if d0 not in dis_terms:
            continue
        terms, weights = augment_disease_hpo(dis_terms[d0], rng, all_dis_term_sets, None)
        q_rows.append({"query_id": f"clinvar::{d0}", "kind": "clinvar_disease",
                       "disease": d0, "hpo": ";".join(terms),
                       "weights": ";".join(f"{k}={v}" for k, v in sorted(weights.items())),
                       "excluded": "", "split": "train"})
    for c in pp_cases:
        s = split_of_gene.get(c["gene"], "train")
        # a case's causal variants drive per-case queries (pos_keys column:
        # cases sharing a variant each keep their own query)
        q_rows.append({"query_id": f"pp::{c['case_id']}", "kind": "phenopacket",
                       "disease": c["disease"], "hpo": ";".join(c["hpo"]),
                       "weights": ";".join(f"{t}=0.5" for t in c["hpo"]),
                       "excluded": ";".join(c["excluded"]), "split": s,
                       "pos_keys": ";".join(case_pos[c["case_id"]])})
    # clinvar queries resolve positives through the disease column
    for row in q_rows:
        row.setdefault("pos_keys", "")
    queries = pd.DataFrame(q_rows)
    queries.to_csv(ML_DIR / "queries.csv.gz", index=False, compression="gzip")

    pd.DataFrame(sorted(gene_counts.items()), columns=["gene", "n_path"]).to_csv(
        ML_DIR / "gene_counts.csv.gz", index=False, compression="gzip")
    stats = {
        "n_queries": len(queries), "n_train_q": int((queries.split == "train").sum()),
        "n_val_q": int((queries.split == "val").sum()),
        "n_test_q": int((queries.split == "test").sum()),
        "n_labels": len(labels), "n_pp_cases": len(pp_cases),
        "n_pos_b": len(pos), "n_neg_pool": len(neg),
        "holdout_genes": len(holdout_genes),
    }
    json.dump(stats, open(ML_DIR / "select_stats.json", "w"), indent=1)
    print("[select] stats:", stats)


# --------------------------------------------------------------------- build
def stage_build(args) -> None:
    """Parse VEP CSQ + AM + LOEUF -> static feature table (variant-level)."""
    from src.prioritization.prioritize import _parse_info, _parse_vcf_header

    header = _parse_vcf_header(str(args.vep_vcf))
    csq_fields = header["csq_fields"]

    # LOEUF gene -> (loeuf, pli) from MANE/canonical rows (shared with inference)
    from src.ml.lookup import load_loeuf
    print("[build] loading LOEUF ...")
    loeuf = load_loeuf()
    print(f"[build] LOEUF: {len(loeuf)} genes")

    gene_counts = pd.read_csv(ML_DIR / "gene_counts.csv.gz")
    path_count = dict(zip(gene_counts["gene"], gene_counts["n_path"]))

    # gnomAD AF comes from the AF side-file (the fast structural pass omits the
    # --af_* flags: they cost ~7.6x more and are only run over candidates).
    from src.ml.lookup import load_af, cadd_lookup, am_lookup, vkey as _vk
    af_path = Path(args.af_vcf) if args.af_vcf else ML_DIR / "train.af.vcf.gz"
    af_map = load_af(af_path)
    print(f"[build] gnomAD AF loaded: {len(af_map)} variants from {af_path}")

    rows = []
    opener = gzip.open if str(args.vep_vcf).endswith(".gz") else open
    with opener(args.vep_vcf, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            chrom, pos, vid, ref, alt = f[0], int(f[1]), f[2], f[3], f[4]
            info = _parse_info(f[7], csq_fields)
            gene = info.get("SYMBOL") or info.get("Gene") or ""
            if gene in (".", ""):
                gene = ""
            af_f = af_map.get(_vk(chrom, pos, ref, alt))
            clin = info.get("CLIN_SIG", "")
            lo, pli = loeuf.get(gene, (None, None))
            rows.append((
                chrom, pos, ref, alt,
                gene,
                info.get("Consequence", ""),
                info.get("IMPACT", ""),
                af_f,
                None,          # CADD filled in below via tabix (not in CSQ now)
                clin,
                info.get("EXON", ""),
                1 if info.get("CANONICAL") == "YES" else 0,
                1 if info.get("MANE_SELECT") not in ("", None) else 0,
                lo, pli,
                path_count.get(gene, 0),
            ))
    df = pd.DataFrame(rows, columns=[
        "CHROM", "POS", "REF", "ALT", "SYMBOL", "Consequence", "IMPACT",
        "gnomad_af", "CADD_PHRED", "CLIN_SIG", "EXON", "canonical", "mane",
        "gene_loeuf", "gene_pli", "gene_clinvar_path_count"])
    print(f"[build] CSQ parsed: {len(df)} variants "
          f"({(df.SYMBOL != '').sum()} with gene, "
          f"{df.gnomad_af.notna().sum()} with gnomAD AF)")

    # CADD: one tabix pass per CADD file (the VEP plugin was ~7x slower)
    print("[build] CADD lookups (tabix) ...")
    cadd = cadd_lookup([(r.CHROM, int(r.POS), r.REF, r.ALT) for r in df.itertuples()])
    df["CADD_PHRED"] = [
        cadd.get(_vk(c, p, r, a)) for c, p, r, a in
        df[["CHROM", "POS", "REF", "ALT"]].itertuples(index=False)]
    print(f"[build] CADD matched: {df.CADD_PHRED.notna().sum()}/{len(df)}")

    # AlphaMissense: one tabix pass over missense variants (shared lookup)
    print("[build] AlphaMissense lookups (tabix) ...")
    miss = df[df["Consequence"].str.contains("missense", na=False)]
    am = am_lookup([(r.CHROM, int(r.POS), r.REF, r.ALT) for r in miss.itertuples()])
    df["am_pathogenicity"] = [
        am.get(_vk(c, p, r, a), (None, ""))[0] for c, p, r, a in
        df[["CHROM", "POS", "REF", "ALT"]].itertuples(index=False)]
    print(f"[build] AlphaMissense matched: {df.am_pathogenicity.notna().sum()} "
          f"of {len(miss)} missense")

    df.to_csv(ML_DIR / "static_features.csv.gz", index=False, compression="gzip")
    print(f"[build] wrote {ML_DIR/'static_features.csv.gz'}")


# ------------------------------------------------------------------- queries
def stage_queries(args) -> None:
    rng = random.Random(SEED + 2)
    ont = HPOOntology(str(OBO))
    gh = GeneHPO(str(G2P), ont)

    static = pd.read_csv(ML_DIR / "static_features.csv.gz")
    static["vkey"] = (static.CHROM.astype(str) + ":" + static.POS.astype(str)
                      + ":" + static.REF + ":" + static.ALT)
    static = static.set_index("vkey", drop=False)
    labels = pd.read_csv(ML_DIR / "labels.csv.gz")
    labels["vkey"] = labels.CHROM.astype(str) + ":" + labels.POS.astype(str) + ":" + labels.REF + ":" + labels.ALT
    queries = pd.read_csv(ML_DIR / "queries.csv.gz")
    lab_by_key = labels.set_index("vkey")
    # disease -> positive variant keys (exact membership, not substring)
    pos_keys_by_disease: dict[str, list[str]] = defaultdict(list)
    for vkey, dis_str in zip(labels["vkey"], labels["diseases"].fillna("")):
        lab = lab_by_key.loc[vkey, "label"]
        if lab >= 1 and dis_str:
            for d in dis_str.split(";"):
                pos_keys_by_disease[d].append(vkey)

    # annotate negatives with their disease-free benign grade
    benign_list = list(labels.loc[labels.label == 0, "vkey"])
    # hard-negative pool: P/LP variants (label>=1) with their diseases
    path_df = labels[labels.label >= 1].copy()
    path_df["dis0"] = path_df["diseases"].fillna("").map(lambda s: s.split(";")[0])
    path_keys = list(path_df["vkey"])
    path_dis0 = list(path_df["dis0"])

    panel = set(args.panel.split(",")) if args.panel else set()

    rows_out: list[pd.DataFrame] = []
    retrieval_pos_total = retrieval_pos_kept = 0
    per_split_pos = Counter()
    per_split_kept = Counter()
    grade_tot: Counter = Counter()   # (split, label_grade) -> positives seen
    grade_kept: Counter = Counter()  # (split, label_grade) -> positives kept
    reasons: Counter = Counter()     # failure attribution

    for q in queries.itertuples(index=False):
        qterms = set(str(q.hpo).split(";")) if isinstance(q.hpo, str) else set()
        qterms = {t for t in qterms if ont.is_known(t)}
        if not qterms:
            continue
        weights = {}
        if isinstance(q.weights, str):
            for tok in q.weights.split(";"):
                if "=" in tok:
                    k, v = tok.split("=")
                    weights[k] = float(v)
        excluded = (set(str(q.excluded).split(";")) if isinstance(q.excluded, str)
                    and q.excluded == q.excluded and str(q.excluded) != "nan" else set())
        norm = ont.query_norm_factor(qterms)

        # ---- positives for this query
        if q.kind == "phenopacket":
            raw = q.pos_keys if isinstance(q.pos_keys, str) else ""
            pos_keys = [k for k in raw.split(";") if k and k != "nan"]
            if not pos_keys:  # fallback: match through the aggregated label row
                cid = q.query_id.split("::")[1]
                pos_keys = list(labels[(labels.case_ids.fillna("") == cid) &
                                       (labels.label >= 3)]["vkey"])
        else:
            d = q.query_id.split("::")[1]
            pos_keys = list(pos_keys_by_disease.get(d, []))
        if not pos_keys:
            continue
        pos_keys = [k for k in pos_keys if k in static.index]
        if not pos_keys:
            continue

        # ---- negatives for this query
        n_rand = N_PP_NEG if q.kind == "phenopacket" else N_NEG_RANDOM
        neg_rand = rng.sample(benign_list, min(n_rand, len(benign_list)))
        dis0_q = q.query_id.split("::")[1] if q.kind == "clinvar_disease" else ""
        pos_set = set(pos_keys)
        neg_hard: list[str] = []
        tries = 0
        while len(neg_hard) < N_NEG_HARD and tries < N_NEG_HARD * 10:
            i = rng.randrange(len(path_keys))
            tries += 1
            k = path_keys[i]
            if path_dis0[i] == dis0_q or k in pos_set or k in neg_hard:
                continue
            neg_hard.append(k)

        pool_keys = list(dict.fromkeys(
            k for k in pos_keys + neg_rand + neg_hard if k in static.index))
        if not pool_keys:
            continue
        pool = static.loc[pool_keys].copy()
        pool["label"] = [int(lab_by_key.loc[k, "label"]) if k in pos_keys else 0
                         for k in pool["vkey"]]
        # grade-0 for hard negatives (P/LP of other diseases)
        pool.loc[pool.vkey.isin(neg_hard), "label"] = 0

        # ---- dynamic phenotype features per gene
        gene_set = set(pool["SYMBOL"].dropna()) - {""}
        pheno = {g: gh.gene_similarity(qterms, g, norm, weights, excluded)
                 for g in gene_set}
        from src.ml.candidates import PHENO_GATE_MIN_NORM
        pheno_gate = {g for g, s in pheno.items()
                      if s["pheno_resnik"] > PHENO_GATE_MIN_NORM}

        # ---- retrieval filter (parity with inference) + failure accounting
        from src.ml.candidates import retrieval_components
        comp = retrieval_components(pool, pheno_gate, panel, args.af_thresh)
        keep = (comp["rare"] & (comp["cw_ok"] | comp["gate_ok"])) | comp["clin_path"]
        for i in pool.index[pool["label"] > 0]:
            g = int(pool.at[i, "label"])
            grade_tot[(q.split, g)] += 1
            if keep.at[i]:
                grade_kept[(q.split, g)] += 1
            elif not comp["rare"].at[i]:
                reasons["common: AF >= threshold"] += 1
            else:
                gene = pool.at[i, "SYMBOL"]
                res = pheno.get(gene, {}).get("pheno_resnik", 0.0) if gene else 0.0
                if not gene or gene in (".", ""):
                    reasons["no gene annotation"] += 1
                elif res <= 0.0:
                    reasons["gene: zero phenotype overlap"] += 1
                elif res <= PHENO_GATE_MIN_NORM:
                    reasons[f"gene: weak overlap (0<resnik<=gate)"] += 1
                else:
                    reasons["in gate but low consequence? (bug)"] += 1
        n_pos = int((pool.label > 0).sum())
        n_pos_kept = int(((pool.label > 0) & keep).sum())
        retrieval_pos_total += n_pos
        retrieval_pos_kept += n_pos_kept
        per_split_pos[q.split] += n_pos
        per_split_kept[q.split] += n_pos_kept

        pool = pool[keep]
        if pool.empty or (pool.label > 0).sum() == 0:
            continue
        for g, s in pheno.items():
            m = pool["SYMBOL"] == g
            if not m.any():
                continue
            for k, v in s.items():
                pool.loc[m, k] = v
        for col in ("pheno_resnik", "pheno_bma", "pheno_jaccard",
                    "pheno_terms_matched", "pheno_excluded_overlap",
                    "gene_hpo_count", "gene_disease_count"):
            if col in pool.columns:
                pool[col] = pool[col].fillna(0.0)
            else:
                pool[col] = 0.0
        pool["query_id"] = q.query_id
        pool["split"] = q.split
        rows_out.append(pool)

    data = pd.concat(rows_out, ignore_index=True)
    ML_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        sub = data[data.split == split]
        sub.to_csv(ML_DIR / f"dataset_{split}.csv.gz", index=False, compression="gzip")
        print(f"[queries] {split}: {len(sub)} rows, {sub.query_id.nunique()} queries, "
              f"{int((sub.label > 0).sum())} positives")
    by_grade: dict[str, dict] = {}
    by_split_grade: dict[str, dict] = {}
    for (split, g) in sorted(grade_tot):
        t, k = grade_tot[(split, g)], grade_kept[(split, g)]
        by_split_grade[f"{split}|{g}"] = {"total": t, "kept": k,
                                          "recall": k / t if t else None}
    for g in (1, 2, 3):
        t = sum(v for (s, gg), v in grade_tot.items() if gg == g)
        k = sum(v for (s, gg), v in grade_kept.items() if gg == g)
        by_grade[str(g)] = {"total": t, "kept": k,
                            "recall": k / t if t else None}
    stats = {
        "retrieval_pos_total": retrieval_pos_total,
        "retrieval_pos_kept": retrieval_pos_kept,
        "retrieval_recall": retrieval_pos_kept / max(retrieval_pos_total, 1),
        "per_split_recall": {s: per_split_kept[s] / per_split_pos[s]
                             for s in per_split_pos if per_split_pos[s]},
        "by_label_grade": by_grade,          # 3=case causal, 2=P/LP>=2*, 1=P/LP 1*
        "by_split_grade": by_split_grade,
        "fail_reasons": dict(reasons),
        "n_rows": len(data),
    }
    json.dump(stats, open(ML_DIR / "query_stats.json", "w"), indent=1)
    print("[queries] retrieval by grade:", by_grade)
    print("[queries] fail reasons:", dict(reasons))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    s1 = sub.add_parser("select")
    s1.add_argument("--neg-pool", type=int, default=250_000)
    s1.set_defaults(func=stage_select)
    s2 = sub.add_parser("build")
    s2.add_argument("--vep-vcf", default=str(ML_DIR / "variants.vep.vcf.gz"))
    s2.add_argument("--af-vcf", default=None,
                    help="side VCF annotated with --af_gnomade/af_gnomadg "
                         "(default: data/ml/train.af.vcf.gz)")
    s2.set_defaults(func=stage_build)
    s3 = sub.add_parser("queries")
    s3.add_argument("--panel", default="", help="comma-separated gene panel (optional)")
    s3.add_argument("--af-thresh", type=float, default=0.01)
    s3.set_defaults(func=stage_queries)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
