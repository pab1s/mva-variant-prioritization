"""ESM-2 variant embeddings for the ranker (experimental feature block).

The deployed schema already carries AlphaMissense (`am_norm`), which is a
scalar pathogenicity prediction — AlphaMissense does not release embeddings.
This module extracts *representations* from a protein language model instead:

  - `delta`     : last-layer hidden state at the mutated residue, mutant window
                  minus wild-type window (480-d for esm2_t12_35M).
  - `esm_score` : masked-marginal log-odds of the wild-type vs mutant residue
                  (ESM-1v-style zero-shot variant effect score; higher = more
                  deleterious).

Only missense variants with a valid transcript/protein mapping get features;
everything else is zero-filled by the training script.

Input : data/ml/variants.vep.vcf.gz (VEP CSQ) + Ensembl peptide FASTA
Output: data/ml/esm2_35m_{delta.npy,index.csv.gz,meta.json}

Run:
  .venv/bin/python -m src.ml.esm_features build
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ML_DIR = ROOT / "data/ml"
DEFAULT_VEP = ML_DIR / "variants.vep.vcf.gz"
DEFAULT_FASTA = ROOT / "data/references/Homo_sapiens.GRCh38.pep.all.fa.gz"
DEFAULT_OUT = ML_DIR / "esm2_35m"
MODEL = "facebook/esm2_t12_35M_UR50D"
STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _vkey(chrom, pos, ref, alt) -> str:
    return f"{chrom}:{pos}:{ref}:{alt}"


def load_proteins(path: Path) -> dict[str, str]:
    """ENST (versionless) -> protein sequence from an Ensembl pep FASTA."""
    seqs: dict[str, str] = {}
    name: str | None = None
    buf: list[str] = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(buf)
                m = re.search(r"transcript:(\S+)", line)
                name = m.group(1).split(".")[0] if m else None
                buf = []
            elif name:
                buf.append(line.strip())
    if name:
        seqs[name] = "".join(buf)
    return seqs


def parse_missense(vep_path: Path, seqs: dict[str, str]
                   ) -> tuple[pd.DataFrame, dict[str, int]]:
    """Pick the first CSQ entry with a valid protein mapping per variant."""
    from src.prioritization.prioritize import _parse_vcf_header

    csq_fields = _parse_vcf_header(str(vep_path))["csq_fields"]
    assert csq_fields, "VEP VCF has no CSQ header"
    rows: list[tuple] = []
    stats = {"records": 0, "no_csq": 0, "no_missense": 0, "no_valid_tx": 0,
             "ok": 0}
    opener = gzip.open if str(vep_path).endswith(".gz") else open
    with opener(vep_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            stats["records"] += 1
            p = line.rstrip("\n").split("\t")
            info = p[7]
            if "CSQ=" not in info:
                stats["no_csq"] += 1
                continue
            csq = info.split("CSQ=")[1].split(";")[0]
            chosen = None
            n_miss = 0
            for entry in csq.split(","):
                d = dict(zip(csq_fields, entry.split("|")))
                if "missense_variant" not in d.get("Consequence", ""):
                    continue
                n_miss += 1
                feat = d.get("Feature", "").split(".")[0]
                pos_s = d.get("Protein_position", "").split("-")[0]
                aa = d.get("Amino_acids", "")
                if not (feat and pos_s.isdigit() and "/" in aa):
                    continue
                wt, mut = aa.split("/")[:2]
                if wt not in STANDARD_AA or mut not in STANDARD_AA:
                    continue
                seq = seqs.get(feat)
                pos = int(pos_s)
                if not seq or pos < 1 or pos > len(seq):
                    continue
                if seq[pos - 1] != wt:
                    continue
                chosen = (feat, pos, wt, mut, d.get("SYMBOL", ""), len(seq))
                break
            if n_miss == 0:
                stats["no_missense"] += 1
            elif chosen is None:
                stats["no_valid_tx"] += 1
            else:
                stats["ok"] += 1
                rows.append((p[0], int(p[1]), p[3], p[4]) + chosen)
    cols = ["CHROM", "POS", "REF", "ALT", "ENST", "protein_pos", "wt", "mut",
            "SYMBOL", "protein_len"]
    df = pd.DataFrame(rows, columns=cols)
    df["vkey"] = [_vkey(c, p, r, a) for c, p, r, a in
                  df[["CHROM", "POS", "REF", "ALT"]].itertuples(index=False)]
    return df, stats


def _windows(df: pd.DataFrame, seqs: dict[str, str], flank: int
             ) -> tuple[list[str], list[str], list[int]]:
    wt_seqs, mut_seqs, centers = [], [], []
    for r in df.itertuples(index=False):
        seq = seqs[r.ENST]
        i = r.protein_pos - 1
        lo = max(0, i - flank)
        hi = min(len(seq), i + flank + 1)
        wt = seq[lo:hi]
        mut = wt[: i - lo] + r.mut + wt[i - lo + 1:]
        wt_seqs.append(wt)
        mut_seqs.append(mut)
        centers.append(i - lo)
    return wt_seqs, mut_seqs, centers


def build(args) -> None:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    t0 = time.time()
    seqs = load_proteins(Path(args.fasta))
    print(f"[esm] proteins: {len(seqs)}", flush=True)
    df, stats = parse_missense(Path(args.vep), seqs)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    print(f"[esm] missense variants with mapping: {len(df)} {stats}", flush=True)
    if df.empty:
        raise SystemExit("no variants to embed")

    wt_seqs, mut_seqs, centers = _windows(df, seqs, args.flank)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    model = AutoModelForMaskedLM.from_pretrained(args.model).to(device).eval()
    if device == "cuda":
        model = model.half()
    aa_ids = {a: tokenizer.convert_tokens_to_ids(a) for a in STANDARD_AA}
    hidden = model.config.hidden_size
    print(f"[esm] model={args.model} hidden={hidden} device={device} "
          f"variants={len(df)} flank={args.flank}", flush=True)

    n = len(df)
    delta = np.zeros((n, hidden), dtype=np.float16)
    esm_score = np.zeros(n, dtype=np.float32)
    batch = args.batch

    def _forward(seqs_batch: list[str]):
        enc = tokenizer(seqs_batch, return_tensors="pt", padding=True,
                        truncation=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            return model.esm(**enc).last_hidden_state.float()

    for start in range(0, n, batch):
        end = min(start + batch, n)
        idx = np.arange(start, end)
        # token index: +1 for the leading <cls>
        tok_idx = torch.tensor([centers[i] + 1 for i in idx], device=device)
        b_wt = [wt_seqs[i] for i in idx]
        b_mut = [mut_seqs[i] for i in idx]

        h_wt = _forward(b_wt)[torch.arange(len(idx), device=device), tok_idx]
        h_mut = _forward(b_mut)[torch.arange(len(idx), device=device), tok_idx]
        delta[start:end] = (h_mut - h_wt).cpu().numpy().astype(np.float16)

        # masked-marginal: mask the WT residue, score mutant vs WT logits
        masked = []
        for i in idx:
            s = wt_seqs[i]
            c = centers[i]
            masked.append(s[:c] + tokenizer.mask_token + s[c + 1:])
        enc = tokenizer(masked, return_tensors="pt", padding=True,
                        truncation=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = model(**enc).logits
        logits = logits[torch.arange(len(idx), device=device), tok_idx]
        for j, i in enumerate(idx):
            wt_id = aa_ids[df.iloc[i]["wt"]]
            mu_id = aa_ids[df.iloc[i]["mut"]]
            esm_score[i] = float(logits[j, wt_id] - logits[j, mu_id])
        del h_wt, h_mut, logits

        if (start // batch) % 10 == 0 or end == n:
            print(f"[esm] {end}/{n} ({100 * end / n:.1f}%) "
                  f"{(time.time() - t0):.0f}s", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(f"{out}_delta.npy", delta)
    index = df[["vkey", "CHROM", "POS", "REF", "ALT", "SYMBOL", "ENST",
                "protein_pos", "wt", "mut", "protein_len"]].copy()
    index["esm_score"] = esm_score
    index.to_csv(f"{out}_index.csv.gz", index=False, compression="gzip")
    meta = {"model": args.model, "hidden_size": hidden, "flank": args.flank,
            "n_variants": n, "n_missense_in_vep": stats["ok"] + stats["no_valid_tx"],
            "parse_stats": stats, "device": device,
            "seconds": round(time.time() - t0, 1)}
    Path(f"{out}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[esm] wrote {out}_delta.npy {delta.shape} and "
          f"{out}_index.csv.gz", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--vep", default=str(DEFAULT_VEP))
    b.add_argument("--fasta", default=str(DEFAULT_FASTA))
    b.add_argument("--out", default=str(DEFAULT_OUT))
    b.add_argument("--model", default=MODEL)
    b.add_argument("--flank", type=int, default=25)
    b.add_argument("--batch", type=int, default=128)
    b.add_argument("--limit", type=int, default=0, help="smoke-test only")
    b.add_argument("--device", default="cuda")
    b.set_defaults(func=build)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
