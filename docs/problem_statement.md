# Rare Disease, Real Kid: The MVA Hackathon 2026 — Problem Statement

> **Source:** https://sagebio-rare-disease-real-kid-mva-hackathon-2026.hf.space/  
> **Organizers:** MVA Society, Sage Bionetworks, Hugging Face, BEACON (Benchmarking, Evaluation, and Assessment Consortium for Science)  
> **Sponsors:** AWS Imagine Grant Program + Anthropic (Claude Science credits) — $50,000 total prizes  
> **Data IRB:** WCG IRB protocol #20252010 — controlled access, CC BY 4.0 for research outputs  
> **This document is a curated summary for engineering purposes. Refer to the live Space for authoritative rules and timeline updates.**

---

## 1. Context — Why This Hackathon Exists

Every human genome carries millions of small variations; almost all are benign. For ultra-rare diseases, clinicians recognize the syndrome but often do not know which variant(s) are causal.

**Mosaic Variegated Aneuploidy (MVA)** affects **fewer than 50 people worldwide**. It is characterized by mosaic aneuploidies (variable chromosome gains/losses across cells), predisposition to cancer, growth delay and developmental issues. **There is no established treatment** — care is symptomatic plus cancer surveillance/management. The **MVA Society** is funding focused research toward therapies.

A family living with MVA made the exceptionally personal decision to **share their child's real genome + clinical story** — with full explicit consent — and to open it to the global research community in hopes someone can help understand the condition. Illustration courtesy of MVA Society (child's drawing of himself smiling in hospital bed).

> Hackathon is **not** intended to provide general medical care, diagnosis, or professional medical advice.

Access to raw data (FASTQ + VCF + phenotype) is **controlled**: participants must request access on the Hugging Face dataset page and agree to data-use rules. **Outputs (code, reports, submissions) are open under CC BY 4.0** so methods reuse to other undiagnosed cases.

---

## 2. The Data You Are Given (Track 1 Input)

- **Genomic:** FASTQ (whole-genome) + VCF (provided call set). You may re-call or re-process.
- **Clinical:** Symptom / phenotype description for the child (to be mapped to HPO terms).

You are asked to predict which specific variant(s) drive the condition. **Secondary / incidental findings are welcome** and will not penalize the automated score, but should be reported responsibly.

---

## 3. Two Tracks

### Track 1 — Variant Prediction (Computational, Auto-Scored + Live Leaderboard)

**Goal:** From FASTQ/VCF + phenotype, **rank the causal variant(s)** responsible for the child's presentation.

**What you submit:**
- Ranked list of variant predictions (with confidence scores)
- GitHub link
- Methods write-up

**Submission limit:** 6

**Automated scoring — two primary metrics:**

1. **Rank Points** — how high you ranked the true variant(s):
   - `100 pts` if top-ranked variant(s) is correct.
   - Partial credit if correct variant appears within **top 10**.
   - If condition is caused by a **compound heterozygous pair**, correctly identifying **one of the two** gives **half credit**.
2. **F-max** — best balance of precision & recall achieved at any confidence threshold. Rewards pinpointing the right answer without burying it in noise. Secondary/incidental findings do **not** hurt this score.

Your predictions are checked against a **clinically confirmed held-out answer**; leaderboard updates live.

### Track 2 — Drug Repurposing (Hypothesis, Expert-Judged)

**Goal:** Given (or hypothesizing) the disrupted biological mechanism(s), **propose existing approved medication(s)** that are plausible candidates for follow-up investigation. Submissions are **hypotheses for follow-up, not evidence of efficacy**.

**Steps:**
1. Characterize the mechanism behind the variant(s).
2. Propose candidate drug(s) and justify via mechanism, with literature/preclinical reasoning.

**What you submit:**
- Detailed report
- GitHub link
- 3-minute pitch video

**Submission limit:** 1

**Judging criteria (expert panel, weighted):**

| Criterion | Weight | What judges look for |
|---|---|---:|
| **Scientific Rigor** | 35% | Is mechanism characterization sound? Is repurposing well-supported? |
| **Potential Impact** | 25% | If validated, contribution to understanding/diagnosing MVA (this child + others) |
| **Innovation** | 25% | Creative angle, method, or tool |
| **Scalability** | 15% | Could approach apply to other individuals/diseases/datasets |

---

## 4. Prizes — $50,000 total

- **$25,000 cash** via AWS Imagine Grant
- **$25,000 Claude credits** via Anthropic (for Claude Science, to deepen analysis — not cash-equivalent)

| Place | Cash | Claude Credits |
|---|---|---|
| 🥇 1st | $12,000 | $12,000 |
| 🥈 2nd | $7,000 | $7,000 |
| 🥉 3rd | $4,000 | $4,000 |
| 🌟 Innovation | $2,000 | $2,000 |

---

## 5. How to Participate (Summary)

1. **Sign in** with Hugging Face account.
2. **Request data access** on dataset page (agree to use rules; WCG IRB protocol).
3. **Choose track(s):** Track 1, Track 2, or both.
4. **Build & submit** via the `Submit — Track 1` / `Submit — Track 2` tabs — any tools/methods allowed.
5. **Join community** on the Space's Community page for Q&A and teaming.

---

## 6. Timeline (Dates subject to change — watch Community page)

| Date | Milestone |
|---|---|
| **Aug 24** | Hackathon launch; dataset available for request/download |
| **Aug 25** | Submissions open |
| **Oct 24 (23:59 UTC)** | Submissions close; Track 1 leaderboard frozen |
| **Oct 24 – Nov 24** | Track 1 qualitative evaluation; Track 2 expert panel judging |
| **Nov 25** | Winners announced; prizes awarded |

Today (per system): **Aug 29, 2026** — hackathon is live.

---

## 7. What Success Looks Like For This Repo

This repository implements a **reproducible, end-to-end variant prioritization pipeline** aimed at Track 1 (with scaffolding reusable for Track 2 mechanistic follow-up):

```
FASTQ → QC/trim → variant calling/filtering → annotation (VEP, gnomAD, CADD, ClinVar)
       → phenotype encoding (HPO) → prioritization → ranking → submission CSV + report
```

Key design principles:
- **Reproducibility:** `configs/`, `workflows/` (Snakemake/Nextflow), pinned env.
- **Modularity:** each `src/` subpackage is independently testable.
- **Evaluation-aware:** ranking optimized for Rank Points + F-max (calibrated scores, not just hard ranks).
- **Scalability:** same pipeline path should work on other rare-disease cases.

See `../README.md` and module READMEs for usage.

---

## 8. Acknowledgements & Ethics

- Deep gratitude to the child and family for sharing their genome openly.
- Organized by **Sage Bionetworks**, **MVA Society**, **Hugging Face**, **BEACON**; guidance from **Wilhelm Foundation**.
- Data shared under explicit consent and IRB oversight. Do not attempt re-identification. Follow dataset Data Use Agreement strictly.
- Reports and code released **CC BY 4.0**.

---

## 9. Quick Links

- **Hackathon Space (source of truth):** https://sagebio-rare-disease-real-kid-mva-hackathon-2026.hf.space/
- **Dataset (request access):** linked from Space → Dataset page
- **Community Forum:** Space → Community tab
- **Submission tabs:** `Submit - Track 1` / `Submit - Track 2` on Space

---

*Document generated 2026-08-29 for `mva-variant-prioritization`. Update if official rules/timeline change.*
