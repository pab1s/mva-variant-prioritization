# Plan — Dataset de entrenamiento + candidate generation + ranking (LightGBM)

> **Fecha:** 2026-10-23 · **Objetivo:** entrenar un ranker learning-to-rank (LightGBM LambdaRank)
> que, a partir del VCF + fenotipo (HPO), rankee la priorización de variantes.
> **Decisiones acordadas:** método *riguroso/generalizable* (no atajos al caso MVA) ·
> fuentes = datos locales (ClinVar/HPO/CADD/VEP) + Phenopacket Store ·
> **sin datos del paciente** para entrenar (solo datos públicos).

---

## 1. Estado actual del repo (qué hay y qué falta)

| Pieza | Estado | Archivo |
|---|---|---|
| Anotación VEP del VCF del paciente (5M, CSQ con gnomAD/CADD/ClinVar) | ✅ hecho | `data/interim/full.vep.vcf.gz` |
| Scoring heurístico (fenotipo + patogenicidad) | ✅ hecho | `src/prioritization/prioritize.py` |
| Candidate generation (reglas baratas, alta recall) | ✅ esqueleto | `src/ml/candidates.py` |
| Feature vector del ranker (22 features) | ✅ esqueleto | `src/ml/rank_features.py` |
| CLI de entrenamiento LambdaRank + isotonic | ✅ esqueleto | `src/ml/train_ranker.py` |
| **Dataset de entrenamiento (labels/queries)** | ❌ **falta — este plan** | — |
| **Feature de fenotipo real (Resnik, gene↔HPO)** | ❌ falta (`gene_hpo_map=None` → 0.5 neutro) | `src/phenotype/hpo.py` |
| **Evaluación rigurosa (splits, NDCG, recall, calibración)** | ❌ falta | — |
| Modelo entrenado | ❌ falta (`models/ranker.txt`) | — |

Nota: `train_ranker.py` hoy agrupa por `SYMBOL` (gen) — para este diseño el group debe ser
**query_id** (un caso/enfermedad = una query). Cambio menor incluido en §9.

---

## 2. Investigación: fuentes verificadas (todo comprobado hoy, en local o descargado)

| Fuente | Contenido | Cifras verificadas | Uso |
|---|---|---|---|
| **ClinVar** (local, `data/clinvar/clinvar.vcf.gz`) | Variantes con clasificación clínica | 4,599,326 total · **366,141 P/LP** · 1,415,608 B/LB · 276,150 P/LP con OMIM/Orphanet en `CLNDISDB` · **228,455 P/LP joinables a HPOA** (6,218 enfermedades OMIM) | Labels + hard negatives + `clinvar_score` |
| **HPOA `genes_to_phenotype.txt`** (local, `data/hpo/`) | gen ↔ términos HPO ↔ enfermedad | 333,983 anotaciones · 9,177 enfermedades OMIM | Fenotipo de queries (enfermedad→HPO) y gen→HPO |
| **`hp.obo`** (local) | Ontología HPO | ~173k líneas | Resnik/IC (similitud semántica) |
| **Phenopacket Store 0.1.27** (descargado hoy: `data/phenopackets/all_phenopackets.zip`, 19.4 MB) | Casos reales GA4GH: HPO + diagnóstico + variante causal | **10,377 casos** · 100% con variante causal + HPO + OMIM/Orphanet · **10,810 con `vcfRecord` hg38 (chrom/pos/ref/alt)** · 644 CNV (sin coords, excluibles) · 700 genes, 780 enfermedades | **Test held-out realista + training queries** |
| **VEP cache 113 GRCh38** (local) + **CADD** (local) + plugin ClinVar | Features de anotación | ya usados en `full.vep.vcf.gz` | Paridad total train/serve |
| Referencias metodológicas | Ver §12 | — | Diseño |

**Por qué Phenopacket Store es la joya del plan:** son casos publicados con fenotipo *de paciente
real* (ruidoso, incompleto) y variante causal en coordenadas hg38 → se pueden anotar con el mismo
VEP local y evaluar end-to-end (retrieval + ranking) exactamente como el caso MVA. El paper
(Danis et al., HGG Adv 2025) describe precisamente usarlos spikeando las variantes causales en
VCFs de población para benchmark — nosotros usamos su HPO + variante causal y muestreamos
negativos públicos.

---

## 3. Arquitectura: two-stage retrieve & rank (recsys clásico)

```
                      ┌──────────────────────────────────────────────┐
VCF (5M, VEP) ───────▶│ STAGE 1 — CANDIDATE GENERATION (alta recall) │  ~10⁴-10⁵ filas
HPO del paciente ────▶│  unión de:                                    │  (medimos recall!)
                      │   a) contig primario + AF<1% + consecuencia   │
                      │      protein-altering/splice/UTR             │
                      │   b) ClinVar P/LP                            │
                      │   c) panel MVA (dominio, conocido a priori)  │
                      │   d) PHENOTYPE GATE: genes con Resnik(gen,   │
                      │      HPO_paciente) > 0 (o top-N genes)       │
                      └──────────────────┬───────────────────────────┘
                                         ▼
                      ┌──────────────────────────────────────────────┐
                      │ STAGE 2 — RANKING (LightGBM LambdaRank)      │
                      │  features anotación + fenotipo Resnik        │
                      │  score = w·ML + (1−w)·heurístico             │
                      │  isotonic → calibrado para F-max             │
                      └──────────────────┬───────────────────────────┘
                                         ▼
                            ranked_variants.csv (Rank Points + F-max)
```

El stage 1 es **idéntico en entrenamiento e inferencia**: cada query de entrenamiento pasa por el
mismo filtro, para que el ranker aprenda sobre la distribución real de candidatos. El recall del
retrieval se mide explícitamente (§7): si el filtro descarta la causal, nadie la recupera.

---

## 4. Dataset de entrenamiento

### 4.1 Estructura de una query (group de LambdaRank)

```
query_id | HPO_terms (subsampled) | variante (CHROM,POS,REF,ALT) | label | features…
```

- **query = un caso pseudo-paciente**: sus términos HPO + un pool de variantes candidatas con labels.
- **label ∈ {0,1}** (v1; graded relevance es extensión, §11).
- Tamaño por query: 1–10 positivas + 100–300 negativas.

### 4.2 Dos fuentes de queries

**A) Phenopacket queries (los casos reales) — ~10k queries**
- Positiva: la variante causal (`vcfRecord` hg38, 10,810 casos; 644 CNV se excluyen — fuera de
  scope de un ranker SNV/indel).
- HPO: `phenotypicFeatures` reales del caso (sin excluir `excluded`).
- Anotación: lookup en ClinVar (la mayoría ya están) → si no, VEP con cache local sobre un VCF
  mini de ~10k variantes (minutos).

**B) ClinVar-disease queries (escala) — miles de queries**
- Para cada variante P/LP: `CLNDISDB` → enfermedad OMIM → términos HPOA (228k variantes
  joinables, 6,218 enfermedades).
- HPO de la query = términos de la enfermedad **con subsampling aleatorio (p.ej. 70%)** + muestreo
  de términos irrelevantes de enfermedades vecinas → simula el ruido del fenotipo real (sin esto el
  task es trivial: phenotype_score de la positiva ≈ 1.0 siempre).
- Positivas: P/LP de esa enfermedad (cap ~50/enfermedad). Negativas: ver 4.3.

### 4.3 Negativos (solo fuentes públicas, decisión del usuario)

1. **Fáciles:** ClinVar B/LB + VUS muestreados (rare + protein-altering para imitar el pool
   post-retrieval; una fracción con AF común para que el ranker aprenda `rare_score`).
2. **Hard negatives:** variantes P/LP de *otras* enfermedades cuyo HPO tiene baja similitud con la
   query → fuerza al modelo a usar fenotipo, no solo patogenicidad.
3. **Mismo-gen, otra enfermedad:** P/LP en genes phenotípicamente distantes de la query.

### 4.4 Features (paridad train/serve — regla dura)

**Entran al modelo** (disponibles públicamente y en inferencia, cálculo *idéntico* vía
`row_to_vector`):

| Grupo | Features |
|---|---|
| Rareza | `rare_score` (gnomAD AF vía VEP) |
| Consecuencia | `consequence_weight`, `is_lof`, `is_missense`, `is_splice`, `impact_ord`, `exon_frac` |
| Patogenicidad | `cadd_norm`, `cadd_missing`, `clinvar_score` (CLNSIG + revstat) |
| **Fenotipo (NUEVO)** | `pheno_resnik` (Resnik gen↔query HPO), `pheno_jaccard`, `pheno_terms_matched`, `gene_hpo_count`, `gene_disease_count` |
| Transcripción | `canonical`, `mane_select` (VEP) |

**Quedan FUERA del modelo** (solo existen con genotipo del paciente, sin contraparte pública):
`is_het`, `ab_dev`, `dp_norm`, `quality_score`, `qd_norm`, `mq_norm`, `gene_het_count`,
`compound_het_candidate`, `filt_pass`.
→ Siguen vivos en el **componente heurístico** del blend (0.7 ML + 0.3 heurístico ya cableado en
`prioritize.py`). Se documenta como limitación y se habilita en fase 3 con un VCF público (§11).
`rare_score`, `consequence_weight`, `clinvar_score` etc. ya existen; se **recorta** `FEATURES` a
las entrenables y se añaden las de fenotipo.

### 4.5 Splits y anti-leakage (rigor)

- **Test = phenopackets de genes held-out** (700 genes → ~15% ≈ 100+ genes, ~1.5k casos),
  separados por carpeta de gen (cohortes/PMID del mismo gen quedan juntos).
- **Val** = otros genes held-out (early stopping + calibración isotonic).
- **Exclusión a nivel variante:** ninguna variante que sea causal en test aparece como
  positiva en queries ClinVar de train.
- **Dedup por posición** entre splits (misma variante no en dos splits).
- Las queries ClinVar-disease de enfermedades presentes en test se excluyen de train.

### 4.6 Tamaño esperado

- ~10k (phenopacket) + ~2–5k (ClinVar-disease cap) queries · 100–300 filas/query
  → **0.5–3M filas × ~16 features** (cabe en LightGBM sin problema; parquet intermedio).

---

## 5. Candidate generation (detalle)

Reglas actuales (`src/ml/candidates.py`) + phenotype gate nuevo:

```
is_candidate = primary_contig
             ∧ (gnomAD AF < 0.01 o desconocida)
             ∧ ( FILTER PASS o ClinVar P/LP )
             ∧ ( consequence_weight ≥ 0.6        # protein-altering/splice/UTR
               ∨ gen en panel MVA
               ∨ ClinVar P/LP
               ∨ gen en top-N por Resnik(gen, HPO_query) > umbral )
```

- El **phenotype gate es por unión, nunca por intersección**: un gen sin anotación HPO pero con
  consecuencia severa sigue entrando (protege variantes en genes novos).
- Target: **recall ≥ 99% de las positivas** medido por query en eval (métrica obligatoria, §7).
- En inferencia: stream sobre `full.vep.vcf.gz` → se espera ~10⁴–10⁵ candidatos (vs 5M).

---

## 6. Modelo y entrenamiento

```python
params = {
  "objective": "lambdarank", "metric": "ndcg",
  "ndcg_eval_at": [1, 3, 10],
  "lambdarank_truncation_level": 10,   # alineado con Rank Points (parcial hasta top-10)
  "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 20,
  "feature_fraction": 0.8, "seed": 7, "verbose": -1,
}
# group = query_id (¡cambio en train_ranker: hoy usa SYMBOL!)
# early stopping con valid_set (val queries) — num_boost_round=2000, early_stopping_rounds=50
# label_gain: default (binario) — graded relevance queda como extensión
```

- **Calibración:** isotonic (`IsotonicRegression`) ajustado sobre predicciones de **val** (no train
  — el actual fittea en train, es optimista) → scores calibrados para F-max.
- **Blend:** `score = 0.7·ML_norm + 0.3·heurístico` (ya cableado; `model_weight` en config).
- Salida: `models/ranker.txt` + `models/features.json` + `models/calib.pkl` + `models/metrics.json`.

---

## 7. Evaluación (rigurosa, con ablations)

**Baselines a vencer:**
1. Heurístico actual (score de `prioritize.py` sin ML).
2. Solo `clinvar_score` (upper bound tramposo — mide el leakage de labels).
3. Solo fenotipo Resnik (retrieval puro).

**Métricas (sobre test = phenopackets held-out):**
- **Retrieval recall@pool** (§5) — obligatorio ≥99%.
- **NDCG@1/3/10**, **Recall@1/@10** de la variante causal *dentro del pool* (ranking puro).
- **End-to-end recall@1/@10** (retrieval + ranking sobre el pool completo simulado).
- **Estratificado por status ClinVar de la causal**: conocida (P/LP) vs **novel/VUS** → mide
  generalización real a variantes nuevas (aquí es donde el `clinvar_score` puede hacer trampa).
- **Calibración:** ECE / Brier + F-max (best-F1 sobre thresholds).

**Ablations:**
- A1 features completas · A2 sin `clinvar_score` · A3 sin features de fenotipo ·
  A4 solo heurístico · A5 con/sin blend.

Salida: `reports/ml_ranking_report.md` + `models/metrics.json`.

---

## 8. Inferencia en el caso MVA

1. `full.vep.vcf.gz` (5M) → **stage 1** con el phenotype gate usando los 8 HPO de
   `config/hpo_terms.yaml` (Resnik con `hp.obo` + `genes_to_phenotype.txt`) → pool candidato.
2. Features con `row_to_vector` (FEATURES v2) → `lgb.Booster.predict` → blend → isotonic.
3. `rank.py` → `results/ranked_variants.csv` (submissão Track 1).
4. **Sanity checks:** gen del panel MVA (BUB1B/TRIP13/CEP57) presente en el top de candidatos;
   contigs decoy abajo; top-10 estable.

---

## 9. Archivos a crear/modificar

**Nuevos**
| Archivo | Qué hace |
|---|---|
| `src/phenotype/hpo_ontology.py` | Parser propio de `hp.obo` + IC/Resnik (sin deps nuevos; ~150 líneas) |
| `src/phenotype/gene_hpo.py` | Carga `genes_to_phenotype.txt` → gen→HPO, enfermedad→HPO |
| `src/ml/make_dataset.py` | Construye queries (phenopacket + ClinVar-disease), anota (VEP/CADD/ClinVar/HPO), splits → `data/ml/*.parquet` + `labels.csv` |
| `src/ml/evaluate.py` | Splits, métricas, ablations, calibración → `reports/` |
| `notebooks/03_ml_dataset.py` | Exploración/estadísticas del dataset |

**Modificados**
| Archivo | Cambio |
|---|---|
| `src/ml/rank_features.py` | `FEATURES` v2: −genotipo/calidad, +fenotipo (`pheno_resnik`, …) |
| `src/ml/train_ranker.py` | `--query query_id`, early stopping, `lambdarank_truncation_level`, isotonic sobre val |
| `src/ml/candidates.py` | phenotype gate (unión) |
| `src/prioritization/prioritize.py` | `phenotype_score` real (Resnik) en vez de 0.5 neutro; pasar HPO al retrieval |
| `src/phenotype/hpo.py` | `hpo_overlap_score` → Resnik con fallback Jaccard |

**Datos** (gitignored, `data/`): `data/phenopackets/` (ya descargado) ·
`data/interim/clinvar.vep.vcf.gz` (VEP sobre subset ClinVar) · `data/ml/` (dataset final).

---

## 10. Fases de ejecución

| Fase | Trabajo | Dependencias |
|---|---|---|
| **F0 — Ontología** | `hp.obo` parser + IC/Resnik + gen↔HPO loaders; test: Resnik(BUB1B, HPO MVA) > Resnik(gen aleatorio) | local |
| **F1 — Anotación** | VEP del subset ClinVar P/LP + negativos muestreados + 10,810 variantes phenopacket → `clinvar.vep.vcf.gz`; CADD vía tabix local | VEP cache |
| **F2 — Dataset** | `make_dataset.py`: queries A+B, labels, negativos, features, splits anti-leakage → `data/ml/` + estadísticas | F0, F1 |
| **F3 — Entrenamiento** | `train_ranker` v2 + `evaluate.py`: baselines, métricas, ablations, isotonic | F2 |
| **F4 — Retrieval** | phenotype gate en `candidates.py`; medir recall en test | F0, F2 |
| **F5 — Inferencia MVA** | pipeline completo → `results/ranked_variants.csv` + sanity checks | F3, F4 |
| **F6 — Reporte** | `reports/ml_ranking_report.md` (métricas, ablations, limitaciones) | F3–F5 |

Orden crítico: **F0 → F1 → F2 → F3** (F4 paralelo). F5 al final con modelo congelado.

---

## 11. Riesgos y mitigaciones

| Riesgo | Impacto | Mitigación |
|---|---|---|
| **Leakage de labels vía `clinvar_score`** (labels = ClinVar ⇒ feature = target) | Alto: métricas infladas, no generaliza a variantes novas | Ablación A2 sin `clinvar_score` + estrato novel/VUS en eval + exclusión variante-level train/test |
| **Shift de distribución**: negativos = variantes vistas por ClinVar (enriquecidas en coding), inferencia = todas las raras del VCF | Medio | Muestrear negativos rare + protein-altering (imita post-retrieval) + fracción comunes; documentar; fase 3 con VCF público |
| **Sin features de genotipo en train** (GT/AD/DP/QD) | Medio | Fuera del modelo; viven en el blend heurístico; fase 3 (VCF público con FORMAT) |
| Phenopackets CNV (644) sin coordenadas | Bajo | Excluir del ranker SNV/indel; anotar en el writeup |
| Fenotipo de query trivial (mismo HPO que la enfermedad) | Alto si no se hace | Subsampling + términos ruidosos en queries ClinVar-disease; phenopackets ya tienen HPO real |
| MVA: respuesta held-out; causal podría ser novel o CNV/aneuploidia | Medio | El modelo prioriza SNV/indel; aneuploidia/CNV = fuera de scope (señalar en reporte) |
| Tamaño/tiempo de VEP sobre subset ClinVar | Bajo | Solo subset (P/LP + negativos), `--fork`, plugin CADD local |

---

## 12. Extensiones (fase 3, post-plan)

1. **VCF público de fondo** (1000G/UKB slice) → negativos con distribución real + **features de
   genotipo** (`is_het`, `ab_dev`, `dp_norm`, compound-het) entrenables.
2. **Graded labels** (`label_gain`: causal=2, P/LP otra enfermedad=1, benigno=0) alineado con
   Rank Points.
3. **Teacher-student**: correr Exomiser/LIRICAL sobre los phenopackets y distilar sus ranks
   (los benchmarks citados rinden top-1 ≈ 2/3 de casos).
4. Ensemble con el heurístico aprendido (calibración conjunta) y thresholds F-max por modo
   de herencia (AD/AR/XL desde `allelicState` de los phenopackets).

---

## Referencias (consultadas 2026-10-23)

- Phenopacket Store 0.1.27 — https://github.com/monarch-initiative/phenopacket-store/releases/tag/0.1.27
  (`all_phenopackets.zip`, 10,377 casos). Paper: Danis et al., *HGG Advances* 2025 —
  "A corpus of GA4GH phenopackets" (incluye la técnica de spiking de variantes causales en VCFs
  de población para benchmark).
- Kohler et al., "Phenotype-aware prioritisation of rare Mendelian disease variants" (LIRICAL,
  Exomiser, Xrare, PhenIX; 134 casos; top-1 ≈ 2/3) — https://pmc.ncbi.nlm.nih.nlm/articles/PMC9950798
- genomicsITER benchmark (61 casos WES, HPO + causal) —
  https://github.com/genomicsITER/benchmark-germline-variants-prioritizers
- DAVP (medRxiv 2026): 709 casos (LIRDB + simulados + MyGene2 + UDN) —
  https://www.medrxiv.org/content/10.64898/2026.02.17.26346421
- Exomiser: simulación spiking-in en exomas 1000G —
  https://blog.congenica.com/phenotype-based-prioritization-of-variants-with-exomiser
- LightGBM LambdaRank (`lambdarank_truncation_level`, `label_gain`, `group`) —
  https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html
- HPOA `genes_to_phenotype` + `hp.obo` (JAX/monarch) — locales en `data/hpo/`
- ClinVar (NCBI) — local en `data/clinvar/`
