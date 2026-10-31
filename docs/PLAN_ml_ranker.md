# Plan completo — Dataset de entrenamiento + candidate generation + ranking (LightGBM)

> **Fecha:** 2026-10-24 · **Objetivo:** entrenar un ranker learning-to-rank (LightGBM LambdaRank)
> que, a partir del VCF + fenotipo (HPO), rankee variantes para priorización de enfermedades raras.
>
> **Decisiones acordadas:**
> 1. Método **riguroso/generalizable** (no atajos al caso MVA).
> 2. Fuentes = **datos locales** (ClinVar/HPO/CADD/VEP) + **Phenopacket Store** (ya descargado).
> 3. **Sin datos del paciente** para entrenar — solo datos públicos.
> 4. **Sin blend**: el ranker se evalúa y despliega **standalone**; el heurístico corre como
>    sistema separado únicamente para comparación.
> 5. **Modelo generalizable a variant prioritization**: cero genes/features hardcodeados del MVA
>    dentro del modelo (el panel vive solo en candidate generation, como un panel virtual).

---

## 1. Estado actual del repo (qué hay y qué falta)

| Pieza | Estado | Archivo |
|---|---|---|
| Anotación VEP del VCF del paciente (5M, CSQ con gnomAD/CADD/ClinVar) | ✅ hecho | `data/interim/full.vep.vcf.gz` |
| Scoring heurístico (fenotipo + patogenicidad) | ✅ hecho | `src/prioritization/prioritize.py` |
| Candidate generation (reglas baratas, alta recall) | ✅ esqueleto | `src/ml/candidates.py` |
| Feature vector del ranker (22 features) | ✅ esqueleto | `src/ml/rank_features.py` |
| CLI de entrenamiento LambdaRank + isotonic | ✅ esqueleto | `src/ml/train_ranker.py` |
| Phenopacket Store 0.1.27 (10,377 casos) | ✅ descargado | `data/phenopackets/` |
| **Dataset de entrenamiento (labels/queries)** | ❌ **falta — este plan** | — |
| **Feature de fenotipo real (Resnik, gene↔HPO)** | ❌ falta (`gene_hpo_map=None` → 0.5 neutro) | `src/phenotype/hpo.py` |
| **Evaluación rigurosa (splits, NDCG, recall, calibración)** | ❌ falta | — |
| Modelo entrenado | ❌ falta (`models/ranker.txt`) | — |

Nota: `train_ranker.py` hoy agrupa por `SYMBOL` (gen) — el group debe ser **query_id**
(un caso/enfermedad = una query). Cambio menor incluido en §9.

---

## 2. Investigación: fuentes verificadas (comprobado el 2026-10-23/24)

| Fuente | Contenido | Cifras verificadas | Uso |
|---|---|---|---|
| **ClinVar** (local, `data/clinvar/clinvar.vcf.gz`) | Variantes con clasificación clínica | 4,599,326 total · **366,141 P/LP** · 1,415,608 B/LB · 276,150 P/LP con OMIM/Orphanet en `CLNDISDB` · **228,455 P/LP joinables a HPOA** (6,218 enfermedades OMIM) | Labels + hard negatives + `clinvar_score` |
| **HPOA `genes_to_phenotype.txt`** (local, `data/hpo/`) | gen ↔ términos HPO ↔ enfermedad | 333,983 anotaciones · 9,177 enfermedades OMIM · columna `frequency` | Fenotipo de queries y gen→HPO (Resnik ponderado) |
| **`hp.obo`** (local) | Ontología HPO | ~173k líneas | Resnik/IC (similitud semántica) |
| **Phenopacket Store 0.1.27** (`data/phenopackets/all_phenopackets.zip`, 19.4 MB) | Casos reales GA4GH: HPO + diagnóstico + variante causal | **10,377 casos** · 100% con variante causal + HPO + OMIM/Orphanet · **10,810 con `vcfRecord` hg38 (chrom/pos/ref/alt)** · 644 CNV (sin coords) · 700 genes, 780 enfermedades | **Test held-out realista + training queries** |
| **VEP cache 113 GRCh38** + **CADD** + plugin ClinVar (locales) | Features de anotación | ya usados en `full.vep.vcf.gz` | Paridad total train/serve |
| **Recursos externos verificados para enriquecimiento** (§12) | REVEL 526MB · AlphaMissense ~600MB+tabix · SpliceAI 20GB(raw)/91GB(masked) · LOEUF preprocesado (Ensembl) · VKGL 9MB · 1000G 30x per-chr | ver §12 | F1/F2 (rápido) y F7 (fase 3) |

**Por qué Phenopacket Store es la pieza central:** casos publicados con fenotipo *de paciente
real* (ruidoso, incompleto) y variante causal en coordenadas hg38 → se anotan con el mismo VEP
local y se evalúan end-to-end (retrieval + ranking) exactamente como el caso MVA. El paper
(Danis et al., HGG Adv 2025) describe usarlos spikeando las variantes causales en VCFs de
población para benchmark — nosotros usamos su HPO + variante causal y muestreamos negativos
públicos (y en F7 replicamos el spike-in con 1000G).

---

## 3. Arquitectura: two-stage retrieve & rank (recsys clásico)

```
                      ┌──────────────────────────────────────────────┐
VCF (5M, VEP) ───────▶│ STAGE 1 — CANDIDATE GENERATION (alta recall) │  ~10⁴-10⁵ filas
HPO del paciente ────▶│  unión de:                                    │  (medimos recall!)
                      │   a) contig primario + AF<1% + consecuencia   │
                      │      protein-altering/splice/UTR             │
                      │   b) ClinVar P/LP                            │
                      │   c) panel (opcional, config; MVA en este caso)│
                      │   d) PHENOTYPE GATE: genes con Resnik(gen,   │
                      │      HPO_paciente) > 0 (o top-N genes)       │
                      └──────────────────┬───────────────────────────┘
                                         ▼
                      ┌──────────────────────────────────────────────┐
                      │ STAGE 2 — RANKING (LightGBM LambdaRank)      │
                      │  features anotación + fenotipo Resnik        │
                      │  score = predicción ML STANDALONE (sin blend)│
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
query_id | HPO_terms (augmentadas) | variante (CHROM,POS,REF,ALT) | label (graduado) | features…
```

- **query = un caso pseudo-paciente**: sus términos HPO + un pool de variantes candidatas con labels.
- Tamaño por query: 1–10 positivas + 100–500 negativas.

### 4.2 Fuentes de queries

**A) Phenopacket queries (casos reales) — ~10k queries**
- Positiva: la variante causal (`vcfRecord` hg38, 10,810 casos; 644 CNV excluidas — fuera de
  scope de un ranker SNV/indel).
- HPO: `phenotypicFeatures` reales del caso (+ términos negados `excluded` como señal).
- Anotación: lookup en ClinVar (la mayoría ya están); si no, VEP con cache local sobre un VCF
  mini de ~10k variantes (minutos).
- **Multi-positivas (A6)**: phenopackets con par causal (comp-het) → ambos positivos (estructura
  de media-crádito).

**B) ClinVar-disease queries (escala) — miles de queries**
- Para cada variante P/LP: `CLNDISDB` → enfermedad OMIM → términos HPOA (228k variantes
  joinables, 6,218 enfermedades).
- HPO de la query = términos de la enfermedad **augmentados** (§4.4 A5) → simula el ruido del
  fenotipo real (sin esto el task es trivial).
- Positivas: P/LP de esa enfermedad (cap ~50/enfermedad), con validación de gene-disease
  validity (A2).

**C) Spike-in queries (F7, fase 3)** — 1000G background + variante causal inyectada → queries
con GT real (ver §12.C).

### 4.3 Negativos (solo fuentes públicas)

1. **Confiables por AF (A4):** variantes con gnomAD AF >5% (ACMG BA1/BS1) ≈ benigno seguro.
2. **Fáciles:** ClinVar B/LB muestreados (rare + protein-altering para imitar el pool
   post-retrieval; una fracción con AF común para que el ranker aprenda `rare_score`).
3. **Hard negatives:** variantes P/LP de *otras* enfermedades con baja similitud de fenotipo con
   la query → fuerza al modelo a usar fenotipo, no solo patogenicidad.
4. **Mismo-gen, otra enfermedad:** P/LP en genes phenotípicamente distantes de la query.
5. **+VKGL (A3):** clasificaciones P/B independientes de ClinVar como segunda fuente.
6. **VUS:** excluidos por defecto; si entran, solo como distractores con label 0 y downweight
   (nunca como negativos duros — algunos son causales).

### 4.4 Enriquecimiento de labels y queries (menú A — §12)

| ID | Qué | Dónde encaja |
|---|---|---|
| A1 | **Labels graduados** + `CLNREVSTAT`: causal=2 · P/LP ≥2★=1 · P/LP 1★=0.5 · benigno=0, con `label_gain` en LightGBM | F2 + F3 |
| A2 | **Gene-disease validity (GenCC/ClinGen)**: excluir/downweight positives en genes *limited/disputed/refuted* | F2 |
| A3 | **VKGL consensus** (9MB) como segunda fuente de labels P/B | F1 + F2 |
| A4 | **Negativos por AF** (BA1/BS1) | F2 (local) |
| A5 | **Augmentación de ruido HPO**: subsampling 70% + **expansión a ancestros** + muestreo **ponderado por frecuencia** (HPOA `frequency`) + **términos negados** → feature `pheno_excluded` | F0 (mecanismo) + F2 (aplicación) |
| A6 | **Queries multi-positivas** (pares comp-het de phenopackets) | F2 |

### 4.5 Features (paridad train/serve — regla dura)

**Entran al modelo** (disponibles públicamente y en inferencia, cálculo *idéntico* vía
`row_to_vector`):

| Grupo | Features | Enriquecimiento (menú B) |
|---|---|---|
| Rareza | `rare_score` (gnomAD AF vía VEP) | A4 (AF-based negatives refuerzan) |
| Consecuencia | `consequence_weight`, `is_lof`, `is_missense`, `is_splice`, `impact_ord`, `exon_frac` | B4 SpliceAI → `spliceai_max` |
| Patogenicidad | `cadd_norm`, `cadd_missing`, `clinvar_score` (CLNSIG + revstat) | B2 `revel`, B3 `alphamissense` |
| **Fenotipo (NUEVO)** | `pheno_resnik` (ponderado por frecuencia), `pheno_jaccard`, `pheno_terms_matched`, `pheno_excluded`, `gene_hpo_count`, `gene_disease_count` | A5 + B6 |
| **Gen-génico (NUEVO)** | `gene_loeuf` / `gene_pli` (constraint), `gene_clinvar_path_count` | B1 + B5 |
| Transcripción | `canonical`, `mane_select` (VEP) | — |

**Genéricidad (decisión):** se **eliminan** `is_mva_gene` y el boost MVA hardcodeado de
`phenotype_score` — el modelo no ve listas de genes ni símbolos; solo similitud de fenotipo
gen-agnóstica. El panel MVA (y cualquier panel) vive exclusivamente en candidate generation (§5).

**Quedan FUERA del modelo** (solo existen con genotipo del paciente, sin contraparte pública):
`is_het`, `ab_dev`, `dp_norm`, `quality_score`, `qd_norm`, `mq_norm`, `gene_het_count`,
`compound_het_candidate`, `filt_pass`.
→ **Sin blend** (decisión acordada): el ranker se evalúa y despliega **standalone**. Estas
señales corren únicamente en el **sistema heurístico separado**, que se reporta como baseline
independiente. Se reincorporan como features entrenables en **F7** con un VCF público (§12.C).

### 4.6 Splits y anti-leakage (rigor)

- **Test = phenopackets de genes held-out** (700 genes → ~15% ≈ 100+ genes, ~1.5k casos),
  separados por carpeta de gen (cohortes/PMID del mismo gen quedan juntos).
- **Val** = otros genes held-out (early stopping + calibración isotonic).
- **Exclusión a nivel variante:** ninguna variante que sea causal en test aparece como
  positiva en queries ClinVar de train.
- **Dedup por posición** entre splits (misma variante no en dos splits).
- Queries ClinVar-disease de enfermedades presentes en test → fuera de train.

### 4.7 Tamaño esperado

- ~10k (phenopacket) + ~2–5k (ClinVar-disease cap) queries · 100–500 filas/query
  → **0.5–5M filas × ~18–22 features** (parquet intermedio; LightGBM sin problema).

---

## 5. Candidate generation (detalle)

Reglas actuales (`src/ml/candidates.py`) + phenotype gate nuevo:

```
is_candidate = primary_contig
             ∧ (gnomAD AF < 0.01 o desconocida)
             ∧ ( FILTER PASS o ClinVar P/LP )
             ∧ ( consequence_weight ≥ 0.6        # protein-altering/splice/UTR
               ∨ gen en panel (config, opcional)
               ∨ ClinVar P/LP
               ∨ gen en top-N por Resnik(gen, HPO_query) > umbral )
```

- El **phenotype gate es por unión, nunca por intersección**: un gen sin anotación HPO pero con
  consecuencia severa sigue entrando (protege variantes en genes novos).
- El **panel es solo un prior opcional de dominio** (config; uso general = panel vacío → el gate
  es fenotipo + consecuencia). **Nunca entra al modelo** — así el ranker es genérico y el dominio
  se inyecta de forma transparente y removible en el retrieval.
- Target: **recall ≥ 99% de las positivas** medido por query en eval (métrica obligatoria, §7).
- En inferencia: stream sobre `full.vep.vcf.gz` → ~10⁴–10⁵ candidatos (vs 5M).

---

## 6. Modelo y entrenamiento

```python
params = {
  "objective": "lambdarank", "metric": "ndcg",
  "ndcg_eval_at": [1, 3, 10],
  "lambdarank_truncation_level": 10,   # alineado con Rank Points (parcial hasta top-10)
  "label_gain": "…",                   # A1: graduado (causal > P/LP > benigno)
  "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 20,
  "feature_fraction": 0.8, "seed": 7, "verbose": -1,
}
# group = query_id (¡cambio en train_ranker: hoy usa SYMBOL!)
# early stopping con valid_set (val queries): num_boost_round=2000, early_stopping_rounds=50
```

- **Score final = predicción del ranker STANDALONE** (sin blend con el heurístico — decisión
  acordada). El heurístico se ejecuta como **sistema separado** para la comparación de la §7.
- **Calibración:** isotonic (`IsotonicRegression`) sobre predicciones de **val** (no train — el
  actual fittea en train, es optimista) → scores calibrados para F-max.
- Salida: `models/ranker.txt` + `models/features.json` + `models/calib.pkl` + `models/metrics.json`.

---

## 7. Evaluación (rigurosa, con ablations)

**Baselines (sistemas separados, nunca mezclados):**
1. Heurístico actual completo (score de `prioritize.py` sin ML) — standalone.
2. Solo `clinvar_score` (upper bound tramposo — mide el leakage de labels).
3. Solo fenotipo Resnik (retrieval puro).

**Métricas (sobre test = phenopackets held-out):**
- **Retrieval recall@pool** (§5) — obligatorio ≥99%.
- **NDCG@1/3/10**, **Recall@1/@10** de la variante causal *dentro del pool* (ranking puro).
- **End-to-end recall@1/@10** (retrieval + ranking sobre el pool completo simulado).
- **Estratificado por status ClinVar de la causal**: conocida (P/LP) vs **novel/VUS** → mide
  generalización real a variantes nuevas (aquí es donde `clinvar_score` puede hacer trampa).
- **Desglose por gen/enfermedad del test** → evidencia de generalización (no memorización).
- **Calibración:** ECE / Brier + F-max (best-F1 sobre thresholds).

**Ablations:**
- A1 features completas · A2 sin `clinvar_score` · A3 sin features de fenotipo ·
  A4 sistema heurístico completo (baseline independiente) ·
  A5 *(opcional, post-hoc)* blend ML+heurístico — no es el default.

Salida: `reports/ml_ranking_report.md` + `models/metrics.json`.

---

## 8. Inferencia en el caso MVA

1. `full.vep.vcf.gz` (5M) → **stage 1** con el phenotype gate usando los 8 HPO de
   `config/hpo_terms.yaml` (Resnik con `hp.obo` + `genes_to_phenotype.txt`) → pool candidato.
2. Features con `row_to_vector` (FEATURES v2) → `lgb.Booster.predict` → isotonic →
   **score standalone** (sin blend con el heurístico).
3. `rank.py` → `results/ranked_variants.csv` (submissão Track 1).
4. **Sanity checks:** gen del panel MVA (BUB1B/TRIP13/CEP57) presente en el top de candidatos;
   contigs decoy abajo; top-10 estable.

---

## 9. Archivos a crear/modificar

**Nuevos**
| Archivo | Qué hace |
|---|---|
| `src/phenotype/hpo_ontology.py` | Parser propio de `hp.obo` + IC/Resnik (sin deps nuevos; ~150 líneas) |
| `src/phenotype/gene_hpo.py` | Carga `genes_to_phenotype.txt` → gen→HPO, enfermedad→HPO, frecuencias, MOI |
| `src/ml/make_dataset.py` | Queries A/B, labels graduados, negativos, features, splits → `data/ml/*.parquet` |
| `src/ml/annotate_variants.py` | Anotación en lote de variantes de train (VEP cache + CADD tabix + LOEUF/REVEL/AM) |
| `src/ml/evaluate.py` | Splits, métricas, ablations, calibración → `reports/` |
| `notebooks/03_ml_dataset.py` | Exploración/estadísticas del dataset |

**Modificados**
| Archivo | Cambio |
|---|---|
| `src/ml/rank_features.py` | `FEATURES` v2: −genotipo/calidad, −`is_mva_gene` y boost MVA (genérico), +fenotipo/constraint (`pheno_resnik`, `gene_loeuf`, …) |
| `src/ml/train_ranker.py` | `--query query_id`, early stopping, `lambdarank_truncation_level`, `label_gain` graduado, isotonic sobre val |
| `src/ml/candidates.py` | phenotype gate (unión, config-driven) |
| `src/prioritization/prioritize.py` | `phenotype_score` real (Resnik) en vez de 0.5 neutro; pasar HPO al retrieval; `--model` escribe el score ML **directo** (eliminar blend 0.7/0.3 cableado) |
| `src/phenotype/hpo.py` | `hpo_overlap_score` → Resnik ponderado por frecuencia con fallback Jaccard |

**Datos** (gitignored, `data/`): `data/phenopackets/` (ya descargado) ·
`data/interim/clinvar.vep.vcf.gz` (VEP sobre subset ClinVar) · `data/ml/` (dataset final) ·
`data/annotations/` (LOEUF/REVEL/AlphaMissense/VKGL).

---

## 10. Fases de ejecución (con enriquecimiento mapeado)

| Fase | Trabajo | Incluye | Dependencias |
|---|---|---|---|
| **F0 — Ontología/fenotipo** | `hp.obo` parser + IC/Resnik + loaders gen↔HPO; **Resnik ponderado por frecuencia**; términos negados; test: Resnik(BUB1B, HPO MVA) > Resnik(gen aleatorio) | A5 (mecanismo), B6 | local |
| **F1 — Anotación** | VEP subset ClinVar P/LP + negativos + 10,810 variantes phenopacket → `clinvar.vep.vcf.gz`; CADD vía tabix; **LOEUF (B1), REVEL (B2), AlphaMissense (B3)** vía tabix; gene-level stats (B5); **VKGL (A3)** | A3, B1, B2, B3, B5 | VEP cache |
| **F2 — Dataset** | `make_dataset.py`: queries A+B, **labels graduados (A1)**, **validity GenCC (A2)**, negativos AF (A4) + VKGL + hard, **augmentación HPO (A5)**, **multi-positivas (A6)**, splits anti-leakage → `data/ml/` + estadísticas | A1–A6 | F0, F1 |
| **F3 — Entrenamiento + eval** | `train_ranker` v2 (`label_gain`, truncation, early stop) + `evaluate.py`: baselines, métricas, ablations, isotonic sobre val | — | F2 |
| **F4 — Retrieval** | phenotype gate en `candidates.py`; medir recall en test | — | F0, F2 |
| **F5 — Inferencia MVA** | pipeline completo → `results/ranked_variants.csv` + sanity checks | — | F3, F4 |
| **F6 — Reporte** | `reports/ml_ranking_report.md` (métricas, ablations, limitaciones) | — | F3–F5 |
| **F7 — Fase 3 (fase 3)** | **VCF público 1000G 30x** → negativos realistas + **features de genotipo** (is_het, ab_dev, dp_norm, comp-het con fase) + **spike-in simulation** (queries sintéticas masivas con GT real) · **SpliceAI (B4)** · **hard-negative mining (D1)** · **distilación Exomiser/LIRICAL (D2)** | §12.C, §12.D | F3–F6 |

**Orden crítico:** F0 → F1 → F2 → F3 (F4 en paralelo). F5 con modelo congelado.
**Ruta mínima** (si se necesita una submission rápido): F0 → F1(básico: solo VEP+CADD) →
F2(sin A1/A2/A3) → F3(sin ablations) → F5 → F6(parcial); el resto itera después.

---

## 11. Riesgos y mitigaciones

| Riesgo | Impacto | Mitigación |
|---|---|---|
| **Leakage de labels vía `clinvar_score`** (labels = ClinVar ⇒ feature = target) | Alto: métricas infladas, no generaliza a variantes novas | Ablación A2 sin `clinvar_score` + estrato novel/VUS en eval + exclusión variante-level train/test |
| **Shift de distribución**: negativos = variantes vistas por ClinVar (enriquecidas en coding), inferencia = todas las raras del VCF | Medio | Muestrear negativos rare + protein-altering + fracción comunes (A4); F7 lo resuelve con VCF público |
| **Sin features de genotipo en train** (GT/AD/DP/QD) | Medio | Fuera del modelo; viven solo en el sistema heurístico (baseline separado, sin mezclar); F7 (VCF público con FORMAT) |
| **Score standalone sin señales de genotipo** (mosaicismo `ab_dev`, compound-het) — consecuencia de "sin blend" | Medio (el caso MVA es mosaic) | Aceptado por decisión: heurístico como baseline independiente; F7 para reincorporarlas |
| Fenotipo de query trivial (mismo HPO que la enfermedad) | Alto si no se hace | Augmentación A5 (subsampling + ancestros + frecuencia + negados) |
| Phenopackets CNV (644) sin coordenadas | Bajo | Excluir del ranker SNV/indel; anotar en el writeup |
| MVA: respuesta held-out; causal podría ser novel o CNV/aneuploidia | Medio | El modelo prioriza SNV/indel; aneuploidia/CNV = fuera de scope (señalar en reporte) |
| Tamaño/tiempo de VEP sobre subset ClinVar | Bajo | Solo subset (P/LP + negativos), `--fork`, plugin CADD local |

---

## 12. Menú de enriquecimiento del dataset (verificado 2026-10-23/24)

### A — Labels y queries (quick wins, van en F1/F2)

| ID | Enriquecimiento | Qué aporta | Costo |
|---|---|---|---|
| A1 | Labels graduados + `CLNREVSTAT` | Alinea el loss con Rank Points; penaliza single-submitter dudosos | Local |
| A2 | Gene-disease validity (GenCC/ClinGen) | Evita aprender de asociaciones *limited/disputed/refuted* | TSV pequeño |
| A3 | VKGL consensus (`vkgl_consensus_20260930.tsv`) | +1 fuente independiente P/B → reduce sesgo monocromático | 9MB |
| A4 | Negativos por AF (BA1/BS1: gnomAD AF>5%) | Miles de negativos confiables sin ClinVar | Local |
| A5 | Augmentación ruido HPO (subsampling + ancestros + frecuencia + negados) | Evita task trivial; simula fenotipo real | Local |
| A6 | Queries multi-positivas (pares comp-het) | Enseña estructura de media-crádito AR | Local |

### B — Features del modelo (descargas pequeñas)

| ID | Feature | Fuente verificada | Tamaño |
|---|---|---|---|
| B1 | LOEUF/pLI (constraint génico) | `ftp.ensembl.org/pub/current/variation/LOEUF/loeuf_dataset_grch38.tsv.gz` (tabix-ready, VEP plugin LOEUF) | MBs |
| B2 | REVEL (missense ensemble, ACMG) | `revel_grch38.tsv.gz` + .tbi | ~526MB |
| B3 | AlphaMissense | `AlphaMissense_hg38.tsv.gz` + .tbi (tabix-ready) | ~600MB |
| B4 | SpliceAI (delta scores) | raw SNV 16GB + indel 4GB (Apache 2.0) / masked 27+64GB (académico); anotar **solo candidatos vía tabix** | 20GB+ |
| B5 | Gene-level derivables: nº enfermedades/gen (HPOA), nº variantes P en ClinVar/gen, longitud génica | computable local | 0 |
| B6 | Resnik ponderado por frecuencia (HPOA `frequency`) + MOI (AR/AD) como contexto | local | 0 |

### C — Fondo público con genotipo (el salto grande, F7)

**VCF 1000G high-coverage 30x** (3,202 samples, GRCh38, **phased**, per-chromosome:
[internationalgenome.org](https://www.internationalgenome.org/data-portal/data-collection/1000genomes_30x)):

- ✅ **Negativos con distribución real** (corrige el shift: hoy los negativos son "vistos por ClinVar").
- ✅ **Features de genotipo entrenables**: `is_het`, `gene_het_count`, **compound-het real con
  fase**, `dp_norm` → reintegra al modelo lo perdido al eliminar el blend.
- ✅ **Spike-in simulation estilo Exomiser**: genoma 1000G + variante causal inyectada → queries
  con GT real + causal garantizada → eval controlado y datos masivos (la técnica usada por el
  benchmark de Exomiser y el corpus de phenopackets).
- 💡 No hacen falta los 23 cromos: descarga por cromosoma (GBs c/u) o sites-only + `-s` un
  sample para simular "un paciente".

### D — Augmentación y evaluación (F7)

| ID | Qué | Por qué |
|---|---|---|
| D1 | **Hard-negative mining iterativo**: entrenar v1 → extraer top-ranked wrong en eval → retrain | Ataca la frontera de error |
| D2 | **Distilación Exomiser/LIRICAL** sobre phenopackets (sus ranks como label suave) | Señal de fenotipo que ClinVar no da (benchmark: top-1 ≈ 2/3 de casos) |
| D3 | **Orphanet** (descarga libre: disease→HPO con frecuencia+MOI) para ampliar queries B | Más cobertura de enfermedades |

---

## 13. Extensiones (allá de F7)

1. **Graded labels** completos con `label_gain` calibrado a Rank Points (ya iniciado en A1).
2. **Blend opcional ML+heurístico** (hoy descartado como default por decisión — solo como
   experimento post-hoc A5 de §7) y thresholds F-max por modo de herencia (AD/AR/XL desde
   `allelicState` de los phenopackets).
3. **Ensemble** con distilación de Exomiser/LIRICAL (D2) y calibración conjunta.

---

## Referencias (consultadas 2026-10-23/24)

- Phenopacket Store 0.1.27 — https://github.com/monarch-initiative/phenopacket-store/releases/tag/0.1.27
  (`all_phenopackets.zip`, 10,377 casos). Paper: Danis et al., *HGG Advances* 2025 —
  "A corpus of GA4GH phenopackets" (incluye la técnica de spiking de variantes causales en VCFs
  de población para benchmark).
- Kohler et al., "Phenotype-aware prioritisation of rare Mendelian disease variants" (LIRICAL,
  Exomiser, Xrare, PhenIX; 134 casos; top-1 ≈ 2/3) — https://pmc.ncbi.nlm.nih.gov/articles/PMC9950798
- genomicsITER benchmark (61 casos WES, HPO + causal) —
  https://github.com/genomicsITER/benchmark-germline-variants-prioritizers
- DAVP (medRxiv 2026): 709 casos (LIRDB + simulados + MyGene2 + UDN) —
  https://www.medrxiv.org/content/10.64898/2026.02.17.26346421
- Exomiser: simulación spiking-in en exomas 1000G —
  https://blog.congenica.com/phenotype-based-prioritization-of-variants-with-exomiser
- LightGBM LambdaRank (`lambdarank_truncation_level`, `label_gain`, `group`) —
  https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html
- gnomAD constraint (LOEUF/pLI) — https://gnomad.broadinstitute.org/downloads#v4-constraint ·
  VEP LOEUF plugin — https://raw.githubusercontent.com/ensembl-variation/VEP_plugins/main/LOEUF.pm
- REVEL / AlphaMissense / SpliceAI / VKGL (tamaños y licencias verificados) —
  https://download.molgeniscloud.org/downloads/vip/resources/GRCh38/
- 1000G high-coverage 30x — https://www.internationalgenome.org/data-portal/data-collection/1000genomes_30x
- HPOA `genes_to_phenotype` + `hp.obo` (JAX/monarch) — locales en `data/hpo/`
- ClinVar (NCBI) — local en `data/clinvar/`
