# Informe final — Ranking ML de priorización de variantes (Track A)

**Fecha:** 2026-10-31 (actualizado 2026-11-04) · Estado: F0→F6 completados · `pytest tests/` 16/16 · `swap.done` presente en `data/interim/.repair/`.

---

## 1. Resultado corto

| Artefacto | Archivo |
|---|---|
| Ranking Track-1 (primary, modelo `ml_catboost`, 21 features) | `results/ranked_variants.csv` (5.079.064 filas; 79.797 candidatos, 1,6%) |
| Overlay clínico (`ml_full`, LightGBM 23 features) | `results/ranked_variants_mlfull.csv` |
| Evaluación en genes held-out (test: 1.765 queries con candidatos) | `reports/eval_report.json` |
| Interpretabilidad SHAP | `reports/catboost_report.md`, `reports/shap_analysis.md` |
| AF y candidate generation (experimental) | `reports/af_candidate_generation.md` |
| Modelos | `models/ml_catboost/` (primary), `models/ranker.txt` (overlay), `models/ml_no_excluded/` (LightGBM previo), `models/ml_no_excluded_af/` (experimental), `models/ml_no_excluded_clinvar/` (alternativa), `models/ml_no_pheno/` |

**Modelo primary elegido: `ml_catboost` (CatBoost YetiRank, 21 features)** — mismas features y split que LightGBM, sin `pheno_excluded_overlap` ni `clinvar_score`. Test held-out: **0,9460 NDCG@10 / 0,8850 R@1 / 97,55 RP**; LightGBM previo: 0,9357 / 0,8669. El bootstrap pareado da **+0,0103 NDCG@10** (IC 95% 0,0059–0,0150) y **+0,0181 R@1** (0,0079–0,0283). En el full-list, BUB1B queda **rank 2 (stop) / 3 (missense)** con 8 HPO y **1 / 2** con 7 HPO. Detalle en `reports/catboost_report.md`.

---

## 2. Evaluación (test held-out: 1.765 queries con candidatos, 1.857 filas causales)

| Sistema | NDCG@10 | Recall@1 | Recall@10 | Rank Points | Brier | F-max |
|---|---:|---:|---:|---:|---:|---:|
| **ml_catboost** (primary, 21 feats) | **0.9460** | **0.8850** | **0.9977** | **97.55** | 0.0078¹ | **0.768** |
| ml_no_excluded (LightGBM previo, 21 feats) | 0.9357 | 0.8669 | 0.9960 | 96.97 | 0.051 | 0.754 |
| ml_no_clinvar (previo, 22 feats, con `pheno_excluded_overlap`) | 0.9381 | 0.8714 | 0.9972 | 97.10 | 0.050 | 0.783 |
| ml_full (overlay LightGBM, 23 feats) | 0.9404 | 0.8771 | 0.9960 | 97.13 | 0.049 | 0.759 |
| heuristic_ann (sistema separado, sin ML) | 0.9050 | 0.8510 | 0.9535 | 93.32 | 0.083 | 0.521 |
| resnik_only (solo fenotipo) | 0.9310 | 0.9184 | 0.9518 | 93.85 | 0.150 | 0.609 |
| clinvar_only (upper bound con leakage) | 0.7435 | 0.7467 | 0.7467 | 74.67 | 0.624 | 0.050 |
| ml_no_pheno (sin fenotipo) | 0.3649 | 0.1405 | 0.6612 | 45.81 | 0.138 | 0.184 |

¹ Brier isotónico de CatBoost; su score raw no es una probabilidad y el Brier raw no es comparable. Con isotónica: CatBoost 0,0078, LightGBM 0,0082 y overlay 0,0080.

Recall@10 estratificado de `ml_catboost`: causal conocida (ClinVar P/LP) **0.9977** (1.318 queries) vs **causal novel** **0.9978** (447) → sin leakage de la anotación ClinVar de variante (la feature no está en el modelo).

*Brier de las filas LightGBM = raw (sin calibrar); F-max raw. La ventaja del overlay no es de calibración, es del prior ClinVar.*

> **SHAP:** el análisis del primary CatBoost está en `reports/catboost_report.md`; el análisis LightGBM previo queda en `reports/shap_analysis.md`. CatBoost confirma el mismo patrón: `pheno_bma` (0,733), `pheno_jaccard` (0,721) y `cadd_norm` (0,637) dominan; la consecuencia pesa poco.

> **AF y candidate generation (experimental):** se probó AF real de gnomAD y una candidate generation más estricta. Como feature del ranker, `rare_score` empeora (0,9305). Con AF + gate floor 0,4 el modelo LightGBM da 0,9381, pero el recall de retrieval grado 3 baja a 95,0% y los ranks MVA no mejoran. **No se promovió.** Detalle en `reports/af_candidate_generation.md`.

---

## 3. Retrieval (candidate generation)

| Label | Kept/Total | Recall |
|---|---:|---:|
| grade 3 (causal phenopacket) | 12,504/12,985 | 96.3% |
| grade 2 (ClinVar P/LP ≥2★) | 80,438/81,564 | 98.6% |
| grade 1 (ClinVar P/LP 1★) | 101,501/102,846 | 98.7% |

Fallos restantes: 204 con gate de solape fenotípico débil y 1,318 sin solape fenotípico. El Recall@10 del causal en test es **0.997** (el ranking full-list no pierde causales ya recuperadas).

---

## 4. Interpretación del caso MVA

### 4.1 Hallazgo principal: BUB1B, no ANXA5

El ranking primary (CatBoost) coloca en el **top-3** del full-list dos variantes de **BUB1B**, el gen principal de MVA1 (OMIM 257300):

| Variante | Proteína | ClinVar | gnomAD v4 | GT | CADD | Rank CatBoost 8 HPO | Rank CatBoost 7 HPO | Rank LightGBM 8 HPO | Rank LightGBM 7 HPO | Rank overlay 8 HPO |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 15:40209701 T>G | p.Leu737Ter | 533901 · P/LP · MVA1 · múltiples submitters, sin conflictos | exoma 7.9e-5; genoma 3.3e-5 | 0/1 | 36.0 | **2** | **1** | 12 | **5** | 15 |
| 15:40220612 T>G | p.Asn1002Lys | no está en ClinVar para ese alelo (el VUS de esa posición es T>A) | exoma 6.8e-7 (1/1.46M); genoma 0 | 0/1 | 24.5 | **3** | **2** | 7 | **2** | 7 |

Todos los ranks son **full-list** (5.079.064 variantes). «CatBoost» = primary actual; «LightGBM» = `ml_no_excluded` previo; «overlay» = `ml_full` de 23 features. No hay ranks candidate-only en esta tabla.

- Ambas son heterocigotas, PASS, con buena profundidad (46 y 28) y balance alélico ~0.5.
- `compound_het_candidate=True` para ambas; `gene_het_count=10` en BUB1B.
- BUB1B tiene `phenotype_score=0.9975` (8 términos) / `0.9991` (7 términos): el motor fenotípico (Resnik/BMA) coincide con el clúster MVA.
- El stop_gained es una **variante patogénica conocida de MVA1** (ClinVar 533901; condición *Mosaic variegated aneuploidy syndrome 1*). La missense es **novel/ultra-rara** (1 alelo en 1.46M exomas), CADD 24.5, en la región C-terminal (dominio quinasa) de BUB1B; p.Leu737Ter truncaría antes de esa región.
- Interpretación: **arquitectura compatible con MVA1 recesivo por heterocigosis compuesta en BUB1B**. No es un diagnóstico: falta fase y la missense es un VUS.
- El primary CatBoost (21 features; NDCG@10 0,9460 / R@1 0,8850) deja el full-list en **stop 2 / missense 3** (8 HPO) y **stop 1 / missense 2** (7 HPO). Es la mejor señal MVA obtenida. LightGBM previo daba 12/7 y 5/2.

### 4.2 El rank 1 (ANXA5) es un artefacto del término parental HP:0200067

- Con los 8 HPO, **ANXA5** es rank 1 (SCORE 0.6688): variante homocigota splice_region/polypyrimidine, CADD 4.0, `phenotype_score=0.701`.
- Al quitar **HP:0200067** (historia parental de abortos recurrentes), ANXA5 cae a rank **33.260** en el full-list de CatBoost (`phenotype_score=0.0015`), y el stop_gained de BUB1B sube a **1** (la missense a **2**). Con LightGBM previo, ANXA5 caía a **95.277** y el stop subía a **5**.
- ANXA5 no es un gen MVA; su asociación es con pérdida gestacional recurrente. El modelo no distingue fenotipo del probando de historia familiar: trata los 8 términos por igual. Este es el principal sesgo de entrada del caso.
- Si el objetivo es el probando MVA, el resultado robusto es **BUB1B**; si se quiere explorar la historia parental, ANXA5 es un hallazgo incidental, no la causa del cuadro del niño.

### 4.3 Los otros genes MVA

| Gen | Evidencia en el paciente | Rank en el full-list (21 feats) |
|---|---|---:|
| TRIP13 | solo intrónicas; no candidatas | 601.826 |
| CEP57 | solo intrónicas/upstream; no candidatas | 1.535.309 |
| BUB1 | 3'UTR homocigota; no candidata | 184.358 |

No hay segundo hit codificante en TRIP13/CEP57/BUB1. Si el fenotipo es MVA, la señal molecular apunta a **BUB1B**.

### 4.4 Top no-MVA: artefactos de la rama ClinVar

En el ranking primary previo, LINC03095 (upstream, LP), THAP7 (downstream, P/LP) y PDAP1 (downstream, P/LP) ocupaban los puestos 2–4: son **variantes no codificantes** con `phenotype_score=0.5` (neutro). Entran por la rama "ClinVar P/LP exenta de rareza" y el ranker les da un score medio por CADD/consecuencia.

Con el primary CatBoost (8 HPO, full-list) el top no-MVA es **THAP7 rank 4**, **PDAP1 rank 5** y **ENPP1 ranks 6–10** (downstream, ClinVar P/LP). MAP2K3 sube a **16**: no es MVA, y su score lo sostienen CADD/AlphaMissense con fenotipo negativo. La rama ClinVar de la generación de candidatos sigue siendo una limitación (THAP7/PDAP1/ENPP1), pero el locus BUB1B queda por delante en el top-3.

### 4.5 Advertencias técnicas

- **ClinVar del VEP es position-overlap, no allele-specific**: en 15:40220612 el paciente tiene T>G, pero ClinVar local tiene T>A (VUS) y T>C (likely benign). El CSQ les asigna `Uncertain_significance&Likely_benign` por solapamiento. El primary CatBoost (21 features) **no usa `clinvar_score`**; el overlay `ml_full` sí. Para la missense, `clinvar_score=0,5` es el valor neutro (VUS), no una llamada P/LP; su rank 7 en el overlay no es evidencia de patogenicidad. El rank honesto del primary CatBoost es **3** (8 HPO) / **2** (7 HPO) para la missense y **2** / **1** para el stop; LightGBM previo daba 7/2 y 12/5.
- **Sin fase**: el VCF no tiene PS y no hay BAM/CRAM; ambas variantes son 0/1. No se puede afirmar que estén en trans. Padres/Sanger o long-read resolverían.
- **Sin CNVs**: el pipeline solo ve small variants; una deleción en BUB1B no se detecta.
- **Sin AF en el modelo** (deferred): la consulta directa a gnomAD v4 confirma que ambas variantes son raras/ultra-raras; la asunción unknown→rare del primary es correcta en este caso.
- **Probabilidades calibradas bajas en el primary CatBoost**: BUB1B stop 0.0776, missense 0.0776, ANXA5 0.6688, MAP2K3 0.0430. En LightGBM previo eran 0.047 (stop), 0.067 (missense), 0.670 (ANXA5) y 0.012 (MAP2K3). El valor útil sigue siendo el rank relativo, no la probabilidad absoluta: el modelo es incierto en este pool y la isotónica no está pensada para un VCF de un solo paciente.

### 4.6 Qué haría falta para cerrar el caso

1. Fase de las dos variantes BUB1B (padres, Sanger/long-read; no hay BAM).
2. CNV/CMA de BUB1B para descartar una deleción como segundo hit.
3. Estudio de aneuploidía mosaic variegated (cariotipo/CMA/FISH en linfocitos/fibroblastos) para apoyo funcional.
4. Clasificación ACMG de p.Asn1002Lys (PM2, PP3, posible PS3 si funcional) y estudio de la proteína/quinasa BUBR1.
5. Si se confirma trans + aneuploidía, el caso es MVA1 por BUB1B; el ranking de CatBoost habrá puesto el locus en **top 1–3 del full-list de 5.079.064 variantes** (stop 2/1 y missense 3/2 con 8/7 HPO) sin saber el diagnóstico.

### 4.7 Lectura SHAP del caso

El análisis completo de CatBoost está en `reports/catboost_report.md`; el de LightGBM previo, en `reports/shap_analysis.md`. Aquí va lo esencial de CatBoost:

- **Global:** `pheno_bma` (mean |SHAP| 0,733; 2,256 en causales), `pheno_jaccard` (0,721; 1,174), `cadd_norm` (0,637), `pheno_resnik` (0,264; 1,107), `pheno_terms_matched` (0,228), `consequence_weight` (0,200). El modelo sigue siendo un matcher fenotípico con CADD como tie-breaker.
- **BUB1B stop_gained:** `cadd_norm` +0,800, `pheno_bma` +0,646, `pheno_resnik` +0,358, `pheno_terms_matched` +0,185, `gene_hpo_count` +0,083, `consequence_weight` -0,077; raw +1,921. El truncamiento no recibe crédito de consecuencia.
- **BUB1B missense:** `pheno_bma` +0,674, `cadd_norm` +0,639, `pheno_resnik` +0,349, `am_norm` +0,262, `pheno_terms_matched` +0,191; raw +1,925. Queda a 1 puesto del stop en el full-list.
- **ANXA5:** `pheno_bma` +2,960, `pheno_jaccard` +1,859, `pheno_resnik` +1,264, `cadd_norm` -1,186, `gene_path_count` +0,559; raw +5,358. Sin `HP:0200067`, ANXA5 cae a 33.260 y BUB1B sube al top-2: el rank 1 es el término parental.
- **MAP2K3 (rank 16):** `cadd_norm` +0,931, `pheno_jaccard` -0,675, `pheno_terms_matched` +0,522, `pheno_bma` -0,487, `am_norm` +0,410. Lo suben CADD/AlphaMissense; el fenotipo MVA es negativo.
- **No codificantes del top:** THAP7, PDAP1 y ENPP1 (downstream, ClinVar P/LP) ocupan los puestos 4–10. Son artefactos de la rama ClinVar/candidate generation, no del prior fenotípico.
- **Overlay:** `clinvar_score` es rank 3 global (0,42), pero su magnitud es mayor en benignos (0,42) que en causales (0,16): sobre todo demota entradas benignas. Para la missense de BUB1B el valor es 0,5 (neutro/VUS), no P/LP; el rank 7 del overlay no es evidencia clínica.
- **Features excluidas:** CatBoost no usa `pheno_excluded_overlap` ni `clinvar_score`. La ablación de `pheno_excluded_overlap` en LightGBM está en `reports/shap_analysis.md` §2.

### 4.8 Baseline con un modelo fundacional tabular: TabICLv2

Se comparó el ranker con **TabICLv2** (BSD-3, el mejor TFM completamente abierto en Hugging Face; TabPFN 3.5 sigue gated), usando las mismas **21 features** y el mismo test:

| Sistema | NDCG@10 | Recall@1 | Rank Points |
|---|---:|---:|---:|
| **CatBoost YetiRank (primary)** | **0,9460** | **0,8850** | **97,55** |
| LightGBM `ml_no_excluded` (previo) | 0,9357 | 0,8669 | 96,97 |
| TabICLv2 21feat · 50k · 2 est. · grade3 | 0,9095 | 0,8147 | 95,47 |

El bootstrap pareado por query da una ventaja de LightGBM sobre TabICL de **+0,0262 NDCG@10** (IC 95% 0,0189–0,0340) y **+0,0521 Recall@1** (0,0363–0,0686). CatBoost, el primary actual, queda **+0,0365 NDCG@10** por encima de TabICL en el mismo test (diferencia puntual; no se repitió el bootstrap contra CatBoost). En candidatos MVA, TabICL reproduce ANXA5 rank 1, **mejora la missense de BUB1B (rank 3 vs 7 en LightGBM)** pero **pierde el stop_gained patogénico (rank 87 vs 12 en LightGBM; CatBoost lo deja en 2)**. Con el set completo de variables (39/37 features) TabICL empeora aún más (0,8633/0,8093 NDCG@10) y colapsa BUB1B; ver `reports/tabicl_report.md` §3.3. No se promueve a primary.

Detalle completo, ablaciones de target/tamaño, SHAP global/local y caveats: **`reports/tabicl_report.md`**.

---

## 5. Infraestructura: incidentes y arreglos

| # | Incidente | Causa raíz | Fix |
|---|---|---|---|
| 1 | 16 procesos VEP muertos; cache con huecos | tarball truncado (1.9GB de 23.9GB); cromos 4-9, X,Y ausentes | `bin/repair_vep_cache.sh` — descarga íntegra + re-extracción y verificación de 24 cromosomas + MT; smoke de `_var` cache 5/5 |
| 2 | systemd-oomd mata el batch (RAM+swap >90%) | VEP `--everything` + `--af_gnomade` retiene ficheros de región en memoria y no los libera (~7GB por proceso en cromosomas grandes) | `bin/annotate_chunked.sh` — pasadas por cromosoma con marcadores; `--fork 2`, `--buffer_size 1000`; `--no-everything` y sin `--af_*` en la pasada estructural |
| 3 | `--custom ClinVar` ~5GB en el padre de VEP | hash in-memory del custom | movido a side-file mediante `load_af`/`cadd_lookup`/`am_lookup` (una pasada tabix) |
| 4 | 1.92M candidatos (38%) | chr4-9/X/Y sin CSQ + "intergenic" ficticia | pasada estructural re-anotada: **79,797 (1.6%)** |
| 5 | clúster ANXA5 intron en el top | el gate admitía variantes de baja consecuencia cuando no hay AF | `af_available=False` — el gate arm se desactiva sin AF (train y serve a la vez) |
| 6 | watcher marcaba FATAL por una pasada anterior | grep del log acumulativo | watcher re-armado con detección de estado actual |

---

## 6. Limitaciones (documentadas)

- **gnomAD AF diferido (fase 3)**: el coste de VEP variation files (~7.6×) sobre esta caja golpeó oomd; una pasada de AF en ventanas de 5Mb se estimó en ~5.4h. Para reanudarla: eliminar `data/interim/.repair/af_deferred.done` y correr de nuevo `bin/repair_vep_cache.sh` (las fases estructurales ya hechas se saltan). Con AF deferido, `af_available=False` y el ranker no ve `rare_score` discriminante; la señal fenotípica (Resnik) queda como la principal (ablation: 0.365 sin fenotipo).
- Features de genotipo sin contraparte pública (GT/AD/DP, QD/MQ, comp-het) viven en el sistema heurístico separado (sin blend, decisión del usuario). Fase 3: fondo 1000G (3,202 genomas con FORMAT) para entrenar con genotipo.
- Panel de genes por HPO top-100 (Resnik), no un panel MVA fijo. El ranker es genérico; el panel es un prior opcional en `config/example_config.yaml` (`panel_source: hpo`).
- Calibración isotonic ajustada en val; las probabilidades absolutas en el pool del paciente son bajas (dominio shift). Para submission, usar el rank; para umbrales, re-calibrar.
- **`pheno_excluded_overlap` es train-only**: se calcula con los fenotipos excluidos de Phenopacket Store, pero `prioritize.py` no recibe términos negativos y en serving vale siempre 0. El 0,938 del test está inflado por esta feature; el modelo desplegado, reentrenado sin ella y sin `clinvar_score`, da NDCG@10 0,9357 / R@1 0,8669. Ver `reports/shap_analysis.md` §2.
- **Features muertas:** `rare_score` (constante 1,0 por AF diferido), `canonical` y `mane` (constantes en el VEP local) tienen 0 splits y 0 SHAP. Eliminarlas o arreglarlas antes de un nuevo entrenamiento.

---

## 7. Deliverables

- **Submission Track-1:** `results/ranked_variants.csv` (primary `ml_catboost` de 21 features; 5.079.064 filas; 79.797 candidatos; RANK 1..N, SCORE calibrado, columnas de auditoría `ml_score`, `is_candidate`, `Consequence`, `CADD_PHRED`, `AD_REF/AD_ALT`, `gene_het_count`).
- **Overlay clínico:** `results/ranked_variants_mlfull.csv` (mismo formato, `ml_full` LightGBM de 23 features; útil si el causal es una variante publicada en ClinVar, con la cautela del match alélico).
- **Modelos:** `models/ml_catboost/` (primary CatBoost), `models/ranker.txt` (overlay LightGBM), `models/ml_no_excluded/` (LightGBM previo), `models/ml_no_excluded_af/` (experimental), `models/ml_no_excluded_clinvar/` (alternativa 22 features con ClinVar, no desplegada), `models/ml_no_pheno/`.
- **Interpretabilidad:** `reports/catboost_report.md`, `reports/catboost_shap_global.csv`, `reports/catboost_shap_mva.csv`, `reports/shap_analysis.md` (LightGBM previo), `results/shap_mva_overlay.csv`.
- **AF / candidate generation (experimental):** `reports/af_candidate_generation.md`, `data/interim/af_experimental.tsv.gz`, `bin/af_cache_lookup.py`, `bin/merge_candidate_ranking.py`.
- **Comparación de modelos:** `reports/model_comparison.md`, `reports/model_comparison.json`, `bin/model_comparison.py`.
- **Baseline TFM:** `reports/tabicl_report.md`, `reports/tabicl_metrics.json`, `reports/tabicl_ablation.json`, `reports/tabicl_shap_global.csv`, `reports/tabicl_shap_mva.csv`, `reports/tabicl_vs_lightgbm_bootstrap.json`.
- **Reproducibilidad:** `data/interim/.repair/` con marcadores re-entrantes; `bin/repair_vep_cache.sh`, `bin/run_ml_pipeline.sh`.

---

## 8. Reproducción del pipeline

```bash
./bin/repair_vep_cache.sh          # verifica cache + re-anota train/paciente (resumable)
./bin/run_ml_pipeline.sh           # build → queries → train → evaluate → inference ×2 → sanity
pytest tests/                      # 16/16 verificación
```

Para la sensibilidad sin el término parental:

```bash
.venv/bin/python bin/mva_sensitivity.py
```

Para el análisis SHAP global y las figuras:

```bash
.venv/bin/python bin/shap_analysis.py   # reports/shap_global_*.csv/.json/.png
.venv/bin/python bin/shap_plots.py      # shap_global_strata.png + shap_mva_local.png
```

Para la inferencia del primary desplegable sobre el VCF completo:

```bash
.venv/bin/python -m src.prioritization.prioritize \
  --vcf data/interim/full.vep.vcf.gz --hpo config/hpo_terms.yaml \
  --config config/example_config.yaml \
  --model models/ml_no_excluded/ranker.txt -o results/ranked_variants.csv
```

Para el baseline TabICLv2 (21 features, 50k, 2 est., GPU):

```bash
.venv/bin/python bin/tabpfn_benchmark.py --model tabicl \
  --max-train-rows 50000 --n-estimators 2 --tabicl-batch-size 1 --kv-cache \
  --target grade3 --drop-features clinvar_score,pheno_excluded_overlap \
  --predict-candidates --shap --shap-explain 120 --shap-local \
  --outdir /tmp/opencode/tabicl_v2_21feat_best2
```

---

## 9. Fase 3 (pendiente)

1. gnomAD AF windowed (5Mb; ya scripteado: `bin/af_parallel.sh` + `annotate_af_windowed.sh`): quitar `af_deferred.done` y re-ejecutar `repair_vep_cache.sh`.
2. Fondo 1000G (3,202 genomas) para re-entrenar con genotipo (GT/AD/DP/quality).
3. Ensamble con Exomiser/LIRICAL como teacher.
4. Ampliar el pool de train (UDN, DECIPHER; documentos clínicos → HPO completo).
5. Features intrínsecas de gen: se probaron 16 (constraint gnomAD, estructura, herencia HPOA, IC e interacciones). Todas empeoraron el test held-out; no se incorporan. Detalle y ablación en `reports/gene_feature_ablation.md`.
6. AF real y candidate generation (experimental): el AF de gnomAD se probó para filtrar candidatos, no como feature del ranker. La corrección de `is_pathogenic_clinsig` elimina el artefacto MAP2K3. Con el pool limpio (gate floor 0,4) el ranker sube a 0,9381, pero el recall de retrieval grado 3 baja a 95,0% y los ranks MVA no mejoran (1/7/12). **No se promovió**; el trade-off está en `reports/af_candidate_generation.md`.
7. Modelos alternativos (2026-11-04): XGBoost `rank:ndcg` (0,9446), CatBoost `YetiRank` (0,9460) y un MLP listwise (0,9519) superan a LightGBM (0,9357). **CatBoost se promovió a primary**; full-list MVA stop 2 / missense 3 (8 HPO) y 1/2 (7 HPO). El DL es el mejor ranking pero calibra peor y requiere integración. Detalle en `reports/catboost_report.md` y `reports/model_comparison.md`.

---

*Generado 2026-10-31 · actualizado 2026-11-04 · Artefactos clave: `models/ml_catboost/metrics.json`, `reports/eval_report.json`, `reports/catboost_report.md`, `data/interim/.repair/repair.log`.*
