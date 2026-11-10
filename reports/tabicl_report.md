# Benchmark TabICLv2 — modelo fundacional tabular para el ranking de variantes

**Fecha:** 2026-10-31 (actualizado 2026-11-03) · **Rol:** baseline independiente, **no desplegado** · **Modelo primario:** LightGBM `ml_no_excluded` (21 features)

**Pregunta:** ¿un modelo fundacional tabular (TFM) open-source, entrenado in-context sobre las mismas features y el mismo split, supera al ranker LightGBM? **Respuesta corta: no.** TabICLv2 es el mejor TFM completamente abierto disponible y se acerca, pero pierde de forma significativa en NDCG@10, Recall@1 y Rank Points. El primary sigue siendo LightGBM.

---

## 1. Elección del modelo

TabPFN 3.5 está **gated** (requiere aceptar licencia Prior Labs y un `TABPFN_TOKEN`). No había token en el entorno y no se intentó rodear la licencia. Siguiendo la instrucción de usar el mejor modelo fundacional tabular **OSS** disponible en Hugging Face, se eligió **TabICLv2**:

| Criterio | TabICLv2 | TabPFN 3.5 | TabPFN v2 |
|---|---|---|---|
| Licencia | **BSD-3-Clause** (código + pesos) | gated, licencia Prior Labs | pesos abiertos, licencia Prior Labs |
| Acceso | HF `jingang/TabICL`, sin token | requiere `TABPFN_TOKEN` | sin token |
| SHAP nativo | **sí** (`tabicl.shap`, background all-NaN) | permutation | permutation |
| 4 GB GPU | sí (50k filas × 2 estimadores, 3,84 GB pico) | no validado | sí |
| Estado | v2 (feb-2026), ICML 2026 | no ejecutado | baseline secundario |

TabICLv2 se describe como el TFM completamente abierto con mejor rendimiento de su generación y es ~10× más rápido que TabPFN 2.5 en GPU grande ([código](https://github.com/soda-inria/tabicl), [paper](https://huggingface.co/papers/2602.11139)). Es la elección defendible para un benchmark reproducible sin credenciales. **TabPFN v2 se ejecutó solo como smoke test/secundario**, no como sustituto de 3.5.

No se hizo un bake-off contra TabDPT, Mitra o TabFM: la elección se basó en licencia BSD-3, rendimiento reportado en benchmarks abiertos, SHAP nativo y encaje en 4 GB. Es una limitación del benchmark, no una afirmación de que TabICLv2 sea mejor que todos ellos en este dataset.

Dependencias: `tabicl==2.2.0`, checkpoint `tabicl-classifier-v2-20260212.ckpt`, `shap==0.54.1`, CUDA en RTX 2050 (4 GB).

---

## 2. Setup experimental

- **Features:** las **21** del primary desplegable: `FEATURES` menos `pheno_excluded_overlap` y `clinvar_score`, derivadas con `src/ml/rank_features.derive_frame` — paridad train/serve con `prioritize.py`.
- **Split:** el mismo test held-out por gen: 86.412 filas, 1.765 queries, 1.857 causales (`label=3`).
- **Target:** `grade3` (`label >= 3`) binario. TabICL es un clasificador de filas, no un ranker de pares: cada variante se puntúa de forma independiente y luego se ordena dentro de `query_id`.
- **Train:** submuestra **query-level** de 50.000 filas (49.980; 709 queries; 506 positivos grade3), es decir 5,9% de las queries y 6,1% de las filas de train. El límite es la GPU de 4 GB.
- **Modelo (mejor config):** `n_estimators=2`, `kv_cache=True`, `batch_size=1`, `random_state=7`.
- **Coste:** fit 19,8 s; predicción del test completo 35,0 s; pico VRAM 3.835,9 MB.

La comparación es **conservadora para TabICL**: ve 50k filas frente a las 816k de LightGBM, y no puede explotar la estructura de lista (pares intra-query). Aun así pierde.

---

## 3. Métricas en test

| Sistema | NDCG@10 | Recall@1 | Recall@10 | Rank Points | Brier | ECE | F-max |
|---|---:|---:|---:|---:|---:|---:|---:|
| **LightGBM `ml_no_excluded`** (21 feats, desplegado) | **0,9357** | **0,8669** | **0,9960** | **96,97** | 0,0082¹ | — | 0,755 |
| LightGBM primary previo (22 feats, con `pheno_excluded_overlap`) | 0,9381 | 0,8714 | 0,9972 | 97,10 | 0,0075¹ | — | 0,783 |
| **TabICLv2 21feat · 50k · 2 est. · grade3** | **0,9095** | **0,8147** | **0,9960** | **95,47** | **0,0116** | **0,0100** | **0,709** |
| TabPFN v2 22feat · 10k · gt0 (secundario) | 0,7895 | 0,5592 | 0,9955 | 90,29 | 0,0148 | 0,0080 | 0,493 |
| Heuristic ANN (sin ML) | 0,9050 | 0,8510 | 0,9535 | 93,32 | 0,083 | — | 0,521 |

¹ Brier isotónico; el resto de Brier/ECE/F-max son sobre scores raw del clasificador, no directamente comparables con la isotónica de LightGBM.

TabICLv2 recupera el **99,6%** de los causales en top-10 (igual que LightGBM), pero **coloca el causal en el puesto 1 menos veces**: 81,5% vs 86,7%. La mayor pérdida está en la cabeza del ranking, no en la cobertura. En NDCG@10 supera ligeramente al heuristic (0,9095 vs 0,9050), pero pierde en Recall@1 (0,8147 vs 0,8510).

### 3.1 Ablación de target (21 features, 50k filas, 1 estimator)

| Target | NDCG@10 | Recall@1 | Rank Points |
|---|---:|---:|---:|
| `gt0` (label > 0, cualquier evidencia) | 0,8433 | 0,6805 | 92,18 |
| `ge2` (P/LP ≥2★ o causal) | 0,8191 | 0,6249 | 91,24 |
| **`grade3` (solo causal phenopacket)** | **0,8981** | **0,7960** | **94,73** |
| `multiclass` (4 grados, score esperado) | 0,8283 | 0,6431 | 91,77 |

`grade3` gana con claridad. Tiene sentido: el objetivo del ranking es identificar la variante causal del fenopacket, no separar P/LP de benigno. Los targets que mezclan grados intermedios diluyen la señal fenotípica.

### 3.2 Tamaño de train y nº de estimadores (target `grade3`, 21 features)

| Config | NDCG@10 | Recall@1 | Rank Points | VRAM |
|---|---:|---:|---:|---:|
| 50k filas, 1 est. | 0,8981 | 0,7960 | 94,73 | ~2,6 GB |
| **50k filas, 2 est.** | **0,9095** | **0,8147** | **95,47** | **3,84 GB** |
| 60k filas, 1 est. | 0,8985 | 0,7977 | 94,83 | 3,1 GB |
| 70k filas, 1 est. | 0,8905 | 0,7722 | 94,61 | 3,6 GB |

El óptimo validado en 4 GB es **50k / 2 estimadores**: el ensamblado aporta +0,011 NDCG y +0,019 R@1 frente a 1 estimator, a costa de acercarse al límite de VRAM. Más filas no ayudan (70k es peor). No se exploró CPU offload ni checkpoints de mayor capacidad.

### 3.3 Con el set completo de variables (39 features)

Se probó el esquema completo: `FEATURES` (23) + `GENE_INTRINSIC_FEATURES` (16) = **39 features**. Incluye `clinvar_score` y `pheno_excluded_overlap`.

Matiz importante: **`pheno_excluded_overlap` es train-only**; en serving vale siempre 0. Por eso el modelo de 39 features **no es desplegable** y sus métricas de test no son una estimación de serving. Para el set “todo lo disponible en serving” se probó también **37 features** (sin `pheno_excluded_overlap` ni `clinvar_score`).

| Config | Features | Train | Est. | NDCG@10 | Recall@1 | Recall@10 | Rank Points |
|---|---:|---:|---:|---:|---:|---:|---:|
| **TabICL 21 (mejor)** | 21 | 50k | 2 | **0,9095** | **0,8147** | 0,9960 | 95,47 |
| TabICL 21 (matched) | 21 | 40k | 2 | 0,8932 | 0,7881 | 0,9881 | 94,41 |
| TabICL 39 | 39 | 40k | 2 | 0,8633 | 0,7156 | 0,9966 | 93,44 |
| TabICL 39 | 39 | 50k | 1 | 0,8361 | 0,6635 | 0,9875 | 91,70 |
| TabICL 39 | 39 | 60k | 1 | 0,8420 | 0,6765 | 0,9909 | 92,01 |
| TabICL 37 | 37 | 40k | 2 | 0,8093 | 0,6357 | 0,9807 | 89,32 |

En 4 GB, **50k/2est con 39 features hace OOM**; 40k/2est es el máximo que entra. La comparación justa es a 40k/2est:

- **39 features vs 21:** −0,0299 NDCG@10 (IC 95% 0,0198–0,0396 a favor de 21) y −0,0725 Recall@1.
- **37 features vs 21:** −0,0839 NDCG@10 y −0,1524 Recall@1.

El bootstrap pareado está en `reports/tabicl_full_features_bootstrap.json`.

**Candidatos MVA con el set completo:**

| Variante | TabICL 21 (mejor) | TabICL 39 (40k/2est) | TabICL 37 (40k/2est) |
|---|---:|---:|---:|
| ANXA5 | **1** | 2 | 2 |
| BUB1B missense | **3** | **18.101** | 81 |
| BUB1B stop | **87** | **32.587** | 195 |
| MAP2K3 | **29** | 102 | 6.080 |

El modelo de 39 features **colapsa BUB1B**: la missense cae al puesto 18.101 y el stop al 32.587. El de 37 features es menos catastrófico, pero sigue muy lejos del de 21 (missense 81, stop 195).

**Por qué el 39 es especialmente malo:** `pheno_excluded_overlap` entra en el top-3 del SHAP global (0,0416), pero en serving siempre vale 0. El modelo aprende a depender de una feature que no existe en inferencia; el candidate reconstruction la pone a 0 y el modelo queda fuera de distribución. El 37 elimina esa dependencia, pero sigue perdiendo porque las features de gen diluyen la señal fenotípica.

**SHAP global del 39:** `pheno_resnik` 0,0671, `pheno_bma` 0,0416, `pheno_excluded_overlap` 0,0416, `pheno_jaccard` 0,0243, `gene_hpo_count` 0,0206, `gene_path_count` 0,0173. Las features de gen aparecen, pero el ranking empeora.

Conclusión: **TabICL no se beneficia del set completo**. El baseline de 21 features sigue siendo el mejor y el único desplegable. Es coherente con la ablación de LightGBM: las features intrínsecas de gen no aportan en genes held-out.

---

## 4. Bootstrap pareado por query (LightGBM − TabICL)

5.000 remuestreos de las 1.765 queries, mismas filas y mismas etiquetas para ambos modelos:

| Métrica | LightGBM | TabICLv2 | Diferencia | IC 95% | Gana / pierde / empata |
|---|---:|---:|---:|---|---:|
| NDCG@10 | 0,9357 | 0,9095 | **+0,0262** | 0,0189 – 0,0340 | 231 / 111 / 1.423 |
| Recall@1 | 0,8669 | 0,8147 | **+0,0521** | 0,0363 – 0,0686 | 155 / 63 / 1.547 |
| Rank Points | 96,97 | 95,47 | **+1,50** | 1,04 – 1,99 | 227 / 108 / 1.430 |

El IC no cruza 0 en ninguna métrica (`p_diff_le_0 = 0` en las 5.000 réplicas). La ventaja de LightGBM es pequeña en magnitud absoluta pero consistente: en ~81% de las queries ambos empatan (suelen acertar el top-1), y cuando discrepan LightGBM gana ~2× más veces. Artefacto: `reports/tabicl_vs_lightgbm_bootstrap.json`.

---

## 5. Ranking de candidatos MVA (79.797 candidatos)

Mismo VCF, mismos 8 HPO, mismas features. **Los ranks de TabICL son candidate-only** (79.797 candidatos); los de LightGBM son del full-list (5.079.064 variantes), pero solo los candidatos reciben score ML, así que el orden relativo de los candidatos es comparable. Las columnas de rank absoluto no deben compararse 1:1 con el full-list.

| Variante | LightGBM 21 · 8 HPO | LightGBM 21 · 7 HPO | TabICL 21 · 8 HPO (mejor) | TabICL 21 · 8 HPO (60k, 1 est.) |
|---|---:|---:|---:|---:|
| ANXA5 4:121669728 T>TA | **1** | 95.277 | **1** | 2 |
| BUB1B missense 15:40220612 T>G | **7** | **2** | **3** | 5 |
| MAP2K3 17:21300880 C>T | 213 | 224 | 29 | 825 |
| BUB1B stop_gained 15:40209701 T>G | **12** | **5** | **87** | **1.772** |

Lectura:

- **ANXA5 rank 1 se reproduce en ambos modelos.** No es una idiosincrasia de LightGBM: es la señal fuerte de `HP:0200067` (historia parental de abortos) en el bloque fenotípico, que ambos modelos explotan. Sigue siendo un artefacto de indicación, no evidencia de MVA.
- **TabICL captura mejor la missense de BUB1B (rank 3 vs 7)**, pero **pierde el stop_gained patogénico (rank 87 vs 12)**. El stop es la variante con ClinVar P/LP y CADD 36; el TFM, sin `clinvar_score` y con target `grade3`, no la prioriza.
- **MAP2K3** (ClinVar *conflicting*, sin match fenotípico) queda 29 en TabICL y 213 en LightGBM. En ambos está fuera del top-10; TabICL no lo sobrerankea tanto porque no tiene `clinvar_score`.
- Los dos modelos coinciden en el top-1 (ANXA5) y en que BUB1B es el mejor candidato no-ANXA5. La diferencia práctica es que LightGBM pone el stop en el top-15 y TabICL lo deja en el top-100.
- **Con el set completo de variables, TabICL colapsa BUB1B** (missense 18.101, stop 32.587 con 39 features; 81/195 con 37). Ver §3.3.

---

## 6. SHAP

### 6.1 Global

`tabicl.shap` usa un background de **una fila all-NaN** y el masking nativo de NaN de TabICL. Los valores están en **espacio de probabilidad** (P(grade3)), no en raw margin. Background base = **0,0196**. Se explicaron 120 filas de test estratificadas (60 causales / 60 benignas).

| # | Feature | mean \|SHAP\| | Causal | Benigno |
|---:|---|---:|---:|---:|
| 1 | `pheno_bma` | **0,0661** | **0,1229** | 0,0093 |
| 2 | `pheno_resnik` | **0,0407** | **0,0773** | 0,0042 |
| 3 | `pheno_jaccard` | 0,0395 | 0,0703 | 0,0087 |
| 4 | `gene_hpo_count` | 0,0373 | 0,0720 | 0,0026 |
| 5 | `gene_path_count` | 0,0182 | 0,0347 | 0,0016 |
| 6 | `gene_disease_count` | 0,0177 | 0,0309 | 0,0045 |
| 7 | `pheno_terms_matched` | 0,0131 | 0,0236 | 0,0026 |
| 8 | `impact_ord` | 0,0131 | 0,0214 | 0,0047 |
| 9 | `cadd_missing` | 0,0113 | 0,0190 | 0,0035 |
| 10 | `am_norm` | 0,0106 | 0,0198 | 0,0015 |
| 11 | `cadd_norm` | 0,0104 | 0,0179 | 0,0030 |

La historia es la misma que en LightGBM, pero **más concentrada en fenotipo**: las cuatro primeras features son fenotípicas o de tamaño del gen, y `cadd_norm` es un tie-breaker menor (0,0104). La separación causal/benigno es enorme (13–30×) en `pheno_bma`/`pheno_resnik`/`gene_hpo_count`. Sin `clinvar_score`, el TFM no tiene una señal de patogenicidad publicada; toda la evidencia de "esto es causal" pasa por el match fenotípico.

### 6.2 Local — los cuatro candidatos MVA

`base_value = 0,01959`; `pred = P(grade3)`; los SHAP suman `pred − base` (con diferencias <0,005 por batching interno).

| Variante | pred | Base+ΣSHAP | Top contribuciones |
|---|---:|---:|---|
| **BUB1B stop** 15:40209701 T>G | 0,0088 | 0,0088 | `cadd_norm` +0,0130 · `pheno_jaccard` −0,0124 · `gene_disease_count` −0,0123 · `pheno_bma` +0,0115 · `gene_hpo_count` +0,0104 |
| **BUB1B missense** 15:40220612 T>G | 0,0720 | 0,0723 | `gene_hpo_count` +0,0300 · `pheno_jaccard` −0,0261 · `pheno_resnik` +0,0151 · `pheno_bma` +0,0115 · `am_norm` +0,0103 |
| **ANXA5** 4:121669728 T>TA | 0,3477 | 0,3437 | `pheno_bma` +0,2161 · `gene_hpo_count` −0,1232 · `gene_path_count` +0,1089 · `pheno_resnik` +0,0932 · `pheno_jaccard` +0,0578 |
| **MAP2K3** 17:21300880 C>T | 0,0159 | 0,0159 | `pheno_jaccard` −0,0698 · `pheno_bma` −0,0567 · `pheno_resnik` +0,0336 · `am_norm` +0,0294 · `pheno_terms_matched` +0,0186 |

- **BUB1B stop:** el fenotipo empuja (+0,0115 `pheno_bma`, +0,0104 `gene_hpo_count`), pero `pheno_jaccard` y `gene_disease_count` lo compensan y CADD es el único empuje fuerte (+0,0130). Resultado: score bajo y rank 87.
- **BUB1B missense:** mejor explicada. `gene_hpo_count`, `pheno_resnik` y `pheno_bma` suman (+0,057 en conjunto); `pheno_jaccard` resta; AlphaMissense aporta +0,010. Es la variante que TabICL prioriza mejor (rank 3).
- **ANXA5:** `pheno_bma` +0,216 domina; `gene_hpo_count` resta −0,123; `gene_path_count` +0,109 y el resto del bloque fenotípico suman. La señal parental de `HP:0200067` está repartida entre las features fenotípicas, igual que en LightGBM.
- **MAP2K3:** el bloque fenotípico es **negativo** (`pheno_jaccard` −0,070, `pheno_bma` −0,057) y solo CADD/AM lo sostienen. El TFM lo baja a 29; no es un artefacto de ClinVar porque esa feature no existe en este modelo.

---

## 7. Veredicto

1. **No se promueve TabICLv2 a primary.** LightGBM gana en las tres métricas de ranking con IC 95% excluyendo 0, y su inferencia sobre 5M de variantes es mucho más barata.
2. **TabICLv2 es un baseline útil y honesto.** Confirma que la señal principal es fenotípica y que ANXA5 rank 1 no es un artefacto del ranker, sino de la query. También muestra que un TFM genérico no sustituye un ranker entrenado con pares intra-query en este régimen de datos.
3. **El intercambio de errores es informativo.** TabICL rescata mejor la missense (3 vs 7), pero pierde el stop patogénico (87 vs 12). Para un caso clínico, perder la variante P/LP conocida es más grave que perder un VUS; el ranking de LightGBM es preferible.
4. **El set completo de variables tampoco ayuda a TabICL.** Con 39 features baja a 0,8633 NDCG@10 y colapsa BUB1B (missense 18.101, stop 32.587). Con 37 features (sin las dos no desplegables) baja a 0,8093. El baseline de 21 sigue siendo el mejor.
5. **TabPFN 3.5 sigue pendiente, no bloqueante.** Si en el futuro hay `TABPFN_TOKEN`, el mismo harness lo ejecuta con `--model tabpfn --version 3.5`. No se justifica posponer el análisis por ello.

---

## 8. Caveats

- **TabICL no es un learning-to-rank.** Puntúa filas independientemente; no ve la lista ni las etiquetas de las otras variantes de la query. La comparación es justa en features y split, pero no en objetivo.
- **Submuestra de train:** 50k de 816k filas (6,1%), limitada por 4 GB de VRAM. Más datos no mejoraron en las pruebas (70k es peor); 2 estimadores sí.
- **Una sola semilla** (`random_state=7`). La ablación de target/tamaño usa la misma semilla; no se estimó varianza entre semillas. El bootstrap pareado sí cuantifica la incertidumbre sobre queries.
- **Test Phenopacket Store, no pool MVA.** Hay domain shift: el paciente MVA tiene 8 HPO y el 93% de las queries test tienen términos excluidos que serving no recibe.
- **Ranks de TabICL candidate-only.** Los 79.797 candidatos no son las 5.079.064 variantes del full-list; los ranks no son intercambiables con los de `results/ranked_variants.csv`. Los de LightGBM en la tabla sí son full-list.
- **SHAP en probabilidad, background sintético.** El background all-NaN es una referencia de “sin información”, no una distribución de pacientes. Las features fenotípicas están correlacionadas y el reparto de crédito es arbitrario. SHAP explica el modelo, no la biología.
- **El modelo de 39 features no es desplegable.** Incluye `pheno_excluded_overlap`, que en serving vale siempre 0; el propio SHAP lo pone en el top-3 (0,0416). El 37 elimina esa dependencia, pero sigue peor que 21. La comparación de serving es 37, no 39.
- **50k/2est con 39 features hace OOM** en la GPU de 4 GB; 40k/2est es el máximo validado. Las diferencias de 39 entre corridas (~0,001 NDCG) son ruido de batching/seed.
- **El caso MVA sigue sin confirmación clínica:** no hay fase, CNV/CMA ni mosaico. Nada de este benchmark cambia esa limitación.

---

## 9. Reproducción

```bash
# TabICLv2, mejor configuración validada (21 features, 50k, 2 est., SHAP + candidatos)
.venv/bin/python bin/tabpfn_benchmark.py --model tabicl \
  --max-train-rows 50000 --n-estimators 2 --tabicl-batch-size 1 --kv-cache \
  --target grade3 --drop-features clinvar_score,pheno_excluded_overlap \
  --predict-candidates --shap --shap-explain 120 --shap-local \
  --outdir /tmp/opencode/tabicl_v2_21feat_best2

# Set completo de 39 features (no desplegable; 40k/2est es el máximo en 4 GB)
.venv/bin/python bin/tabpfn_benchmark.py --model tabicl \
  --train data/ml/dataset_train_genefeat.csv.gz \
  --val data/ml/dataset_val_genefeat.csv.gz \
  --test data/ml/dataset_test_genefeat.csv.gz \
  --include-gene-features --drop-features "" \
  --max-train-rows 40000 --n-estimators 2 --tabicl-batch-size 1 --kv-cache \
  --target grade3 --predict-candidates --shap --shap-explain 120 --shap-local \
  --outdir /tmp/opencode/tabicl_v2_39feat_40k_2est_full

# Set de 37 features disponibles en serving (sin pheno_excluded_overlap ni clinvar_score)
.venv/bin/python bin/tabpfn_benchmark.py --model tabicl \
  --train data/ml/dataset_train_genefeat.csv.gz \
  --val data/ml/dataset_val_genefeat.csv.gz \
  --test data/ml/dataset_test_genefeat.csv.gz \
  --include-gene-features --drop-features clinvar_score,pheno_excluded_overlap \
  --max-train-rows 40000 --n-estimators 2 --tabicl-batch-size 1 --kv-cache \
  --target grade3 --predict-candidates --shap --shap-explain 120 --shap-local \
  --outdir /tmp/opencode/tabicl_v2_37feat_40k_2est_full

# TabPFN v2 (secundario, 22 features)
.venv/bin/python bin/tabpfn_benchmark.py --model tabpfn --version 2 \
  --max-train-rows 10000 --n-estimators 1 --target gt0 \
  --outdir /tmp/opencode/tabpfn2_full

# TabPFN 3.5 (solo con licencia aceptada y TABPFN_TOKEN)
TABPFN_TOKEN=... .venv/bin/python bin/tabpfn_benchmark.py --model tabpfn \
  --version 3.5 --max-train-rows 50000 --n-estimators 2 \
  --target grade3 --drop-features clinvar_score,pheno_excluded_overlap \
  --outdir /tmp/opencode/tabpfn35
```

**Artefactos:**

| Archivo | Contenido |
|---|---|
| `reports/tabicl_metrics.json` | métricas, ranking MVA y SHAP de la config 21feat/50k/2est |
| `reports/tabicl_ablation.json` | ablación de target, tamaño de train y nº de estimadores (21 y 22 features); TabPFN v2 |
| `reports/tabicl_shap_global.csv` / `.json` | importancia global TabICL (120 filas explicadas) |
| `reports/tabicl_shap_mva.csv` | SHAP local de los 4 candidatos + `base_value` + `pred_proba` |
| `reports/tabicl_vs_lightgbm_bootstrap.json` | bootstrap pareado por query (21 features) |
| `reports/tabicl_full_features.json` | métricas y candidatos MVA del set completo (39/37) |
| `reports/tabicl_full_features_bootstrap.json` | bootstrap 21 vs 39/37 a 40k/2est |
| `reports/tabicl_metrics_39.json`, `reports/tabicl_metrics_37.json` | métricas, candidatos y SHAP del set completo |
| `reports/tabicl_shap_global_39.csv`, `reports/tabicl_shap_global_37.csv` | SHAP global del set completo |
| `reports/tabicl_shap_mva_39.csv`, `reports/tabicl_shap_mva_37.csv` | SHAP local MVA del set completo |
| `/tmp/opencode/tabicl_v2_21feat_best2/ranked_candidates.csv` | ranking TabICL de los 79.797 candidatos |
| `bin/tabpfn_benchmark.py` | harness TabICL/TabPFN (gitignored) |

---

*Generado 2026-10-31 · actualizado 2026-11-03 · TabICLv2 `tabicl==2.2.0` · checkpoint `tabicl-classifier-v2-20260212.ckpt` · RTX 2050 4 GB.*
