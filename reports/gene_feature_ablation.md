# Ablación de features intrínsecas del gen — resultado negativo

**Fecha:** 2026-11-03 · **Estado:** experimento completado, **no desplegado** · **Modelo de referencia:** LightGBM `ml_no_excluded` (21 features, test 0,9357 NDCG@10)

**Pregunta:** ¿mejora el ranking añadir variables intrínsecas del gen (constraint, estructura, herencia, especificidad HPO e interacciones gen × consecuencia)?

**Respuesta corta: no.** Se probaron 16 features nuevas, en grupos y una a una. Todas empeoraron el test held-out; la mejor incorporación individual (`gene_num_exons_log`) quedó en −0,0019 NDCG@10 con IC que cruza 0. El modelo desplegado sigue siendo el de 21 features.

---

## 1. Features probadas

### Constraint y estructura (gnomAD v4, MANE/canonical)

| Feature | Descripción |
|---|---|
| `gene_oe_mis` | Observado/esperado de variantes missense. |
| `gene_mis_z` | Z-score de constraint missense. |
| `gene_syn_z` | Z-score de constraint sinónimo. |
| `gene_oe_lof` | Observado/esperado de variantes LoF. |
| `gene_cds_length_log` | `log1p(longitud del CDS)`. |
| `gene_num_exons_log` | `log1p(nº de exones codificantes)`. |
| `gene_segdup_prop` | Fracción del exoma en duplicaciones segmentales. |
| `gene_lcr_prop` | Fracción del exoma en regiones de baja complejidad. |

### Herencia y especificidad fenotípica (HPOA + HPO)

| Feature | Descripción |
|---|---|
| `gene_inheritance_ar` | 1 si el gen tiene anotación de herencia autosómica recesiva. |
| `gene_inheritance_ad` | 1 si tiene herencia autosómica dominante. |
| `gene_inheritance_xl` | 1 si tiene herencia ligada al X. |
| `gene_hpo_ic_mean` | Información media (IC) de los términos HPO del gen. |
| `gene_hpo_ic_max` | IC máximo de los términos HPO del gen. |

### Interacciones gen × variante

| Feature | Descripción |
|---|---|
| `is_lof_x_loeuf` | `is_lof × loeuf_norm`. |
| `is_missense_x_mis_z` | `is_missense × gene_mis_z`. |
| `cadd_x_loeuf` | `cadd_norm × loeuf_norm`. |

Fuentes: `data/annotations/loeuf_dataset_grch38.tsv.gz`, `data/hpo/genes_to_phenotype.txt`, `data/hpo/hp.obo`. Todas son públicas y están disponibles en serving.

---

## 2. Diseño experimental

- Se enriquecieron los datasets existentes (`dataset_{train,val,test}_genefeat.csv.gz`) con las columnas raw de gen, sin re-anotar el VCF.
- Mismo split, mismas queries, mismas etiquetas, mismos genes held-out.
- LightGBM LambdaRank con los hiperparámetros del primary: `lr=0.05`, 63 hojas, `min_data_in_leaf=50`, early stopping 100, seed 7.
- Baseline: 21 features del modelo desplegado (sin `pheno_excluded_overlap`, sin `clinvar_score`).
- Se probaron 7 configuraciones de grupo y 10 de una sola feature.
- Métrica principal: NDCG@10 en test. Se añadió bootstrap pareado por query (5.000 remuestreos) contra el baseline.

---

## 3. Resultados por grupo

| Configuración | Features | NDCG@10 | Recall@1 | Rank Points | ΔNDCG@10 vs baseline [IC 95%] |
|---|---:|---:|---:|---:|---|
| **Baseline** | **21** | **0,9357** | **0,8669** | **96,97** | — |
| + constraint/estructura | 29 | 0,9296 | 0,8533 | 96,65 | **−0,0061** [−0,0098, −0,0025] |
| + herencia/HPO IC | 26 | 0,9296 | 0,8521 | 96,72 | **−0,0061** [−0,0102, −0,0021] |
| + interacciones | 24 | 0,9281 | 0,8538 | 96,44 | **−0,0076** [−0,0113, −0,0041] |
| + constraint + interacciones | 32 | 0,9285 | 0,8499 | 96,65 | **−0,0072** [−0,0110, −0,0034] |
| + herencia + interacciones | 29 | 0,9282 | 0,8510 | 96,57 | **−0,0075** [−0,0121, −0,0033] |
| + todo | 37 | 0,9215 | 0,8357 | 96,24 | **−0,0142** [−0,0189, −0,0096] |

Todos los grupos empeoran de forma significativa. El daño crece al acumular features, lo que apunta a sobreajuste y no a una feature concreta mal calibrada.

---

## 4. Resultados añadiendo una sola feature

| Feature añadida | NDCG@10 | Recall@1 | Rank Points | ΔNDCG@10 [IC 95%] |
|---|---:|---:|---:|---|
| **Baseline (ninguna)** | **0,9357** | **0,8669** | **96,97** | — |
| `gene_num_exons_log` | 0,9338 | 0,8606 | 96,91 | −0,0019 [−0,0056, +0,0016] |
| `gene_inheritance_ar` | 0,9314 | 0,8595 | 96,65 | −0,0043 [−0,0077, −0,0009] |
| `is_missense_x_mis_z` | 0,9309 | 0,8584 | 96,65 | −0,0048 [−0,0082, −0,0015] |
| `gene_oe_mis` | 0,9308 | 0,8578 | 96,69 | −0,0049 [−0,0087, −0,0011] |
| `cadd_x_loeuf` | 0,9308 | 0,8561 | 96,68 | −0,0050 [−0,0088, −0,0013] |
| `gene_oe_lof` | 0,9303 | 0,8584 | 96,56 | −0,0055 [−0,0091, −0,0017] |
| `gene_cds_length_log` | 0,9294 | 0,8538 | 96,61 | −0,0063 [−0,0097, −0,0030] |
| `is_lof_x_loeuf` | 0,9268 | 0,8499 | 96,39 | −0,0089 [−0,0130, −0,0051] |
| `gene_mis_z` | 0,9227 | 0,8425 | 96,15 | −0,0130 [−0,0172, −0,0090] |
| `gene_hpo_ic_mean` | 0,9226 | 0,8414 | 96,29 | −0,0131 [−0,0172, −0,0090] |

Solo `gene_num_exons_log` no empeora de forma significativa, pero tampoco mejora: su IC cruza 0 y el punto estimado es negativo. No hay evidencia para incorporarla.

---

## 5. Por qué falla

1. **Mejora en validación, empeora en test.** El baseline pasa de 0,9172 en val a 0,9357 en test. El modelo con todas las features pasa de 0,9195 en val a 0,9215 en test. Las features nuevas ayudan a ajustar los genes de train/val, pero **no generalizan a los genes held-out del test**. Es el patrón clásico de sobreajuste a priors de gen.
2. **Redundancia con lo que ya hay.** `gene_oe_mis`, `gene_mis_z` y `gene_oe_lof` están correlacionados con `loeuf_norm`, `pli_norm` y `gene_path_count`. `gene_inheritance_*` y `gene_hpo_ic_*` están correlacionados con `gene_hpo_count` y `gene_disease_count`. El modelo ya tenía la mayor parte de esa señal.
3. **Las anotaciones de enfermedad no son neutrales.** Herencia y IC de HPO derivan de asociaciones gen-enfermedad. En un test con genes held-out, pueden favorecer genes conocidos y no transferir a genes novel.
4. **Las interacciones añaden ruido.** El modelo de árboles ya puede aprender combinaciones de `is_lof` con `loeuf_norm`; añadirlas explícitamente aumenta la varianza sin mejorar la generalización.
5. **El modelo es fenotipo-dominante.** `pheno_bma` sigue siendo la feature principal. Las features de gen son de segundo orden y no compensan el ruido que introducen.
6. **SHAP confirma el reweighting.** En el modelo con todo, `gene_cds_length_log` entra en el top-12 (0,076 mean |SHAP|), pero las magnitudes de `pheno_bma` (1,059 vs 0,948), `cadd_norm` (0,607 vs 0,506) y `pheno_jaccard` (0,526 vs 0,389) suben. El modelo se reajusta alrededor de las nuevas features; no añade señal nueva.

### 5.1 TabICLv2 con el set completo

También se pasó TabICLv2 con el set completo. El resultado es coherente: el set completo empeora el TFM.

| Config | Features | Train | Est. | NDCG@10 | Recall@1 |
|---|---:|---:|---:|---:|---:|
| TabICL 21 (mejor) | 21 | 50k | 2 | **0,9095** | **0,8147** |
| TabICL 21 (matched) | 21 | 40k | 2 | 0,8932 | 0,7881 |
| TabICL 39 | 39 | 40k | 2 | 0,8633 | 0,7156 |
| TabICL 37 | 37 | 40k | 2 | 0,8093 | 0,6357 |

El modelo de 39 features incluye `pheno_excluded_overlap`, que es train-only; en serving vale 0. Por eso **no es desplegable** y sus candidatos MVA colapsan: BUB1B missense 18.101, stop 32.587. El de 37 (sin las dos no desplegables) es la comparación de serving: missense 81, stop 195, ANXA5 2. Ambos muy lejos del TabICL de 21 (missense 3, stop 87, ANXA5 1).

Detalle completo en `reports/tabicl_report.md` §3.3 y `reports/tabicl_full_features.json`. La conclusión es la misma que en LightGBM: **las features intrínsecas de gen no ayudan en genes held-out**, y en TabICL además diluyen la señal fenotípica.

---

## 6. Decisión

- **No se añaden al modelo desplegado.** El primary sigue siendo `ml_no_excluded` con 21 features.
- El bloque experimental queda **gated** detrás de `--include-gene-features` en `train_ranker.py`; el esquema por defecto (`FEATURES`) no cambia.
- `prioritize.py` incluye el bloque en la matriz solo si el `features.json` del modelo lo requiere; el primary de 21 features no lo usa.
- Los datasets enriquecidos y los scripts quedan para reproducir el experimento, no para el pipeline de producción.

Si en el futuro se quiere insistir, el camino no es añadir más features de gen, sino:

1. Selección de features con validación anidada por gen.
2. Restricciones monotónicas o un modelo lineal interpretable para el bloque de gen.
3. Un modelo separado de prior de gen, combinado solo si mejora en el estrato novel.
4. Más datos de genes held-out, no más priors de genes conocidos.

---

## 7. Reproducción

```bash
# 1. enriquecer los datasets existentes (no re-anota el VCF)
.venv/bin/python bin/enrich_gene_features.py

# 2. ablación por grupos
.venv/bin/python bin/gene_feature_ablation.py

# 3. prueba de una sola feature
.venv/bin/python bin/gene_feature_single.py
```

**Artefactos:**

| Archivo | Contenido |
|---|---|
| `reports/gene_feature_ablation.json` | métricas, calibración, estratos y bootstrap de los 7 grupos |
| `reports/gene_feature_single.json` | métricas y bootstrap de las 10 features individuales |
| `reports/tabicl_full_features.json` | TabICL con 39/37 features: métricas y candidatos MVA |
| `reports/tabicl_full_features_bootstrap.json` | bootstrap 21 vs 39/37 |
| `data/ml/dataset_{train,val,test}_genefeat.csv.gz` | datasets enriquecidos (experimentales) |
| `bin/enrich_gene_features.py` | enriquecimiento de datasets |
| `bin/gene_feature_ablation.py` | driver de ablación por grupos |
| `bin/gene_feature_single.py` | driver de ablación individual |

---

*Generado 2026-11-03 · LightGBM LambdaRank · test held-out 1.765 queries · bootstrap pareado 5.000 remuestreos.*
