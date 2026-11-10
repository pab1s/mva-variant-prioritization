# CatBoost YetiRank — nuevo modelo primario

**Fecha:** 2026-11-04 · **Estado:** **promovido a primary** · **Features:** 21 (sin `pheno_excluded_overlap`, sin `clinvar_score`) · **Split:** el mismo test held-out (1.765 queries, 1.857 causales)

**Pregunta:** ¿mejora el ranking si sustituimos LightGBM por CatBoost con el mismo esquema de features y el mismo split?

**Respuesta corta: sí.** CatBoost `YetiRank` sube NDCG@10 de **0,9357 a 0,9460** y R@1 de **0,8669 a 0,8850**, con mejor calibración. En el full-list, BUB1B pasa a **rank 2 (stop) / 3 (missense)**; con 7 HPO, **1 / 2**.

---

## 1. Modelo y entrenamiento

- `CatBoostRanker`, `loss_function="YetiRank"`, 400 iteraciones, `lr=0.05`, `depth=6`, `l2_leaf_reg=3.0`, early stopping 50, seed 7.
- Mismo train/val/test y mismas 21 features que `ml_no_excluded`.
- Entrenamiento CPU: **381 s**; mejor iteración **387**.
- Artefactos: `models/ml_catboost/model.cbm`, `features.json`, `calib.pkl` (isotónica sobre val), `metrics.json`, `model_type.txt`.

## 2. Métricas en test

| Modelo | NDCG@10 | Recall@1 | Recall@10 | Rank Points | Brier | F-max |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM (previo) | 0,9357 | 0,8669 | 0,9960 | 96,97 | 0,0082 | 0,755 |
| **CatBoost YetiRank (primary)** | **0,9460** | **0,8850** | **0,9977** | **97,55** | **0,0078** | **0,768** |
| XGBoost `rank:ndcg` | 0,9446 | 0,8850 | 0,9966 | 97,40 | 0,0079 | 0,772 |
| DL MLP ListNet | 0,9519 | 0,9031 | 0,9972 | 97,59 | 0,0092 | 0,764 |

Bootstrap pareado vs LightGBM (5.000 remuestreos):

| Métrica | Diferencia | IC 95% |
|---|---:|---|
| NDCG@10 | **+0,0103** | 0,0059 – 0,0150 |
| Recall@1 | **+0,0181** | 0,0079 – 0,0283 |
| Rank Points | **+0,58** | 0,35 – 0,83 |

CatBoost también mejora la calibración: Brier **0,0078** vs 0,0082, F-max **0,768** vs 0,755.

## 3. Full-list MVA (5.079.064 variantes)

| Variante | CatBoost 8 HPO | CatBoost 7 HPO | LightGBM previo 8 HPO |
|---|---:|---:|---:|
| BUB1B stop 15:40209701 T>G | **2** | **1** | 12 |
| BUB1B missense 15:40220612 T>G | **3** | **2** | 7 |
| ANXA5 4:121669728 T>TA | **1** | 33.260 | 1 |
| MAP2K3 17:21300880 C>T | 16 | 8 | 213 |

- **El locus BUB1B queda en el top-3 con 8 HPO y en el top-2 con 7 HPO.** Es la mejor señal MVA obtenida.
- ANXA5 sigue rank 1 con 8 HPO; al quitar `HP:0200067` cae a 33.260. Es el artefacto parental, no MVA.
- MAP2K3 sube a 16 (8 HPO) / 8 (7 HPO), pero sigue fuera del top-10 y no es MVA.
- Top no-MVA con 8 HPO: ANXA5 1, BUB1B stop 2, BUB1B missense 3, THAP7 4, PDAP1 5, ENPP1 6–10. Los no codificantes ClinVar siguen siendo una limitación de candidate generation.

## 4. SHAP global (CatBoost)

| # | Feature | mean \|SHAP\| | Causal |
|---:|---|---:|---:|
| 1 | `pheno_bma` | **0,733** | **2,256** |
| 2 | `pheno_jaccard` | **0,721** | **1,174** |
| 3 | `cadd_norm` | 0,637 | 0,621 |
| 4 | `pheno_resnik` | 0,264 | 1,107 |
| 5 | `pheno_terms_matched` | 0,228 | 0,180 |
| 6 | `consequence_weight` | 0,200 | 0,081 |
| 7 | `am_norm` | 0,170 | 0,179 |
| 8 | `gene_path_count` | 0,156 | 0,241 |
| 9 | `gene_hpo_count` | 0,093 | 0,174 |
| 10 | `cadd_missing` | 0,090 | 0,071 |
| 11 | `gene_disease_count` | 0,086 | 0,102 |
| 12 | `is_splice` | 0,058 | 0,039 |

El patrón es el mismo que en LightGBM: **fenotipo dominante**, CADD como tie-breaker, consecuencia baja. `pheno_bma` y `pheno_jaccard` son los motores; `cadd_norm` es el tercero.

## 5. SHAP local — MVA

`raw_score = bias + Σ SHAP`, bias ≈ 0 (CatBoost).

**BUB1B stop (raw +1,921; cal 0,0776):**
`cadd_norm` +0,800 · `pheno_bma` +0,646 · `pheno_resnik` +0,358 · `pheno_terms_matched` +0,185 · `gene_hpo_count` +0,083 · `consequence_weight` -0,077.

**BUB1B missense (raw +1,925; cal 0,0776):**
`pheno_bma` +0,674 · `cadd_norm` +0,639 · `pheno_resnik` +0,349 · `am_norm` +0,262 · `pheno_terms_matched` +0,191 · `consequence_weight` -0,088.

**ANXA5 (raw +5,358; cal 0,6688):**
`pheno_bma` +2,960 · `pheno_jaccard` +1,859 · `pheno_resnik` +1,264 · `cadd_norm` -1,186 · `gene_path_count` +0,559.

**MAP2K3 (raw +1,434; cal 0,0430):**
`cadd_norm` +0,931 · `pheno_jaccard` -0,675 · `pheno_terms_matched` +0,522 · `pheno_bma` -0,487 · `am_norm` +0,410.

Lectura: BUB1B se sostiene por **CADD + fenotipo**; ANXA5 por el bloque fenotípico parental; MAP2K3 por CADD/AM con fenotipo negativo. El stop y la missense quedan empatados en calibración (0,0776) y separados por 1 puesto en el full-list.

## 6. Comparación con los otros modelos

- **CatBoost** es el mejor equilibrio ranking/calibración: +0,0103 NDCG, +0,0181 R@1, Brier 0,0078.
- **XGBoost** queda cerca: +0,0088 NDCG, +0,0181 R@1, Brier 0,0079; es más rápido de entrenar (~44 s).
- **DL MLP ListNet** es el mejor ranking (+0,0162 NDCG, +0,0363 R@1), pero calibra peor (Brier 0,0092) y requiere integración PyTorch/SHAP. No se promueve.
- **LightGBM** queda como modelo previo y baseline de comparación; el overlay `ml_full` sigue siendo LightGBM.

## 7. Caveats

- **Un solo test held-out** (Phenopacket Store). No hay validación externa.
- **CatBoost una semilla**; el benchmark DL/XGBoost tiene 2–3 semillas. La mejora de CatBoost es consistente con el bootstrap, pero falta repetir semillas.
- **Mismo pool de candidatos antiguo** (79.797). No se probó con la candidate generation AF experimental.
- **El full-list de 7 HPO se truncó por cuota de disco** al escribir; los ranks MVA del top son válidos (el sort global se hizo antes de escribir), pero no se guardó el archivo completo de 7 HPO.
- **El artefacto ANXA5 persiste**: CatBoost no distingue fenotipo del probando de historia parental.
- **Sin confirmación clínica**: sigue faltando fase, CNV/CMA y estudio de aneuploidía.

## 8. Artefactos

| Archivo | Contenido |
|---|---|
| `models/ml_catboost/model.cbm` | modelo CatBoost YetiRank |
| `models/ml_catboost/features.json`, `calib.pkl`, `metrics.json` | schema, calibrador y métricas |
| `reports/catboost_metrics.json` | métricas val/test |
| `reports/catboost_shap_global.csv/.json/.png` | SHAP global |
| `reports/catboost_shap_mva.csv` | SHAP local de los 4 alelos MVA |
| `reports/catboost_mva_7hpo.csv` | ranks MVA con 7 HPO |
| `reports/catboost_top20_7hpo.csv` | top-20 con 7 HPO |
| `results/ranked_variants.csv` | full-list CatBoost 8 HPO (5.079.064 filas) |

---

*Generado 2026-11-04 · CatBoost 1.2.10 · 21 features · test 1.765 queries · bootstrap 5.000 remuestreos.*
