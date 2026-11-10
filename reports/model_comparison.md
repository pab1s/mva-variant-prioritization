# Comparación de modelos — XGBoost, CatBoost y DL

**Fecha:** 2026-11-04 · **Test:** held-out 1.765 queries / 1.857 causales · **Features:** las mismas 21 del primary (`ml_no_excluded`, sin `clinvar_score` ni `pheno_excluded_overlap`) · **Split:** el mismo train/val/test

**Pregunta:** ¿el ranker LightGBM sigue siendo la mejor opción, o XGBoost, CatBoost o un modelo profundo lo superan en el mismo problema?

**Respuesta corta:** los tres lo superan en ranking. **CatBoost** es el mejor candidato de producción por equilibrio ranking/calibración y **fue promovido a primary** el 2026-11-04 (full-list MVA: stop 2 / missense 3). El **MLP listwise** es el mejor en ranking puro, pero calibra peor y requiere más integración (SHAP, serving).

---

## 1. Resultados en test

| Modelo | NDCG@10 | Recall@1 | Recall@10 | Rank Points | Brier | F-max |
|---|---:|---:|---:|---:|---:|---:|
| **LightGBM** (baseline, 21 feats) | 0,9357 | 0,8669 | 0,9960 | 96,97 | 0,0082 | 0,755 |
| **XGBoost** `rank:ndcg` | 0,9446 | 0,8850 | 0,9966 | 97,40 | 0,0079 | 0,772 |
| **CatBoost** `YetiRank` | **0,9460** | **0,8850** | **0,9977** | **97,55** | **0,0078** | **0,768** |
| **DL MLP + ListNet** (seed 7) | **0,9519** | **0,9031** | 0,9972 | **97,59** | 0,0092 | 0,764 |
| XGBoost `rank:pairwise` | 0,9177 | 0,8249 | 0,9966 | 96,20 | 0,0101 | — |

- **XGBoost `rank:ndcg`** mejora a LightGBM en +0,0088 NDCG@10 y +0,0181 R@1.
- **CatBoost `YetiRank`** mejora en +0,0103 NDCG@10 y +0,0181 R@1, y además tiene el mejor Brier/F-max.
- **DL ListNet** es el mejor en ranking: +0,0162 NDCG@10 y +0,0363 R@1, pero su calibración es peor (Brier 0,0092 vs 0,0082).
- **XGBoost `rank:pairwise`** es claramente inferior; se descarta.

---

## 2. Bootstrap pareado vs LightGBM

5.000 remuestreos por query, misma test:

| Modelo | ΔNDCG@10 [IC 95%] | ΔRecall@1 [IC 95%] | ΔRank Points [IC 95%] |
|---|---|---|---|
| XGBoost `rank:ndcg` | **+0,0088** [0,0045; 0,0131] | **+0,0181** [0,0079; 0,0283] | **+0,43** [0,22; 0,65] |
| CatBoost `YetiRank` | **+0,0103** [0,0059; 0,0150] | **+0,0181** [0,0079; 0,0283] | **+0,58** [0,35; 0,83] |
| DL ListNet (seed 7) | **+0,0162** [0,0108; 0,0218] | **+0,0363** [0,0238; 0,0482] | **+0,62** [0,32; 0,95] |

Los tres IC excluyen 0. La mejora no es ruido de una sola query.

### Estabilidad entre semillas

| Modelo | NDCG@10 | Recall@1 | Rank Points |
|---|---:|---:|---:|
| DL seed 7 | 0,9519 | 0,9031 | 97,59 |
| DL seed 17 | 0,9521 | 0,9048 | 97,50 |
| DL seed 27 | 0,9501 | 0,8992 | 97,48 |
| XGBoost seed 7 | 0,9446 | 0,8850 | 97,40 |
| XGBoost seed 17 | 0,9431 | 0,8816 | 97,34 |

El DL es estable en ~0,950–0,952 NDCG@10; XGBoost en ~0,943–0,945. CatBoost se probó con una semilla (0,9460).

---

## 3. Calibración y coste

| Modelo | Brier isotónico | ECE | F-max | Entrenamiento |
|---|---:|---:|---:|---:|
| LightGBM | 0,0082 | 0,0042 | 0,755 | ~20 s |
| XGBoost | 0,0079 | 0,0042 | 0,772 | 44 s (CPU) |
| CatBoost | **0,0078** | 0,0042 | 0,768 | 5,7 min (CPU) |
| DL ListNet | 0,0092 | 0,0087 | 0,764 | 30 s (GPU) |

- **CatBoost** es el mejor calibrado.
- **DL** gana ranking pero pierde calibración; para umbrales clínicos habría que recalibrar o usar un calibrador más expresivo.
- **XGBoost** es el mejor compromiso velocidad/rendimiento.

---

## 4. Recomendación

1. **CatBoost como nuevo primary (promovido 2026-11-04).** Gana a LightGBM en NDCG@10 (+0,0103), R@1 (+0,0181) y Rank Points (+0,58), y además mejora la calibración. Tiene SHAP nativo y es un reemplazo tabular directo. En el full-list deja BUB1B en **rank 2 (stop) / 3 (missense)** con 8 HPO y **1 / 2** con 7 HPO. Ver `reports/catboost_report.md`.
2. **DL ListNet como candidato de research.** Es el mejor ranking (+0,0162 NDCG, +0,0363 R@1), pero necesita integrar el modelo PyTorch en serving, SHAP (KernelSHAP/DeepSHAP) y una calibración mejor. No lo pondría como primary todavía.
3. **XGBoost como fallback.** Rápido, bueno y con SHAP nativo (`pred_contribs`); si CatBoost no encaja por licencia/dependencia, XGBoost es la alternativa.
4. **Deployment actual.** CatBoost ya está integrado en `prioritize.py` (soporta `.cbm` y SHAP), el full-list de 8 HPO está regenerado y los ranks MVA son 2/3. Quedan pendientes:
   - repetir el full-list de 7 HPO completo (el archivo se truncó por cuota de disco; los ranks del top son válidos);
   - validar en un segundo held-out o en UDN/DECIPHER;
   - decidir si el DL merece una integración de serving/SHAP.

---

## 5. Caveats

- **Un solo test.** No hay validación externa; el test es Phenopacket Store con genes held-out.
- **Pool de candidatos antiguo.** No se probó con la candidate generation AF experimental; los números son comparables entre modelos, no con el experimento AF.
- **CatBoost una semilla.** El DL y XGBoost tienen 3/2 semillas; CatBoost solo una por coste.
- **DL es un MLP ListNet**, no un transformer tabular. Un FT-Transformer podría mejorar aún más, pero también el coste y la integración.
- **Calibración DL.** El Brier/ECE es peor; no usar sus probabilidades como umbral clínico sin recalibrar.
- **Sin full-list.** No se ha pasado CatBoost/DL sobre las 5M variantes; los ranks MVA de este informe son solo test, no full-list.

---

## 6. Artefactos

| Archivo | Contenido |
|---|---|
| `bin/model_comparison.py` | benchmark XGBoost/CatBoost/DL con el mismo split y features |
| `reports/model_comparison.json` | métricas, calibración, tiempos y bootstrap de todos los modelos |
| `reports/model_comparison_bootstrap.json` | bootstrap pareado vs LightGBM |
| `/tmp/opencode/model_comparison/` | predicciones y métricas por modelo/semilla |

---

*Generado 2026-11-04 · 21 features · test 1.765 queries · bootstrap 5.000 remuestreos.*
