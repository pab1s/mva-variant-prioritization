# gnomAD AF (`rare_score`) y candidate generation — resultados experimentales

**Fecha:** 2026-11-03 · **Estado:** experimento completado, **no desplegado** · **Modelo desplegado:** LightGBM 21 features (test original 0,9357)

**Pregunta:** ¿mejora el sistema si añadimos AF real de gnomAD (`rare_score`) y endurecemos la generación de candidatos?

**Respuesta corta:** el AF real es útil para **filtrar candidatos comunes**, pero `rare_score` como feature del ranker **empeora** el test. La candidate generation más estricta reduce artefactos, pero **no mejora los ranks MVA** (siguen 1/7/12) y baja el recall de retrieval grado 3 de 96,3% a 95,0%. Por el criterio acordado, **no se promovió**.

---

## 1. AF real desde la cache de VEP

El pipeline tenía `rare_score` constante 1,0 porque el side-file de AF no existía. Se generó un lookup rápido desde la cache `_var.gz` de VEP:

- `bin/af_cache_lookup.py`: lee solo las ventanas de 1 Mb que contienen nuestras variantes, toma el **máximo AF** del alelo ALT (semántica `MAX_AF`) y cachea cada ventana.
- Cobertura: **318.191 / 5.012.204** variantes del VCF completo (6,3%); en los datasets de train/val/test, **36–39%** de las filas tienen AF conocido. Las variantes sin AF se tratan como raras, igual que antes, pero ahora los comunes conocidos se filtran.
- Artefactos: `data/interim/af.tsv.gz` (todas las variantes del VCF), `data/interim/af_all.tsv.gz`, `data/interim/af_windows/` (cache por ventana).

`rare_score` ahora varía de verdad:

| AF | `rare_score` |
|---|---|
| desconocido o <0,001 | 1,0 |
| <0,01 | 0,9 |
| <0,05 | 0,6 |
| <0,5 | 0,3 |
| ≥0,5 | 0,05 |

---

## 2. `rare_score` como feature del modelo

Se entrenó el primary de 21 features con los datasets enriquecidos con AF:

| Modelo | Features | NDCG@10 | Recall@1 | Rank Points |
|---|---:|---:|---:|---:|
| Baseline original (AF desconocido) | 21 | **0,9357** | **0,8669** | 96,97 |
| Con AF real, incluyendo `rare_score` | 21 | 0,9305 | 0,8555 | 96,69 |
| Con AF real, sin `rare_score` | 20 | **0,9357** | **0,8669** | 96,97 |

**Conclusión:** `rare_score` real **no aporta al ranker** y cuesta ~0,005 NDCG y ~0,011 R@1. El AF sí es útil en la etapa de retrieval, donde filtra variantes comunes antes de que lleguen al ranker. El modelo desplegable recomendado **no usa `rare_score` como feature** (20 features); el AF se usa en candidate generation.

---

## 3. Cambios en candidate generation

Se hicieron tres cambios:

1. **`is_pathogenic_clinsig` corregido.** Antes, `"Conflicting_classifications_of_pathogenicity"` contenía la subcadena `"pathogenic"` y pasaba como P/LP. Ahora se excluyen `conflicting`, `uncertain`, `benign`, `not_provided`, `drug_response`, `risk_factor`, `association`, `protective`, `affects` y `other`. Esto elimina el artefacto MAP2K3.
2. **AF-aware rarity.** La regla `rare = AF desconocido o AF < 0,01` usa el AF real. En el VCF completo se descartaron **22.755 candidatos comunes** que antes pasaban como raros.
3. **Suelo de consecuencia en el gate fenotípico.** El brazo `rare & gene en gate` exige `consequence_weight >= 0,4` para evitar que variantes intrónicas/upstream/downstream de genes con match fenotípico inunden el pool. El brazo ClinVar **no** tiene suelo, para no perder P/LP no codificantes reales.

### Efecto en recall de retrieval

| Regla | Grade 1 | Grade 2 | Grade 3 | Candidatos full VCF |
|---|---:|---:|---:|---:|
| Original (AF desconocido, sin gate) | 98,69% | 98,62% | **96,30%** | 79.797 |
| AF + ClinVar fix, sin suelo de gate | **99,26%** | **99,16%** | **97,33%** | 234.445 |
| AF + ClinVar fix + gate suelo 0,4 | 98,57% | 98,37% | 95,04% | 57.554 |

Lectura:

- **Sin suelo de gate** se gana recall (97,33% en grado 3), pero el gate admite intrónicos/upstream de ANXA5 y el top del full-list se llena de ellos. Es un problema de precisión.
- **Con suelo 0,4** se limpia el top (ANXA5 intrónico desaparece), pero se pierden ~1,3 puntos de recall en grado 3. Es un trade-off real, no un bug.
- El recall de grado 3 con suelo 0,4 (95,04%) sigue siendo alto, pero es inferior al 96,30% original. La decisión depende de si se prioriza precisión del top o recall.

---

## 4. Métricas del ranker sobre el nuevo pool

El pool con suelo 0,4 tiene 57.554 candidatos (vs 79.797 original) y el test tiene 81.860 filas (vs 86.412). Sobre ese pool:

| Modelo | NDCG@10 | Recall@1 | Recall@10 | Rank Points |
|---|---:|---:|---:|---:|
| Modelo original (21 features, sin AF) | **0,9381** | **0,8701** | **0,9960** | **97,12** |
| Modelo AF (21 features, con `rare_score`) | 0,9347 | 0,8638 | 0,9954 | 96,94 |
| Modelo AF sin `rare_score` (20 features) | 0,9346 | 0,8627 | 0,9966 | 96,91 |

El **modelo original sobre el pool limpio** es el mejor: 0,9381 / 0,8701. No usa AF como feature, pero se beneficia de que el pool ya no contiene los comunes ni los intrónicos de ANXA5. El modelo AF no mejora; confirma que `rare_score` no debe ser feature.

Recall@10 end-to-end aproximado para grado 3:

- Original: 96,30% × 99,60% ≈ **95,9%**.
- Pool con suelo 0,4 + modelo original: 95,04% × 99,60% ≈ **94,7%**.
- Pool sin suelo + modelo original: 97,33% × 99,55% ≈ **96,9%**, pero con el top contaminado por ANXA5 intrónico.

El mejor recall end-to-end es el pool sin suelo; el mejor top es el pool con suelo 0,4. Hay que elegir.

---

## 5. Recomendación

1. **Mantener AF en candidate generation**, no como feature del ranker. `rare_score` real no ayuda y el modelo de 20 features (sin `rare_score`) reproduce el baseline.
2. **Mantener la corrección de `is_pathogenic_clinsig`**: elimina el artefacto MAP2K3 y no cuesta recall relevante.
3. **Para el deployment, elegir el gate con suelo 0,4 si se prioriza el top limpio**, o sin suelo si se prioriza recall. Mi recomendación clínica es **suelo 0,4**: un top con ANXA5 intrónico no es aceptable, y el recall de grado 3 sigue en 95%.
4. **No promover el modelo AF con `rare_score`**. El modelo original de 21 features (o su equivalente de 20 sin `rare_score`) sobre el pool limpio es mejor.
5. **No promover la candidate generation experimental**: los ranks MVA no mejoran (1/7/12) y el recall baja. Queda documentada para una futura decisión.

---

## 6. Resultado full-list y decisión

Se hizo un rerun resumible: candidate-only (57.554 candidatos) + merge con el full-list anterior. Resultado:

- Full-list: **5.079.064 filas**, **57.554 candidatos**.
- MVA: **ANXA5 1**, **BUB1B missense 7**, **BUB1B stop 12**. Igual que el deployment anterior.
- **MAP2K3:** 27.348 (deja de ser candidato por ClinVar *conflicting* + AF común).
- Test sobre el pool limpio: **0,9381 / 0,8701 / 0,9960 / 97,12**; retrieval grado 3 **95,0%**.
- El modelo AF de 21 features con `rare_score`: 0,9347 / 0,8638; peor.

**Decisión:** no se promovió. El deployment sigue con el modelo de 21 features y el full-list anterior. Los cambios de AF/candidate generation quedan como experimento reproducible, no como pipeline de producción.

---

## 7. Artefactos

| Archivo | Contenido |
|---|---|
| `bin/af_cache_lookup.py` | lookup AF desde la cache VEP (paralelo, resumible) |
| `data/interim/af_experimental.tsv.gz` | AF de las variantes del VCF (experimental) |
| `data/ml/static_features_af.csv.gz` | static features con AF (experimental) |
| `data/ml/dataset_*_af.csv.gz` | datasets enriquecidos con AF (experimental) |
| `models/ml_no_excluded_af/` | modelo AF de 21 features (experimental) |
| `reports/gene_feature_ablation.md` | ablación de features intrínsecas de gen |
| `data/ml/query_stats.json` | recall de retrieval del deployment actual |

---

*Generado 2026-11-03 · AF desde VEP cache GRCh38 113 · experimento no promovido.*
