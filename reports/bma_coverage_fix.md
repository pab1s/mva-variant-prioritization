# Corrección de `pheno_bma` — normalización con cobertura

**Fecha:** 2026-11-21 · **Estado:** código corregido, modelo reentrenado como experimental, **no promovido** · **Test:** mismo held-out (1.765 queries)

**Cambio:** `resnik_bma` ahora divide por el peso de **todos** los términos de la query, no solo por los que matchean. Un gen que acierta un único término específico ya no puede sacar 1,0.

```python
# antes: solo términos con match
wsum = sum(w for q in matched_terms)
bma = sum(w * best[q] for q in matched_terms) / wsum

# ahora: todos los términos; los no matcheados suman 0
wsum = sum(w for q in query_terms)
bma = sum(w * best[q] for q in query_terms) / wsum
```

Implementado en `src/phenotype/hpo_ontology.py:similarity()` y cubierto por `tests/test_phenotype.py::test_bma_is_coverage_aware`.

## Efecto en MVA

| Caso | Modelo anterior | Modelo con BMA corregido |
|---|---|---|
| **8 HPO** (incluye `HP:0200067` parental) | ANXA5 #1 · BUB1B #2/#3 | ANXA5 #5 · BUB1B **#68** |
| **7 HPO** (sin `HP:0200067`) | BUB1B #1/#2 · ANXA5 #33.260 | **BUB1B #1/#2** · ANXA5 #31.273 |

Ranks entre los 79.797 candidatos (los no candidatos quedan por debajo, así que equivalen al full-list).

**Lectura:** la corrección baja el BMA de ANXA5 de 1,000 a 0,125, pero también el de BUB1B de 0,347 a 0,303. El modelo reentrenado aprende un coeficiente distinto para la nueva distribución y, con 8 HPO, BUB1B cae a #68. El siguiente sesgo es `pheno_jaccard`, que sigue normalizando por la unión con todo el set de anotaciones del gen: ANXA5 recibe +1,57 de SHAP y BUB1B −0,23. Sin corregir Jaccard, el BMA correcto solo no alcanza.

Con 7 HPO (fenotipo del probando, sin el término parental) el resultado es el mejor obtenido: **BUB1B #1 (missense) y #2 (stop)**, ANXA5 fuera del top-30k. El BMA corregido no es el que arregla MVA por sí solo; el arreglo práctico es separar historia parental de fenotipo del probando.

## Métricas en test

El dataset canónico se regeneró con `python -m src.ml.make_dataset queries`: mismas 816.123/73.595/86.412 filas y mismos positivos. Solo cambian 54.424 filas de train (6,7%); val/test no cambian porque sus queries ya cubrían todos los términos. Por eso el test no puede medir la mejora del feature; mide el reentrenamiento.

| Modelo | NDCG@10 | Recall@1 | Rank Points | Brier | F-max |
|---|---:|---:|---:|---:|---:|
| CatBoost anterior | **0,9460** | **0,8850** | **97,55** | 0,0078 | **0,768** |
| CatBoost BMA corregido | 0,9441 | 0,8822 | 97,42 | 0,0079 | 0,766 |

Bootstrap pareado por query contra el anterior:

| Métrica | Δ | IC 95% |
|---|---:|---|
| NDCG@10 | −0,0019 | [−0,0042; +0,0004] |
| Recall@1 | −0,0028 | [−0,0091; +0,0034] |
| Rank Points | −0,1246 | [−0,2153; −0,0397] |

La diferencia en NDCG/R@1 no es significativa; Rank Points baja 0,12 (significativo pero pequeño). El cambio no se promueve por el empeoramiento en el 8-HPO, no por el test.

## Decisión y siguiente paso

1. **Código:** se queda la corrección (es la definición correcta de BMA) y el test.
2. **Modelo:** `models/ml_catboost_bmafix/` es experimental y **no se despliega**. El primary `models/ml_catboost/` sigue siendo el anterior; hay que reentrenarlo antes de usar el código corregido en serving, o el modelo verá una distribución de `pheno_bma` distinta a la de entrenamiento.
3. **Siguiente bloqueador:** `pheno_jaccard` con denominador `|Q ∪ G|`. Para MVA, ANXA5 tiene 2 términos anotados y BUB1B 110, así que el Jaccard premia al gen poco anotado. La corrección coherente es normalizar por la query (`|Q ∩ G| / |Q|`) o eliminar Jaccard y dejar que `pheno_terms_matched` aporte la cobertura.
4. **Input:** para el probando, usar los 7 HPO y tratar `HP:0200067` como historia familiar aparte. Con eso, incluso el modelo con BMA corregido deja BUB1B en top-2.

## Reproducir

```bash
# 1. Código corregido + test
.venv/bin/python -m pytest tests/test_phenotype.py -q

# 2. Regenerar datasets con el pipeline canónico (~45 min)
.venv/bin/python -m src.ml.make_dataset queries

# 3. Reentrenar el primary experimental
.venv/bin/python -m src.ml.train_esm_ranker --backend catboost --esm none \
  --tag catboost_bmafix --compare-base catboost_base \
  --save-dir models/ml_catboost_bmafix

# 4. Puntuar candidatos MVA (rápido: solo el pool de 79.797)
.venv/bin/python -m src.prioritization.prioritize \
  --vcf /tmp/opencode/mva_candidates.vep.vcf.gz \
  --hpo /tmp/opencode/hpo_terms_7.yaml \
  --config config/example_config.yaml \
  --model models/ml_catboost_bmafix/model.cbm \
  --candidates-out /tmp/opencode/mva_bmafix_7hpo_candidates.csv \
  -o /tmp/opencode/mva_bmafix_7hpo_ranked.csv
```
