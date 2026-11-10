# Embeddings ESM-2 en el ranker — experimento

**Fecha:** 2026-11-10 · **Test:** held-out 1.765 queries / gene-disjoint · **Features base:** las 21 del primary (`ml_catboost`, sin `clinvar_score` ni `pheno_excluded_overlap`) · **Split:** el mismo train/val/test

**Pregunta:** ¿aportan las representaciones de un modelo fundacional de proteínas sobre las 21 features tabulares del primary?

**Respuesta corta:** no en CatBoost. El bloque ESM-2 mejora de forma significativa a LightGBM (+0,0055 NDCG@10), pero CatBoost —el primary desplegado— ya captura esa señal: +0,0009 NDCG@10 con IC 95% que cruza 0 y sólo 1,1% de importancia. **No se promueve.** El primary sigue siendo CatBoost con 21 features.

---

## 1. Qué modelo fundacional y por qué

AlphaMissense no publica embeddings: sólo un score escalar de patogenicidad, que ya está en el esquema como `am_norm`. Para usar representaciones reales se eligió **ESM-2** (`facebook/esm2_t12_35M_UR50D`, 480 dimensiones), el modelo de lenguaje de proteínas estándar para variantes missense.

Por variante missense se extrajeron dos bloques:

| Bloque | Definición | Dims |
|---|---|---:|
| `esm_score` | Log-odds marginal enmascarado del residuo WT frente al mutante (estilo ESM-1v; mayor = más deletéreo) | 1 |
| `esm_delta` | Estado oculto de la última capa en el residuo mutado, ventana mutante menos ventana WT (51 aa, ±25) | 480 → PCA |

El PCA se ajusta **sólo con variantes presentes en train** (50.512 variantes únicas) y se aplica a val/test. Las variantes no missense reciben ceros más un flag `esm_present`.

## 2. Cobertura y validación del score

- 59.097 de 59.201 variantes missense del VEP tienen transcrito, posición proteica y residuo WT validados contra el proteoma Ensembl release 113 (**99,8%**).
- Tiempo de extracción: 4,9 min en GPU (RTX 2050, fp16).
- AUC del `esm_score` para separar P/LP de benignos en variantes missense únicas: **0,615**.
- En las mismas variantes, AlphaMissense (`am_pathogenicity`) da **0,956**. El score de ESM-2 35M es mucho más débil y correlaciona sólo ρ=0,275 con AlphaMissense.

| Grupo | n | `esm_score` medio |
|---|---:|---:|
| Benigno (label 0) | 36.242 | 0,342 |
| P/LP o causal (label > 0) | 22.855 | 1,393 |

## 3. LightGBM: el bloque sí aporta

Mismo split, mismas features base, 2.000 árboles con early stopping. Δ y IC 95% por bootstrap pareado de 5.000 remuestreos por query contra el baseline.

| Config | Features | NDCG@10 | Recall@1 | Rank Points | Brier | ΔNDCG@10 [IC 95%] |
|---|---:|---:|---:|---:|---:|---|
| base | 21 | 0,9318 | 0,8567 | 96,84 | 0,0085 | — |
| `score` | 23 | 0,9342 | 0,8635 | 96,86 | 0,0084 | +0,0024 [−0,0014; +0,0061] |
| `delta16` | 38 | 0,9270 | 0,8470 | 96,52 | 0,0088 | −0,0048 [−0,0090; −0,0006] |
| `delta48` | 70 | 0,9308 | 0,8527 | 96,79 | 0,0085 | −0,0010 [−0,0050; +0,0030] |
| `delta96` | 118 | 0,9331 | 0,8584 | 96,91 | 0,0086 | +0,0013 [−0,0030; +0,0053] |
| **`both48`** | **71** | **0,9373** | **0,8686** | **97,04** | **0,0085** | **+0,0055 [+0,0011; +0,0096]** |

- `delta16` es el único bloque que **empeora** de forma significativa: 16 componentes no capturan suficiente información y añaden ruido.
- `both48` es el mejor: gana +0,0055 NDCG@10 y +0,0119 Recall@1 con IC que excluye 0.
- En LightGBM los embeddings sí se usan: 21,6% de la importancia total del modelo `both48`.

## 4. CatBoost (primary): la mejora desaparece

Mismos hiperparámetros que el primary (`YetiRank`, 400 iteraciones, depth 6, lr 0,05, semilla 7). Δ contra el CatBoost base reentrenado en esta misma corrida.

| Config | Features | NDCG@10 | Recall@1 | Rank Points | Brier | F-max | ΔNDCG@10 [IC 95%] | ΔRecall@1 [IC 95%] |
|---|---:|---:|---:|---:|---:|---:|---|---|
| base | 21 | 0,9460 | 0,8850 | 97,55 | 0,0078 | 0,768 | — | — |
| `score` | 23 | **0,9469** | 0,8878 | 97,56 | 0,0079 | 0,748 | +0,0009 [−0,0014; +0,0034] | +0,0028 [−0,0028; +0,0085] |
| `both48` | 71 | 0,9466 | 0,8878 | 97,46 | 0,0079 | 0,764 | +0,0006 [−0,0028; +0,0042] | +0,0028 [−0,0057; +0,0113] |

- Ningún IC excluye 0: no hay evidencia de mejora en ranking.
- El `both48` incluso pierde −0,085 Rank Points (IC incluye 0) y baja el F-max de calibración de 0,768 a 0,764.
- Importancia de las features ESM en CatBoost: **1,1%** (`score`) y **4,2%** (`both48`), frente al 7,0% y 21,6% en LightGBM. CatBoost apenas parte por ellas.

## 5. Interpretación

1. **La señal ya está cubierta.** AlphaMissense (AUC 0,956) domina el espacio de patogenicidad missense y ya es una feature. El score ESM-2 35M (AUC 0,615) es redundante y mucho más ruidoso.
2. **El delta ayuda a LightGBM porque compensa su menor capacidad de interacción**, no porque añada información nueva. CatBoost, con boosting ordenado y árboles simétricos, no necesita ese atajo.
3. **El split gene-disjoint es exigente.** Los embeddings a nivel de variante no pueden memorizar genes de test; cualquier ganancia tiene que venir de biología transferible, y con 35M de parámetros no aparece.
4. **Riesgo de sobreajuste al test.** Se probaron seis configuraciones; la mejora de LightGBM es la única con IC que excluye 0 y no se replicó en el modelo de producción. No se debe leer como una mejora validada del sistema.

## 6. Recomendación

- **No promover** el bloque ESM-2 al primary. El modelo desplegado sigue siendo `models/ml_catboost/` con 21 features.
- Si se quiere insistir, el siguiente paso razonable es un modelo más fuerte (**ESM-2 150M/650M** o **ESM-1v** para el score zero-shot), no aumentar componentes PCA del 35M.
- Otra vía más prometedora: usar embeddings sólo para **imputar AlphaMissense** en variantes sin cobertura, en lugar de añadirlos como features globales.
- El código queda reproducible y aislado; no toca `src/ml/rank_features.py` ni el serving.

## 7. Reproducir

```bash
# 1. Descargar el proteoma Ensembl 113 (una vez)
curl -L -o data/references/Homo_sapiens.GRCh38.pep.all.fa.gz \
  https://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/pep/Homo_sapiens.GRCh38.pep.all.fa.gz

# 2. Extraer embeddings ESM-2 (GPU; ~5 min)
.venv/bin/python -m src.ml.esm_features build --batch 512

# 3. Ablación LightGBM (6 configs; ~3,5 min)
.venv/bin/python -m src.ml.esm_ablation

# 4. CatBoost final con el mejor bloque
.venv/bin/python -m src.ml.train_esm_ranker --backend catboost --esm both \
  --n-components 48 --tag catboost_both48 --compare-base catboost_base \
  --save-dir models/ml_catboost_esm
```

Artefactos: `data/ml/esm2_35m_{delta.npy,index.csv.gz,meta.json}` (ignorados por git), `reports/esm_ablation.json`, `reports/esm_ranker_catboost.json`, y el modelo experimental `models/ml_catboost_esm/` (también ignorado por git; 71 features, calibrador y métricas). No se despliega ni se referencia desde `prioritize.py`.
