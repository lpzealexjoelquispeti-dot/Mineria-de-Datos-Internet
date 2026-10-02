# Informe actualizado H3_3 — cuatro mejoras

Se mantiene `feature/dashboard-web` y su arquitectura. Se inspeccionaron ambas implementaciones antes de editar. Solo se trasladaron KFold reproducible, la representación alternativa de habitaciones evaluada por CV, verificación local SHA-256 y exportación de reglas. Sin merge, cherry-pick, rama nueva, commit ni push. La implementación inicial se conserva como anexo histórico al final.

## Comparación directa con rama-cris

Referencia inspeccionada: `origin/rama-cris`, commit `8819a1416a8e0f6e797dcc7d2e8a1f0ce358c984`, obtenido mediante fetch; no se incorporaron commits a la rama de trabajo.

| Aspecto | Implementación actual anterior | rama-cris inspeccionada | Decisión aplicada |
|---|---|---|---|
| Arquitectura | src, script, notebook, API, dashboard, outputs/modelado | módulo, descarga, script, guía, outputs/h3_3 | Conservar arquitectura y rutas actuales |
| Partición | 80/20, 69/18, seed 777 | 70/30, 60/27 | Conservar 80/20 y exactamente los municipios previos |
| CV | cv=5 sin shuffle, n_jobs=-1 | KFold shuffle seed777, n_jobs=1 | KFold shuffle seed777, mantener n_jobs=-1 |
| Habitaciones | Media de códigos 1–8 | Porcentaje 3–8 sobre total; inválidos como False | Comparar A/B; B sobre códigos válidos 1–8, inválidos ausentes |
| Electricidad | Disponibilidad 1–4 | Red pública solamente | Mantener disponibilidad 1–4 |
| Computadora/celular | Sí/(Sí+No) | Sí/total universo | Mantener respuestas determinadas |
| Fuentes | SHA-256 en src.pipeline y EDA | Verificación y script con descarga/hashes propios | Reutilizar src.pipeline.sha256; sin descarga ni hash inventado |
| Reglas | Gráfico y joblib | export_text y ruta | Exportar el best_estimator_ final y ruta reutilizable |
| Resultados | Árbol profundidad3 y métricas propias | Otro split/modelo con resultados propios | Reentrenar completamente; no copiar cifras de Cris |

La revisión incluyó los módulos, scripts, notebooks, tests, README, backend/frontend y artefactos indicados en la solicitud; en Cris también su guía, script de descarga y outputs/h3_3.

## 1. Archivos modificados (19)

- `README.md`
- `backend/app/schemas/responses.py`
- `backend/tests/test_arbol_regresion.py`
- `frontend/src/components/RegressionTreeSection.tsx`
- `frontend/src/types/api.ts`
- `notebooks/H3_3_Arbol_Regresion_Internet_LaPaz.ipynb`
- `outputs/graficos/arbol_regresion_niveles_0_3.png`
- `outputs/graficos/regresion_real_vs_predicho.png`
- `outputs/graficos/residuos_arbol_regresion.png`
- `outputs/modelado/comparacion_baseline_arbol_regresion.csv`
- `outputs/modelado/dataset_regresion_municipal.csv`
- `outputs/modelado/gridsearch_arbol_regresion.csv`
- `outputs/modelado/importancia_variables_arbol_regresion.csv`
- `outputs/modelado/informe_h3_3.md`
- `outputs/modelado/metricas_arbol_regresion.json`
- `outputs/modelado/predicciones_arbol_regresion_test.csv`
- `outputs/modelos/arbol_regresion.joblib`
- `src/decision_tree_regression.py`
- `tests/test_decision_tree_regression.py`

`scripts/entrenar_arbol_regresion.py` se mantiene: ya delega al módulo y no necesita duplicar la nueva lógica. `src.pipeline`, rutas/servicio backend y código H3_1/H3_2 permanecen iguales.

## 2. Archivos creados (6)

- `outputs/modelado/comparacion_features_habitaciones.csv`
- `outputs/modelado/folds_cv_arbol_regresion.csv`
- `outputs/modelado/fuentes_censo_sha256.json`
- `outputs/modelado/reglas_arbol_regresion.txt`
- `scripts/verificar_fuentes_censo.py`
- `src/census_sources.py`

No se elimina ninguno de los artefactos anteriores. `particion_arbol_regresion.csv` permanece idéntico byte por byte.

## 3. Cambio en GridSearchCV

Dos búsquedas controladas, una por representación de habitaciones, sobre los mismos 69 municipios train. Cada una conserva 608 combinaciones (19 × 8 × 4), 3.040 ajustes de validación, un refit y `n_jobs=-1`, `scoring="neg_mean_squared_error"`. Total: 1.216 candidatos, 6.080 ajustes CV y dos refits. Un único pipeline común incluye imputación por mediana aprendida dentro de cada fold.

## 4. KFold con shuffle confirmado

`KFold(n_splits=5, shuffle=True, random_state=777)`. El mismo objeto se pasa a ambas búsquedas. Las pruebas comprueban tipo, número de folds, shuffle, semilla, igualdad de índices y folds, y exclusión de test. El JSON guarda `gridsearch.configuracion_cv`.

## 5–6. CV de las variantes A y B

| Variante | MSE CV (pp²) | Desviación CV (pp²) | Mejores parámetros | Seleccionada |
|---|---|---|---|---|
| A | 33.128464626 | 10.453981313 | {"max_depth": 7, "min_samples_leaf": 1, "min_samples_split": 8} | False |
| B | 32.305246106 | 11.463718428 | {"max_depth": 7, "min_samples_leaf": 1, "min_samples_split": 8} | True |

A conserva la media de códigos; B reemplaza solo esa variable por porcentaje con tres o más. El dataset mantiene ambas columnas para auditoría. No se evalúan ambos árboles en test para elegir.

## 7. Variante seleccionada

**B**, mediante `seleccion_dinamica_por_cv_train`. `SELECTED_FEATURE_SET` documenta esa política dinámica, y las features finales se guardan en `seleccion_features.variables` y `pipeline.feature_names_in_`. `MODEL_FEATURES` representa A al preparar X por compatibilidad; no se usa para imponer el modelo final.

## 8. Razón de selección

Variante B tiene menor MSE medio de validación cruzada sobre train. Diferencia B − A = **-0.823218521 pp²**. No hubo empate: la tolerancia predefinida es `1e-8 + 1e-6 × min(MSE_A, MSE_B)`, aquí 0.000032315246 pp². Si hubiera equivalencia se preferiría B por interpretar ocho o más sin promediar esa categoría. La selección recibe exclusivamente los MSE CV train; los tests alteran resultados test irrelevantes y vigilan que la evaluación final ocurre después de seleccionar.

## 9. Features finales

- `pct_urbano`
- `pct_con_energia`
- `pct_computadora`
- `pct_celular`
- `pct_3_o_mas_habitaciones`
- `promedio_personas`

B = códigos de habitaciones 3–8 / respuestas válidas 1–8 × 100 en el universo TIC. El código 8 significa ocho o más. Los inválidos se excluyen del denominador y quedan NaN si no hay respuestas válidas. Computadora/celular siguen Sí/(Sí+No), electricidad disponibilidad 1–4, personas 0–9999; se conservan conteos válidos. Se mantienen 87 municipios, 1.069.838 viviendas TIC, target del EDA, nombres/códigos/denominadores y lista de predictores prohibidos.

## 10. Hiperparámetros finales

```json
{
  "max_depth": 7,
  "min_samples_leaf": 1,
  "min_samples_split": 8
}
```

El pipeline guardado es exactamente el `best_estimator_` de la búsqueda B, refit solo con train.

## 11. Mejor MSE CV

**32.305246106 pp²**. `best_score_ = -32.305246106`.

## 12. Desviación y MSE por fold

Desviación poblacional de los cinco MSE: **11.463718428 pp²**. No es un intervalo de confianza.

| Fold | MSE validación (pp²) |
|---|---|
| 1 | 38.552101647 |
| 2 | 21.960429659 |
| 3 | 48.945861439 |
| 4 | 34.850851488 |
| 5 | 17.216986296 |

La tabla se guarda en `folds_cv_arbol_regresion.csv` para el árbol seleccionado; los resultados completos de ambos grids incluyen la columna `variante`.

## 13–15. Métricas train, test y baseline

| Conjunto/modelo | MSE (pp²) | RMSE (pp) | MAE (pp) | R² |
|---|---|---|---|---|
| Árbol train | 5.894160981 | 2.427789320 | 1.904241376 | 0.961125697 |
| Árbol test | 44.113496965 | 6.641799227 | 5.047917838 | 0.460415397 |
| Dummy test | 113.197065295 | 10.639410947 | 8.898895796 | -0.384596500 |

Dummy aprende la media de train **50.547050888 %**, conserva su R² negativo y se evalúa en el mismo test. Solo el árbol elegido por CV recibe la evaluación final test, una vez por ejecución del flujo. El árbol base se evalúa únicamente en train. Reejecutar el notebook reproduce la misma selección fija; no se reajusta a partir de test.

## 16. Estructura

Profundidad **7**, **39 nodos**, **20 hojas**. PNG limitado a niveles 0–3; joblib y reglas contienen el árbol completo.

## 17. Importancias

| Predictor | Importancia |
|---|---|
| pct_celular | 0.816376345 |
| pct_urbano | 0.115900395 |
| pct_computadora | 0.034111133 |
| pct_con_energia | 0.018939579 |
| pct_3_o_mas_habitaciones | 0.012006520 |
| promedio_personas | 0.002666029 |

Suma: 1.000000000000. Son contribuciones a divisiones predictivas, sin interpretación causal.

## 18. Reglas principales

El archivo completo se genera con `export_text` del árbol final, los nombres seleccionados y umbrales a seis decimales; se verificó igualdad textual contra el modelo joblib cargado. Algunas rutas completas:

- `pct_celular <= 75.235550` → acceso predicho **31.727548 %** (la ruta también satisface los cortes superiores 86.550304 y 81.898121).
- `pct_celular > 86.550304` y `pct_urbano > 88.684383` → **86.977760 %**.
- `86.550304 < pct_celular <= 88.131950`, `pct_urbano <= 88.684383`, `pct_computadora <= 8.140999` → **44.028103 %**.

`explicar_prediccion` muestra nodo, predictor, umbral real, valor imputado convertido a float32 como usa scikit-learn, condición, hoja y predicción. El notebook incluye un ejemplo. No se ajusta un segundo árbol para reglas ni gráficos.

## 19. Fuentes verificadas

**Vivienda_CPV-2024.csv**

- Ruta: `Base de datos CSV/Vivienda_CPV-2024.csv`
- Tamaño: **490,862,565 bytes**
- SHA-256: `f3cef44bcf103978734e86adf2c8c147a89adcb7146c78f78df7c409107a7aee`

**Diccionario de variables CPV 2024.xlsx**

- Ruta: `Base de datos CSV/Diccionario de variables CPV 2024.xlsx`
- Tamaño: **204,479 bytes**
- SHA-256: `20e1f074145b8277ad18a71bc180a3e2f89f0b38fc9760bf88be0bc766e2db0e`

Coinciden con las huellas previas de `outputs/resumen_eda.json` y con el nuevo manifiesto al repetir la verificación. Se reutiliza la única implementación SHA de `src.pipeline`; no se descargan datos. Tests usan archivos temporales pequeños, incluso cambios de igual tamaño y rechazo sin sobrescribir referencias.

## 20. Notebook

**47 celdas, 23 de código ejecutadas, cero errores**. Ejecutado completamente con nbconvert y salidas guardadas. Explica shuffle, categoría ocho o más, denominadores, comparación A/B, selección por train/CV, métricas elegidas, fuentes, reglas y ruta. Las cifras se obtienen del flujo real; no se hardcodean métricas en las celdas.

## 21. Tests raíz

`python -m pytest -q tests`: **31 passed**, 35,80 s en la ejecución final. Se conservan las pruebas previas adaptando las expectativas de CV/features y se amplían con selección, mismos índices/folds, evaluación posterior única, habitaciones, fuentes, reglas/modelo y artefactos auditables.

## 22. Tests backend

`cd backend` y `python -m pytest -q`: **10 passed**, 1,29 s. El schema expone selección, comparación y fuentes; `gridsearch` conserva su configuración CV. El endpoint sigue leyendo solo el JSON, maneja ausencia con 503 y preserva R² negativo.

## 23. Build frontend

`npm run build`: **correcto**, TypeScript + Vite, 2.216 módulos. JS 679,85 kB, gzip 192,17 kB. Permanece el aviso por chunk >500 kB; no se cambian dependencias. La sección actual incorpora discretamente KFold shuffle/semilla, representación de habitaciones elegida y variables finales. No muestra hashes ni reglas completas.

## 24. H3_1/H3_2 y HTTP

Los tres endpoints reales, servidos temporalmente por Uvicorn en `127.0.0.1`, respondieron **HTTP 200** y sus métricas coinciden con los JSON. El proceso temporal se cerró.

Se cargaron los joblib H3_1/H3_2 y se recalcularon métricas sobre sus **207.869 viviendas test completas** con cada evaluador original: todas coinciden a tolerancia absoluta 1e-12.

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| H3_1 | 0.803443514906 | 0.814424649381 | 0.941582431972 | 0.873399600907 | 0.862539584406 |
| H3_2 | 0.805411100260 | 0.823688733480 | 0.928514641137 | 0.872966072152 | 0.865199100384 |

La comparación SHA de archivos Git previos confirma que los cambios quedan en el alcance autorizado. Fuentes originales, EDA, código/modelos/métricas/notebooks H3_1/H3_2 y partición municipal permanecen iguales. Las seis features originales y el target del dataset mantienen sus valores a tolerancia 1e-12; solo se añade la alternativa de habitaciones.

## 25. git diff --check

Correcto, exit code 0. Se repite después de escribir este informe.

## 26. git status

Rama `feature/dashboard-web`, misma referencia HEAD, sin staging, commit o push. 19 archivos modificados y 6 archivos nuevos, todos en H3_3/documentación o su integración. El estado exacto se incluye a continuación.

```text
## feature/dashboard-web...origin/feature/dashboard-web
 M README.md
 M backend/app/schemas/responses.py
 M backend/tests/test_arbol_regresion.py
 M frontend/src/components/RegressionTreeSection.tsx
 M frontend/src/types/api.ts
 M notebooks/H3_3_Arbol_Regresion_Internet_LaPaz.ipynb
 M outputs/graficos/arbol_regresion_niveles_0_3.png
 M outputs/graficos/regresion_real_vs_predicho.png
 M outputs/graficos/residuos_arbol_regresion.png
 M outputs/modelado/comparacion_baseline_arbol_regresion.csv
 M outputs/modelado/dataset_regresion_municipal.csv
 M outputs/modelado/gridsearch_arbol_regresion.csv
 M outputs/modelado/importancia_variables_arbol_regresion.csv
 M outputs/modelado/informe_h3_3.md
 M outputs/modelado/metricas_arbol_regresion.json
 M outputs/modelado/predicciones_arbol_regresion_test.csv
 M outputs/modelos/arbol_regresion.joblib
 M src/decision_tree_regression.py
 M tests/test_decision_tree_regression.py
?? outputs/modelado/comparacion_features_habitaciones.csv
?? outputs/modelado/folds_cv_arbol_regresion.csv
?? outputs/modelado/fuentes_censo_sha256.json
?? outputs/modelado/reglas_arbol_regresion.txt
?? scripts/verificar_fuentes_censo.py
?? src/census_sources.py
```

## 27. Limitaciones nuevas y persistentes

- La elección A/B utiliza los mismos folds para selección, por lo que el mejor MSE CV puede resultar optimista. La evaluación test reservada sigue siendo externa a esa selección; no se afirma significancia estadística entre variantes.
- El árbol elegido es más profundo y presenta mayor separación train/test. El test tiene solo 18 municipios y la desviación entre folds no es un intervalo de confianza.
- B evita promediar la categoría abierta ocho o más, pero pierde detalle al agrupar habitaciones; los porcentajes determinados dependen de la no respuesta.
- Shuffle no soluciona dependencias espaciales ni demuestra generalización fuera de La Paz 2024. No hay ponderación municipal por tamaño ni inferencia causal/individual.
- Las huellas identifican originales y detectan cambios frente a referencias locales. No constituyen validación oficial externa ni certifican por sí solas el cache departamental. `--forzar-fuente` permite regenerar desde originales.
- Permanece el aviso de tamaño del bundle frontend. La compilación y HTTP/API fueron comprobados; en esta actualización no se inspeccionó visualmente el dashboard en navegador. El PNG del árbol sí se inspeccionó.

## Antes vs. después

| Medida | Antes | Después |
|---|---|---|
| Habitaciones | promedio_habitaciones | pct_3_o_mas_habitaciones (B) |
| CV | KFold 5 sin shuffle | KFold 5, shuffle=True, random_state=777 |
| Train / test | 69 / 18 | 69 / 18, mismos municipios |
| Mejores parámetros | {'max_depth': 3, 'min_samples_leaf': 3, 'min_samples_split': 2} | {'max_depth': 7, 'min_samples_leaf': 1, 'min_samples_split': 8} |
| Profundidad / nodos / hojas | 3 / 13 / 7 | 7 / 39 / 20 |
| MSE CV | 43.934603764 | 32.305246106 |
| Desviación CV | 27.440629174 | 11.463718428 |
| MSE train | 18.898553000 | 5.894160981 |
| RMSE train | 4.347246600 | 2.427789320 |
| MAE train | 3.474540063 | 1.904241376 |
| R2 train | 0.875356634 | 0.961125697 |
| MSE test | 28.011985065 | 44.113496965 |
| RMSE test | 5.292634983 | 6.641799227 |
| MAE test | 4.563589932 | 5.047917838 |
| R2 test | 0.657364823 | 0.460415397 |

Los folds cambiaron: los MSE CV anterior y actual no son una comparación bajo idénticas asignaciones. La comparación controlada es A vs. B con KFold nuevo. El error test **aumentó**: MSE de 28,011985 a 44,113497 pp² y MAE de 4,563590 a 5,047918 pp; R² bajó de 0,657365 a 0,460415. La elección B se justifica exclusivamente por menor MSE CV train (32,305246 vs. 33,128465), sin usar test para revertirla ni afirmar que el modelo sea mejor por el test. El árbol todavía supera al baseline según MSE en ese conjunto.

<details>
<summary>Informe histórico de implementación inicial — resultados anteriores, conservados</summary>

# Informe de implementación y validación H3_3

Implementación exclusiva de H3_3 en `feature/dashboard-web`. No se crearon ramas ni se hizo commit o push. Los resultados siguientes provienen del entrenamiento censal ejecutado, sin datos sintéticos ni aumento artificial de observaciones.

## 1. Archivos creados

- `backend/tests/test_arbol_regresion.py`
- `frontend/src/components/RegressionTreeSection.tsx`
- `frontend/src/hooks/useRegressionTree.ts`
- `notebooks/H3_3_Arbol_Regresion_Internet_LaPaz.ipynb`
- `outputs/graficos/arbol_regresion_niveles_0_3.png`
- `outputs/graficos/regresion_real_vs_predicho.png`
- `outputs/graficos/residuos_arbol_regresion.png`
- `outputs/modelado/comparacion_baseline_arbol_regresion.csv`
- `outputs/modelado/dataset_regresion_municipal.csv`
- `outputs/modelado/gridsearch_arbol_regresion.csv`
- `outputs/modelado/importancia_variables_arbol_regresion.csv`
- `outputs/modelado/informe_h3_3.md`
- `outputs/modelado/metricas_arbol_regresion.json`
- `outputs/modelado/particion_arbol_regresion.csv`
- `outputs/modelado/predicciones_arbol_regresion_test.csv`
- `outputs/modelos/arbol_regresion.joblib`
- `scripts/entrenar_arbol_regresion.py`
- `src/decision_tree_regression.py`
- `tests/test_decision_tree_regression.py`

## 2. Archivos modificados

- `README.md`
- `backend/app/api/routes/geografico.py`
- `backend/app/schemas/responses.py`
- `backend/app/services/data_service.py`
- `frontend/src/pages/Dashboard.tsx`
- `frontend/src/services/api.ts`
- `frontend/src/styles.css`
- `frontend/src/types/api.ts`

## 3. Construcción del dataset municipal

Se cargaron 1,334,496 registros reales de La Paz, utilizando la copia departamental reproducible. Se conservaron 1,069,838 viviendas del universo TIC del EDA: tipos particulares 1–6 y ocupación 0/1. Se agruparon por código municipal completo (idep sin cero inicial + iprov + imun), con nombres del catálogo oficial. Se reutilizaron load_lapaz/read_dictionary, summarize_groups y la limpieza de seleccionar_variables de H3_1. No se excluyeron viviendas por su respuesta a Internet.

Target: todas las viviendas del universo TIC, incluido Internet sin especificar. Porcentajes predictores: respuestas determinadas de cada variable; 9/vacíos no se recodifican como No. Medias: valores válidos según H3_1/H3_2. No se imputa antes de agregar ni antes de separar train/test.

La disponibilidad de electricidad se verificó en el diccionario original: servicio público (1), generador (2), panel solar (3) y otro (4); ausencia (5). Computadora/celular: 1=Sí, 2=No, 9=Sin especificar. Habitaciones válidas 1–8 (8=Ocho o más); personas válidas 0–9999. El dataset agregado se ordena por municipio_codigo; no contiene registros individuales. Los conteos válidos de cada predictor por municipio están en el JSON.

## 4. Número exacto de municipios

**87 municipios**, con cero duplicados y cero targets NaN. No se añadieron municipios ni se replicaron filas.

## 5. Target

`PORCENTAJE_ACCESO_INTERNET` = `pct_algun` = viviendas con v19e_f=1 / universo TIC municipal × 100. Se mantiene sin especificar en el denominador. Diferencia máxima respecto del EDA: 7.105427357601002e-15 pp, por debajo de la tolerancia absoluta de 1e-8 pp. También se verificaron códigos, nombres y denominadores uno a uno.

## 6. Variables predictoras finales

- `pct_urbano`: Porcentaje de viviendas urbanas del universo TIC.
- `pct_con_energia`: Porcentaje con electricidad entre respuestas determinadas.
- `pct_computadora`: Porcentaje con computadora/laptop/tablet entre respuestas determinadas.
- `pct_celular`: Porcentaje con teléfono celular entre respuestas determinadas.
- `promedio_habitaciones`: Media del código de habitaciones válido (8 = ocho o más).
- `promedio_personas`: Media de personas por vivienda con valor válido.

## 7. Prevención de data leakage

Lista cerrada de seis predictores; quedan fuera de X todos los indicadores/derivados de Internet, el target y los identificadores. La agregación no imputa ni utiliza Internet para construir características. SimpleImputer(median) se ajusta dentro de cada fold y el modelo final solo sobre train. GridSearchCV jamás recibe test. Los hiperparámetros no se reajustan a partir de los resultados test. Una prueba altera únicamente Internet en registros reales y comprueba que los predictores permanecen iguales.

## 8. Estadística descriptiva del target

| Medida | Porcentaje municipal |
|---|---:|
| minimo | 24.327418432 |
| maximo | 89.835452774 |
| promedio | 49.386906922 |
| mediana | 48.227535037 |

## 9. Partición

Train: **69** municipios; test: **18**. test_size=0.20, random_state=777, sin stratify. Se mantiene la convención 777 y 80/20 del proyecto frente a 22 y 70/30 del docente; sin stratify. Las listas completas se guardan en metricas_arbol_regresion.json y particion_arbol_regresion.csv.

## 10. GridSearchCV

Búsqueda completa de **608 configuraciones**, **3040 ajustes CV** y un reajuste final. max_depth=1..19, min_samples_split=2..9, min_samples_leaf=1..4; cv=5, neg_mean_squared_error y n_jobs=-1. KFold sin shuffle. Todos los tiempos, parámetros, puntajes train/validación por fold y rankings se conservan en gridsearch_arbol_regresion.csv.

## 11. Mejores hiperparámetros

```json
{
  "max_depth": 3,
  "min_samples_leaf": 3,
  "min_samples_split": 2
}
```

## 12. Mejor MSE de validación cruzada

**43.934603764 pp²**. best_score_ = -43.934603764. El modelo final es best_estimator_ reajustado sobre train.

## 13. Desviación estándar CV

**27.440629174 pp²**. MSE de cada fold: 73.816420647, 25.434329273, 80.785096972, 17.740842934, 21.896328994. Esta dispersión no es un intervalo de confianza.

## 14–16. Métricas train, test y baseline

| Conjunto/modelo | MSE (pp²) | RMSE (pp) | MAE (pp) | R² |
|---|---:|---:|---:|---:|
| Árbol — train | 18.898553000 | 4.347246600 | 3.474540063 | 0.875356634 |
| Árbol — test | 28.011985065 | 5.292634983 | 4.563589932 | 0.657364823 |
| DummyRegressor — test | 113.197065295 | 10.639410947 | 8.898895796 | -0.384596500 |

El baseline aprende solo la media train: **50.547050888 %**. R² negativo del baseline se conserva. El árbol base ilustrativo solo se evalúa en train; test se usa para el modelo elegido por CV y el baseline.

## 17. Diferencia árbol vs. baseline

| Métrica | Árbol menos baseline en test |
|---|---:|
| mse | -85.185080230 |
| rmse | -5.346775964 |
| mae | -4.335305864 |
| r2 | 1.041961323 |

El árbol redujo el MSE en 75.253789 % respecto del baseline. MAE significa una diferencia absoluta promedio de 4.563590 puntos porcentuales en los municipios test; RMSE se expresa también en pp y MSE en pp².

## 18. Importancias de variables

| Predictor | Importancia |
|---|---:|
| pct_celular | 0.869245374 |
| pct_con_energia | 0.130754626 |
| pct_urbano | 0.000000000 |
| pct_computadora | 0.000000000 |
| promedio_habitaciones | 0.000000000 |
| promedio_personas | 0.000000000 |

Suma = 0.9999999999999999. Cero significa que ese predictor no se utilizó en las divisiones de este árbol, sin establecer ausencia de asociación o efectos causales.

## 19. Estructura del árbol

Profundidad real **3**, **13 nodos** y **7 hojas**. La visualización se limita a niveles 0–3; en este ajuste abarca todo el árbol. El joblib conserva el pipeline completo.

## 20. Municipios con mayor error absoluto

| Código | Municipio | Real (%) | Predicho (%) | Residuo (pp) | Error absoluto (pp) |
|---|---|---:|---:|---:|---:|
| 21006 | Villa Libertad Licoma | 51.042223 | 39.965180 | 11.077044 | 11.077044 |
| 20608 | Teoponte | 58.691873 | 48.601346 | 10.090527 | 10.090527 |
| 20805 | La (Marka) San Andrés de Machaca | 42.035967 | 48.601346 | -6.565379 | 6.565379 |
| 21602 | Curva | 33.997585 | 39.965180 | -5.967595 | 5.967595 |
| 20905 | Cairoma | 48.180703 | 54.016831 | -5.836128 | 5.836128 |

Residuos real − predicho: promedio -1.039067795 pp; mínimo -6.565379087 pp; máximo 11.077043661 pp. No se eliminaron municipios por error ni se reajustó el árbol observando test.

## 21. Notebook

`notebooks/H3_3_Arbol_Regresion_Internet_LaPaz.ipynb`: 41 celdas, incluidas 20 celdas de código ejecutadas completamente con nbconvert, sin errores. Tiene salidas guardadas, gráficos, auditoría, CV, métricas, baseline y conclusiones dinámicas. Se corrigió un literal de salto de línea en la celda de limitaciones y se repitió la ejecución completa con éxito.

## 22. Tests Python

`python -m pytest -q tests`: **24 passed** (12,94 s), incluidas las cuatro pruebas anteriores y veinte pruebas/casos H3_3. Cubren unidad municipal, concordancia del target, daños que detienen el flujo, separación reproducible sin stratify, búsqueda solo en train, categorías verificadas, limpieza, aislamiento del target, métricas y R² negativo, mediana train, importancias, serialización, correspondencia de predicciones, partición y artefactos.

## 23. Backend y compatibilidad

`cd backend && pytest -q`: **10 passed** (0,26 s). H3_3 devuelve los artefactos sin entrenar, maneja ausencia con 503 y preserva R² negativo. Los tres endpoints `/api/mineria/regresion-logistica`, `/api/mineria/arbol-clasificacion` y `/api/mineria/arbol-regresion` respondieron **HTTP 200**, tanto en tests como contra Uvicorn temporal.

Se cargaron los pipelines joblib H3_1/H3_2 y se recalcularon sus métricas sobre las **207,869 viviendas test completas**, con los evaluadores propios de cada módulo. Las métricas coinciden con sus JSON a tolerancia 1e-12. Las huellas SHA-256 de **61 archivos originales** de src/scripts/notebooks/outputs coinciden sin cambios.

## 24. Frontend

`npm run build`: **correcto**, TypeScript y Vite; 2.216 módulos transformados. Se utilizó Node Linux v22.21.1 instalado en WSL, porque el primer intento invocó npm de Windows desde una ruta UNC. Vite informa un bundle JS de 679,15 kB (191,96 kB gzip), superior al umbral informativo de 500 kB. No se alteraron dependencias ni lockfile. El servidor Vite temporal respondió HTTP 200.

La sección H3_3 tiene carga independiente, cuatro métricas de regresión, municipios, hiperparámetros, estructura, scatter real/predicho con y=x, importancias, comparación baseline/train/test, tabla de errores y limitaciones. Las secciones de clasificación y su comparación permanecen separadas. Los gráficos PNG se inspeccionaron visualmente. La herramienta de navegador no pudo inicializarse, por lo que no se realizó inspección visual del dashboard en navegador.

## 25. git diff --check

Correcto, exit code 0, sin advertencias de espacios o conflictos.

## 26. git status

```text
On branch feature/dashboard-web
Your branch is up to date with 'origin/feature/dashboard-web'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   README.md
	modified:   backend/app/api/routes/geografico.py
	modified:   backend/app/schemas/responses.py
	modified:   backend/app/services/data_service.py
	modified:   frontend/src/pages/Dashboard.tsx
	modified:   frontend/src/services/api.ts
	modified:   frontend/src/styles.css
	modified:   frontend/src/types/api.ts

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	backend/tests/test_arbol_regresion.py
	frontend/src/components/RegressionTreeSection.tsx
	frontend/src/hooks/useRegressionTree.ts
	notebooks/H3_3_Arbol_Regresion_Internet_LaPaz.ipynb
	outputs/graficos/arbol_regresion_niveles_0_3.png
	outputs/graficos/regresion_real_vs_predicho.png
	outputs/graficos/residuos_arbol_regresion.png
	outputs/modelado/comparacion_baseline_arbol_regresion.csv
	outputs/modelado/dataset_regresion_municipal.csv
	outputs/modelado/gridsearch_arbol_regresion.csv
	outputs/modelado/importancia_variables_arbol_regresion.csv
	outputs/modelado/informe_h3_3.md
	outputs/modelado/metricas_arbol_regresion.json
	outputs/modelado/particion_arbol_regresion.csv
	outputs/modelado/predicciones_arbol_regresion_test.csv
	outputs/modelos/arbol_regresion.joblib
	scripts/entrenar_arbol_regresion.py
	src/decision_tree_regression.py
	tests/test_decision_tree_regression.py

no changes added to commit (use "git add" and/or "git commit -a")
```

## 27. Limitaciones encontradas

- Solo 87 municipios, frente a modelos individuales con muchas más viviendas; test contiene 18 municipios.
- Una única partición y cinco folds producen una evaluación sensible a los territorios disponibles; la desviación CV no es un intervalo de confianza.
- Cada municipio pesa una vez, sin ponderar por número de viviendas; posibles dependencias espaciales no se modelan.
- Promedio de habitaciones usa códigos: 8 representa ocho o más, por lo que es una aproximación inferior en esa categoría.
- Los porcentajes de equipamiento se calculan entre respuestas determinadas; se conservan conteos válidos para auditar no respuesta.
- Acceso declarado en el censo 2024; no mide calidad del servicio ni identifica causalidad o efectos individuales (riesgo de falacia ecológica).
- No se garantiza generalización a otros departamentos, años, países ni condiciones futuras.

- Las pruebas de agregación usan las fuentes censales y la copia departamental locales; estos microdatos permanecen excluidos de Git.

- La herramienta de navegador no se inició; compilación, respuesta HTTP, datos del endpoint y gráficos PNG sí fueron comprobados.

La revisión automática rechazó inicialmente enlazar la vista temporal a 0.0.0.0 por exposición innecesaria a la red. Se utilizó la alternativa permitida 127.0.0.1 y se cerraron los dos servicios temporales tras las comprobaciones.

## Interpretación final

Se predijo el porcentaje municipal de viviendas con Internet mediante pct_urbano, pct_con_energia, pct_computadora, pct_celular, promedio_habitaciones, promedio_personas. CV seleccionó {'max_depth': 3, 'min_samples_leaf': 3, 'min_samples_split': 2}. En test, la diferencia absoluta promedio fue 4.564 puntos porcentuales y RMSE=5.293 pp. R²=0.6574; la variabilidad explicada debe interpretarse con el pequeño test municipal. El árbol superó al baseline según MSE (árbol=28.012; baseline=113.197 pp²). La mayor importancia corresponde a pct_celular (0.8692). Los mayores errores se observan en Villa Libertad Licoma, Teoponte, La (Marka) San Andrés de Machaca, Curva, Cairoma. Las importancias describen contribuciones a la predicción; no permiten atribuir efectos causales.

</details>
