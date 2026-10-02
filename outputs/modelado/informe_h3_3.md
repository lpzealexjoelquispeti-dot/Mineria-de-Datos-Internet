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
