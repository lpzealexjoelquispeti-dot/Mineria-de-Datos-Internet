# Conectividad Digital en La Paz

## Descripción

Aplicación web de solo lectura para visualizar y explorar el análisis del acceso a Internet fijo y móvil en el departamento de La Paz a partir del **Censo de Población y Vivienda 2024**.

El proyecto conserva el notebook y los resultados del EDA, añade pipelines reproducibles de regresión logística, árbol de clasificación y árbol de regresión municipal, expone los artefactos procesados mediante FastAPI y los presenta en un dashboard React. No incluye usuarios, autenticación, administración, pagos ni base de datos transaccional.

La unidad de análisis del EDA y H3_1/H3_2 es el registro de vivienda; H3_3 usa una fila por municipio. El universo TIC comprende viviendas particulares (tipos 1–6) con personas presentes (ocupación 0/1). Los registros fuera de ese universo no se consideran viviendas sin Internet.

## Arquitectura

```text
Censo 2024 (CSV + diccionario oficial)
                 ↓
       Pipeline Python en src/
                 ↓
  outputs/ (JSON y CSV agregados pequeños)
                 ↓
            FastAPI
                 ↓
      React + TypeScript + Vite
```

FastAPI no abre los microdatos censales ni entrena modelos. `src.pipeline` y los scripts de entrenamiento realizan el trabajo intensivo una sola vez y generan artefactos en `outputs/`. El backend carga los JSON pequeños y los conserva en caché de proceso.

El notebook usa las funciones estadísticas centrales de `src/mining.py` para la comparación departamental, las agregaciones geográficas y la detección IQR. Así se mantienen los mismos denominadores y la misma metodología entre el EDA y los resultados servidos.

## Tecnologías

- Python 3.12
- Pandas y NumPy
- scikit-learn para los pipelines predictivos y las métricas de clasificación y regresión
- statsmodels para la inferencia estadística con `Logit`
- joblib para guardar el pipeline entrenado de forma reproducible
- Matplotlib y seaborn para el EDA existente
- openpyxl para el diccionario censal
- FastAPI, Pydantic y Uvicorn
- pytest y httpx
- React 19, TypeScript y Vite
- Recharts para visualización interactiva

## Fuentes

Coloque los originales con sus nombres exactos en `Base de datos CSV/`:

| Archivo | Uso |
|---|---|
| `Vivienda_CPV-2024.csv` | Fuente de todas las observaciones estadísticas |
| `Diccionario de variables CPV 2024.xlsx` | Categorías, descripciones y nombres geográficos oficiales |
| `Cuestionario censal 2024.pdf` | Referencia documental |
| `Persona_CPV-2024.csv` | Solo inspección del EDA; no participa en estadísticas |
| `Emigracion_CPV-2024.csv` | Solo inventario/inspección |
| `Mortalidad_CPV-2024.csv` | Solo inventario/inspección |

Los microdatos están excluidos de Git y nunca son modificados por el pipeline. Las huellas SHA-256 de las fuentes utilizadas se guardan en `outputs/resumen_eda.json`.

## Instalación

Desde la raíz del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r backend/requirements.txt
```

Se mantienen dos archivos de dependencias deliberadamente:

- `requirements.txt` contiene el entorno de minería y notebook.
- `backend/requirements.txt` contiene únicamente la API y sus tests.

Esta separación permite ejecutar el backend sin convertir sus dependencias en parte de la lógica estadística y no rompe el entorno original del EDA.

## Procesamiento de datos

### Regeneración completa desde los CSV originales

Desde la raíz, con el entorno activado:

```bash
python -m src.pipeline
```

El proceso:

1. lee el diccionario oficial incluido;
2. detecta el código de La Paz;
3. recorre `Vivienda_CPV-2024.csv` en bloques de 200.000 filas;
4. conserva 16 variables y filtra La Paz;
5. determina el universo TIC aplicable;
6. valida categorías y correspondencias geográficas;
7. calcula resumen, áreas, provincias, municipios, distribución y outliers;
8. escribe artefactos agregados pequeños en `outputs/`.

La copia departamental comprimida se guarda en:

```text
outputs/datos/vivienda_lapaz_seleccion.csv.gz
```

Esa copia también está excluida de Git porque contiene microdatos.

### Regeneración rápida de desarrollo

Si la copia comprimida ya existe:

```bash
python -m src.pipeline --from-cache
```

Esta opción vuelve a calcular los artefactos agregados desde las filas departamentales reales y evita recorrer otra vez todo el CSV nacional. No cambia fórmulas ni denominadores.

### Notebook

El EDA narrativo y sus nueve gráficos permanecen en `notebooks/EDA_Internet_LaPaz.ipynb`. Para ejecutarlo y guardar salidas:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/EDA_Internet_LaPaz.ipynb
```

## Regresión Logística

La primera iteración estima la probabilidad de acceso a Internet fijo y/o móvil en una vivienda u hogar de La Paz. El target `TIENE_ACCESO_INTERNET` se construye con la variable oficial `v19e_f`: `1` se codifica como acceso y `2` como ausencia de acceso; `9 = Sin especificar` no se convierte en la clase negativa.

Los predictores son `urbrur`, `v01_tipoviv`, `v09_energia`, `v19c_compu`, `v19d_celular`, `v13_habitac` y `tot_pers`. Las variables de Internet fijo, móvil y combinado se excluyen para evitar fuga de información. Las categóricas se imputan y codifican con one-hot encoding; las numéricas se imputan con la mediana.

Entrenar y regenerar los artefactos:

```bash
python scripts/entrenar_regresion_logistica.py
```

El comando usa el conjunto completo de registros válidos, divide `80/20` con `random_state=777` y estratificación, entrena `LogisticRegression(max_iter=1000)`, evalúa train/test, ajusta `statsmodels.api.Logit` sobre train y guarda:

- `outputs/modelado/metricas_regresion_logistica.json`;
- `outputs/modelado/matriz_confusion_test.csv`;
- `outputs/modelado/coeficientes_logisticos.csv`;
- `outputs/modelado/predicciones_test.csv.gz`;
- `outputs/modelado/resumen_statsmodels.txt`;
- `outputs/modelos/regresion_logistica.joblib`.

El modelo se guarda para permitir inferencia reproducible sin volver a ajustar el preprocesamiento. La API no lo entrena en cada petición: lee el JSON generado previamente.

El análisis narrativo está en `notebooks/Regresion_Logistica_Internet_LaPaz.ipynb`. Para ejecutarlo por completo:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 notebooks/Regresion_Logistica_Internet_LaPaz.ipynb
```

Los resultados se sirven mediante `GET /api/mineria/regresion-logistica`. La sección **Regresión Logística** del dashboard consume ese endpoint y muestra métricas, matriz de confusión, curva ROC, train frente a test, coeficientes y Odds Ratios sin cifras hardcodeadas.

## Árbol de Decisión — Clasificación

H3_2 predice el mismo target binario `TIENE_ACCESO_INTERNET` de H3_1 y reutiliza directamente su universo TIC, construcción del target, exclusión del código 9, limpieza de variables y partición estratificada. Los predictores son los mismos: `urbrur`, `v01_tipoviv`, `v09_energia`, `v19c_compu`, `v19d_celular`, `v13_habitac` y `tot_pers`. Ninguna variable de acceso a Internet se usa como predictor.

El pipeline aplica imputación por moda y one-hot encoding a categóricas, imputación por mediana a numéricas y `DecisionTreeClassifier(random_state=777)`, sin escalado. `GridSearchCV` usa cinco folds, `roc_auc`, `n_jobs=-1` y exclusivamente el conjunto de entrenamiento. Debido al tamaño censal, la búsqueda conserva todos los valores solicitados pero se ejecuta en dos etapas: profundidades 1–15 y luego el cruce de `min_samples_split` 2–9 con `min_samples_leaf` 1–5 en la mejor profundidad. No se reduce ni muestrea el dataset.

Entrenar y regenerar artefactos:

```bash
python scripts/entrenar_arbol_clasificacion.py
# Para releer el CSV nacional en lugar del cache departamental:
python scripts/entrenar_arbol_clasificacion.py --forzar-fuente
```

Se calculan accuracy, precision, recall, F1 y ROC-AUC en train y test, matriz de confusión, reporte de clasificación y curva ROC. Además del umbral estándar 0,5, se informa un umbral que maximiza el índice de Youden, seleccionado con probabilidades out-of-fold de train. Las importancias se guardan por feature one-hot y agregadas por variable original.

Artefactos principales:

- `outputs/modelado/metricas_arbol_clasificacion.json`;
- `outputs/modelado/matriz_confusion_arbol_test.csv`;
- `outputs/modelado/importancia_variables_arbol.csv`;
- `outputs/modelado/importancia_variables_arbol_agregada.csv`;
- `outputs/modelado/predicciones_arbol_test.csv.gz`;
- `outputs/modelado/gridsearch_arbol_clasificacion.csv`;
- `outputs/modelado/comparacion_modelos_clasificacion.csv`;
- `outputs/modelos/arbol_clasificacion.joblib`;
- `outputs/graficos/arbol_clasificacion_niveles_0_3.png`.

El notebook académico es `notebooks/H3_2_Arbol_Clasificacion_Internet_LaPaz.ipynb`; importa `src.decision_tree_classification` y consume resultados reales, sin hardcodear métricas. La API los sirve mediante `GET /api/mineria/arbol-clasificacion`.

## Árbol de Decisión — Regresión

H3_3 predice el **porcentaje de viviendas con acceso a Internet de cada municipio de La Paz** a partir de características censales agregadas. Una observación es un municipio, frente a las viviendas individuales de H3_1/H3_2. Es un problema de regresión con `DecisionTreeRegressor`, target continuo `PORCENTAJE_ACCESO_INTERNET` entre 0 y 100; no se compara directamente con métricas de clasificación.

El target conserva exactamente `pct_algun` del EDA: `v19e_f = 1 / universo TIC municipal × 100`, incluyendo respuestas sin especificar en el denominador. Se reutilizan la lectura y el diccionario de `src.pipeline`, la agregación de `src.mining` y la limpieza de H3_1/H3_2. El flujo se detiene si los códigos, nombres, denominadores o porcentajes discrepan de `outputs/tablas/municipios.csv` (tolerancia absoluta de 1e-8 puntos porcentuales). Los municipios se ordenan por código y no se duplican ni se generan observaciones sintéticas.

Predictores comunes y dos alternativas de habitaciones (seis variables por variante):

| Variable | Construcción municipal |
|---|---|
| `pct_urbano` | Porcentaje urbano sobre las respuestas de área válidas del universo TIC |
| `pct_con_energia` | Porcentaje con electricidad; diccionario oficial: disponibilidad 1–4, ausencia 5 |
| `pct_computadora` | Porcentaje con computadora/laptop/tablet entre respuestas determinadas |
| `pct_celular` | Porcentaje con teléfono celular entre respuestas determinadas |
| `promedio_habitaciones` (A) | Media de códigos válidos 1–8; 8 significa ocho o más |
| `pct_3_o_mas_habitaciones` (B) | Códigos 3–8 / respuestas válidas 1–8 × 100 dentro del universo TIC; los inválidos quedan ausentes |
| `promedio_personas` | Media de valores válidos 0–9999 |

Los predictores se agregan sobre el mismo universo TIC, sin filtrar por acceso a Internet. Computadora y celular conservan la limpieza existente: 9 y vacíos son ausentes; no se convierten en No. Sus tasas utilizan respuestas determinadas por variable. Las medias excluyen valores inválidos. Se guardan conteos válidos por municipio y faltantes municipales en el JSON para auditar los denominadores. No se imputan microdatos antes de agregar.

**Prevención de leakage:** una lista cerrada selecciona solo los seis predictores. Se excluyen Internet fijo/móvil/combinado, sus agregados, el target y los identificadores `municipio_codigo`/`municipio`. La imputación municipal por mediana está dentro del pipeline y se aprende exclusivamente de train o del subconjunto train de cada fold. No se aplica escalado.

```bash
python scripts/entrenar_arbol_regresion.py
# Releer los microdatos originales en bloques:
python scripts/entrenar_arbol_regresion.py --forzar-fuente
```

La partición reproducible usa `random_state=777` y `test_size=0.20`, **sin stratify**: se mantiene la convención del proyecto frente a 22 y 70/30 del [notebook oficial del docente](https://github.com/ealaurel/MINERIA_DATOS_2026_2/blob/main/h3_3_Arboles_de_decisi%C3%B3n_regresion.ipynb). Se guardan explícitamente los municipios train/test. Antes de ajustar se informan cantidad de municipios, partición, duplicados, NaN y estadísticos del target.

`GridSearchCV` explora el producto cartesiano completo: profundidades 1–19, `min_samples_split` 2–9 y `min_samples_leaf` 1–4: **608 configuraciones, 3.040 ajustes CV y un reajuste por variante** (6.080 ajustes CV y dos reajustes en total). Usa un mismo `KFold(n_splits=5, shuffle=True, random_state=777)` para A/B, `scoring="neg_mean_squared_error"`, `n_jobs=-1` y solo train. Shuffle evita que los folds dependan del orden municipal y conserva reproducibilidad; no resuelve dependencia espacial.

Se comparan A (promedio de habitaciones) y B (porcentaje con tres o más) sobre el mismo dataset, índices train/test y folds. Se selecciona el menor MSE medio CV; si `|MSE_A - MSE_B| <= 1e-8 + 1e-6 × min(MSE_A, MSE_B)` pp², se prefiere B por interpretar correctamente la categoría abierta 8. **La selección no recibe test**. Después se evalúa una vez el árbol elegido en los 18 municipios test. Las constantes `FEATURE_SET_PROMEDIO_HABITACIONES` y `FEATURE_SET_PCT_HABITACIONES` conservan las alternativas; `SELECTED_FEATURE_SET` declara la política dinámica. `MODEL_FEATURES` mantiene A para compatibilidad al preparar X, y las features finales se leen de `seleccion_features.variables` o `pipeline.feature_names_in_`.

Se conserva el `best_estimator_` elegido, los resultados completos de ambas búsquedas (columna `variante`), parámetros, MSE CV, desviación y MSE por fold. `comparacion_features_habitaciones.csv` documenta ambas variantes y la elección sin métricas test; `folds_cv_arbol_regresion.csv` corresponde al árbol elegido.

Antes de preparar H3_3 se recalculan los SHA-256 y tamaños de los dos originales con `src.pipeline.sha256`. Se contrastan con `outputs/resumen_eda.json` y con el manifiesto previo cuando exista; un cambio detiene el flujo sin sobrescribir la referencia. No se descargan archivos. El manifiesto conserva rutas relativas para permitir trasladar el repositorio. Puede ejecutarse por separado:

```bash
python scripts/verificar_fuentes_censo.py
```

`reglas_arbol_regresion.txt` se genera con `export_text` del mismo `best_estimator_`, usando los nombres seleccionados y el árbol completo. Los umbrales del texto se redondean a seis decimales; `explicar_prediccion(pipeline, municipio)` permite consultar la ruta real con nodo, feature, umbral, valor imputado, condición y predicción final, sin ajustar otro modelo.

Se calculan **MSE, RMSE, MAE y R²** en train/test. MAE y RMSE se interpretan en puntos porcentuales, MSE en pp²; R² puede ser negativo y se conserva. `DummyRegressor(strategy="mean")` aprende únicamente la media de train y se compara en el mismo test. Se extraen importancias reales sin interpretarlas como efectos causales, estructura del árbol, predicciones identificadas y residuos (`real − predicho`). El árbol base se evalúa solo en train para ilustrar el problema.

Artefactos agregados y modelo:

- `outputs/modelado/dataset_regresion_municipal.csv`;
- `outputs/modelado/metricas_arbol_regresion.json`;
- `outputs/modelado/predicciones_arbol_regresion_test.csv`;
- `outputs/modelado/importancia_variables_arbol_regresion.csv`;
- `outputs/modelado/gridsearch_arbol_regresion.csv`;
- `outputs/modelado/comparacion_baseline_arbol_regresion.csv`;
- `outputs/modelado/particion_arbol_regresion.csv`;
- `outputs/modelado/comparacion_features_habitaciones.csv`;
- `outputs/modelado/folds_cv_arbol_regresion.csv`;
- `outputs/modelado/fuentes_censo_sha256.json`;
- `outputs/modelado/reglas_arbol_regresion.txt`;
- `outputs/modelos/arbol_regresion.joblib`;
- `outputs/graficos/arbol_regresion_niveles_0_3.png`;
- `outputs/graficos/regresion_real_vs_predicho.png`;
- `outputs/graficos/residuos_arbol_regresion.png`.

El notebook `notebooks/H3_3_Arbol_Regresion_Internet_LaPaz.ipynb` reutiliza `src.decision_tree_regression`, ejecuta el flujo y narra resultados reales sin hardcodear cifras:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 notebooks/H3_3_Arbol_Regresion_Internet_LaPaz.ipynb
python -m pytest -q tests
```

`GET /api/mineria/arbol-regresion` lee exclusivamente el JSON previamente calculado; no carga microdatos ni entrena durante una request. Si falta, devuelve 503. El dashboard añade **Árbol de Decisión — Regresión** bajo **Predicción territorial**, con métricas, municipios, hiperparámetros, estructura, importancia, Real vs. Predicho, comparación con baseline y errores municipales. Su carga es independiente, de modo que un fallo de H3_3 no bloquea H3_1/H3_2.

**Limitaciones:** hay muchas menos observaciones que en los modelos individuales. Cada municipio pesa una vez, sin ponderar viviendas, y la evaluación depende de una única partición pequeña; la desviación CV no es un intervalo de confianza. No se modela dependencia espacial. La media de habitaciones A aproxima la categoría ocho o más; B evita esa aproximación pero agrupa la cantidad en dos categorías. La selección A/B puede volver optimista el mejor MSE CV; el test reservado es la evaluación externa a esa selección. Las tasas de equipamiento dependen de respuestas determinadas. Son asociaciones territoriales de acceso declarado en 2024, sin inferencia causal ni individual (riesgo de falacia ecológica). No se garantiza generalización a otros departamentos, años, países ni condiciones futuras.

## Backend

Iniciar la API:

```bash
cd backend
source ../.venv/bin/activate
uvicorn app.main:app --reload
```

Direcciones locales:

- API: `http://localhost:8000/api`
- Swagger/OpenAPI: `http://localhost:8000/docs`
- Esquema OpenAPI: `http://localhost:8000/openapi.json`

Configuración opcional:

- `CENSO_OUTPUTS_DIR`: ruta alternativa a `outputs/`.
- `CORS_ORIGINS`: orígenes locales separados por comas.

Por defecto CORS permite únicamente `http://localhost:5173` y `http://127.0.0.1:5173`.

## Endpoints

| Método | Ruta | Datos procesados utilizados |
|---|---|---|
| GET | `/api/health` | Estado del proceso |
| GET | `/api/resumen` | `outputs/resumen_eda.json` |
| GET | `/api/conectividad/area` | Resumen por área del JSON procesado |
| GET | `/api/conectividad/municipios` | `outputs/tablas/municipios.csv` o `municipios_area.csv` |
| GET | `/api/conectividad/provincias` | `outputs/tablas/provincias.csv` |
| GET | `/api/mineria/outliers` | Outliers y límites IQR del JSON procesado |
| GET | `/api/mineria/distribucion` | Tasas municipales procesadas |
| GET | `/api/mineria/regresion-logistica` | Resultados de la primera iteración guardados en `outputs/modelado/` |
| GET | `/api/mineria/arbol-clasificacion` | Resultados H3_2, GridSearchCV, umbral ROC, importancias y comparación guardados |
| GET | `/api/mineria/arbol-regresion` | Resultados H3_3 municipales: errores, CV, baseline, importancia y predicciones guardados |
| GET | `/api/calidad` | `faltantes.csv` y auditorías del JSON |
| GET | `/api/metadata` | Diccionario utilizado, fuente y proceso |
| GET | `/api/hallazgos` | Hallazgo, hipótesis y conclusiones generados por el pipeline |

Parámetros de municipios:

```text
?limit=10&orden=mayor&metrica=internet&area=todos
```

- `orden`: `mayor` o `menor`.
- `metrica`: `internet`, `fijo`, `movil` o `sin_internet`.
- `area`: `todos`, `urbana` o `rural`.

Provincias admite `limit`, `orden` y `metrica`.

## Frontend

Crear la configuración local y ejecutar Vite:

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

El valor predeterminado es:

```env
VITE_API_URL=http://localhost:8000
```

El dashboard queda disponible en `http://localhost:5173`.

La página incluye:

- cinco KPI derivados de `/api/resumen`;
- barras de Internet fijo, móvil, combinado y sin Internet;
- comparación urbano/rural;
- top 10 y bottom 10 municipales;
- distribución municipal por intervalos;
- territorios atípicos según la regla de Tukey;
- hallazgo, hipótesis y conclusiones del análisis;
- filtros por área y métrica;
- sección de Regresión Logística con métricas, matriz de confusión, curva ROC, Odds Ratios y comparación train/test;
- sección **Árbol de Decisión — Clasificación** con métricas, hiperparámetros, estructura, matriz, ROC e importancias agregadas;
- comparación visual de Regresión Logística y Árbol de Decisión con los mismos casos de test;
- sección **Predicción territorial — H3_3** con MAE/RMSE/MSE/R², Real vs. Predicho, baseline, importancia y tabla municipal;
- separación entre clasificación individual (H3_1/H3_2) y regresión territorial (H3_3);
- estados de carga, error y ausencia de datos;
- diseño adaptable a escritorio, tableta y móvil.

## Tests y compilación

Backend:

```bash
cd backend
source ../.venv/bin/activate
pytest -q
```

Frontend:

```bash
cd frontend
npm run build
```

## Estructura del proyecto

```text
Mineria-de-Datos-Internet/
├── backend/
│   ├── app/
│   │   ├── api/routes/
│   │   ├── core/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── main.py
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── types/
│   │   └── utils/
│   ├── .env.example
│   ├── package.json
│   └── vite.config.ts
├── src/
│   ├── mining.py
│   ├── logistic_regression.py
│   ├── decision_tree_classification.py
│   ├── decision_tree_regression.py
│   ├── census_sources.py
│   └── pipeline.py
├── scripts/
│   ├── entrenar_regresion_logistica.py
│   ├── entrenar_arbol_clasificacion.py
│   ├── entrenar_arbol_regresion.py
│   └── verificar_fuentes_censo.py
├── notebooks/
│   ├── EDA_Internet_LaPaz.ipynb
│   ├── Regresion_Logistica_Internet_LaPaz.ipynb
│   ├── H3_2_Arbol_Clasificacion_Internet_LaPaz.ipynb
│   └── H3_3_Arbol_Regresion_Internet_LaPaz.ipynb
├── outputs/
│   ├── datos/
│   ├── graficos/
│   ├── modelado/
│   ├── modelos/
│   ├── tablas/
│   └── resumen_eda.json
├── Base de datos CSV/
├── README.md
└── requirements.txt
```

## Criterios estadísticos y limitaciones

- `1 = Sí`, `2 = No` y `9 = Sin especificar` para los indicadores TIC.
- La tasa principal usa `Sí / universo aplicable`; conserva “Sin especificar” en el denominador.
- Los nombres de municipio y provincia se asignan únicamente cuando el código coincide con el catálogo oficial incluido.
- Los outliers se calculan sobre el porcentaje municipal de algún Internet con la regla de Tukey, `1,5 × IQR`, igual que en el EDA.
- Un outlier es un territorio inusual frente a la distribución; no implica un error.
- La relación urbano/rural es descriptiva y no demuestra causalidad.
- Los modelos identifican asociaciones predictivas; las importancias y coeficientes no representan efectos causales.
- El test se reserva para evaluación final. La optimización y la selección de umbral del árbol usan únicamente train.
- No se mide velocidad, calidad o precio del servicio ni cambios temporales.
