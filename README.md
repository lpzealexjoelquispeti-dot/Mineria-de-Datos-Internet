# Conectividad Digital en La Paz

## Descripción

Aplicación web de solo lectura para visualizar y explorar el análisis del acceso a Internet fijo y móvil en el departamento de La Paz a partir del **Censo de Población y Vivienda 2024**.

El proyecto conserva el notebook y los resultados del EDA, añade pipelines reproducibles de regresión logística y árbol de decisión, expone los artefactos procesados mediante FastAPI y los presenta en un dashboard React. No incluye usuarios, autenticación, administración, pagos ni base de datos transaccional.

La unidad de análisis es el registro de vivienda. El universo TIC comprende viviendas particulares (tipos 1–6) con personas presentes (ocupación 0/1). Los registros fuera de ese universo no se consideran viviendas sin Internet.

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
- scikit-learn para el pipeline predictivo y las métricas de clasificación
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
│   └── pipeline.py
├── scripts/
│   ├── entrenar_regresion_logistica.py
│   └── entrenar_arbol_clasificacion.py
├── notebooks/
│   ├── EDA_Internet_LaPaz.ipynb
│   ├── Regresion_Logistica_Internet_LaPaz.ipynb
│   └── H3_2_Arbol_Clasificacion_Internet_LaPaz.ipynb
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
