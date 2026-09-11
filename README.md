# Minería de Datos: Internet fijo y móvil en La Paz

EDA del acceso declarado a Internet fijo y móvil en el departamento de La Paz, calculado desde los microdatos del **Censo de Población y Vivienda 2024**. El notebook está en español y cubre estructura, calidad, valores atípicos, distribución, relación urbana/rural, hipótesis y conclusiones.

## Fuente y archivos

Coloque los originales, con sus nombres exactos, dentro de `Base de datos CSV/`:

| Archivo | Tamaño aproximado | Uso |
|---|---:|---|
| `Vivienda_CPV-2024.csv` | 490,86 MB | Única fuente de observaciones para todas las estadísticas |
| `Diccionario de variables CPV 2024.xlsx` | 0,20 MB | Descripciones, categorías y catálogos; se lee en cada ejecución |
| `Cuestionario censal 2024.pdf` | 21,07 MB | Preguntas y saltos; páginas PDF 2 y 3 revisadas visualmente |
| `Persona_CPV-2024.csv` | 3.087,56 MB | Solo inspección del encabezado; no se calculan estadísticas ni se cruza |
| `Emigracion_CPV-2024.csv` | 23,81 MB | Solo inspección del encabezado |
| `Mortalidad_CPV-2024.csv` | 19,30 MB | Solo inspección del encabezado |

Los seis archivos suman aproximadamente **3,64 GB** decimales. Los archivos `Zone.Identifier` son metadatos de descarga. El antiguo `La Paz - TIC.xlsx` no participa en el EDA; se conserva el archivo existente.

Se usan los archivos censales proporcionados localmente para el proyecto. Esta entrega no incluye una URL o un manifiesto oficial de descarga con el que certificar su integridad de origen. Las huellas SHA-256 permiten verificar que la ejecución no los modifica.

## Instalación

Use Python 3.12 y cree un entorno desde la raíz del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

En Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Las dependencias incluyen pandas, NumPy, Matplotlib, seaborn, openpyxl (lectura del diccionario) y Jupyter. El notebook no requiere SciPy ni herramientas de PDF: la referencia visual al cuestionario está documentada en su introducción.

## Ejecución

Desde la raíz, con el entorno activado:

```bash
jupyter notebook notebooks/EDA_Internet_LaPaz.ipynb
```

Seleccione el kernel del entorno y ejecute todas las celdas en orden. También puede abrir el notebook en VS Code y usar **Run All**. Las rutas funcionan desde la raíz o desde `notebooks/`; no contienen directorios personales.

Para ejecutar y guardar todas las salidas desde la terminal:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/EDA_Internet_LaPaz.ipynb
```

El CSV utiliza `;`. El notebook detecta el separador, prueba UTF-8 con decodificación estricta y procesa **todo Vivienda** en bloques de 200.000 filas con 16 columnas seleccionadas de las 48 originales. Solo acumula La Paz; no utiliza muestras aleatorias. Después del filtro compacta los códigos repetidos como categorías, conservando sus valores originales. Reserva memoria para el dataset departamental y una copia del universo analítico. El espacio adicional incluye una copia comprimida de las columnas seleccionadas de La Paz.

## Variables y criterios

- `v19e_inetfijo`: Internet fijo en la vivienda.
- `v19f_inetmovil`: Internet móvil (megas o datos).
- `v19e_f`: indicador oficial de Internet fijo en la vivienda o Internet móvil.
- Las tres usan `1 = Sí`, `2 = No`, `9 = Sin especificar`, según el diccionario.
- `urbrur`: `1 = Urbana`, `2 = Rural`.
- `v01_tipoviv`, `v02_condocup`: tipo y ocupación para determinar el universo TIC.
- `v19c_compu`, `v19d_celular`, `v09_energia`, `tot_pers`: computadora/laptop/tablet, teléfono celular, fuente de electricidad y total de personas.
- `v13_habitac`, `v14_dormit`: habitaciones y dormitorios; permiten revisar coherencia. El código 8 agrupa ocho o más.
- `idep`, `iprov`, `imun`: componentes geográficos contrastados con los catálogos de la hoja PERSONA del diccionario. La Paz corresponde a `2`, representado como `02` en el CSV.
- `i00`: identificador candidato, sin definición directa en el diccionario. Se verifica su unicidad en La Paz y no se eliminan filas.

El diccionario no define directamente las cuatro últimas columnas. El notebook documenta esta limitación, valida la correspondencia de los códigos geográficos completos antes de asignar nombres y presenta las filas de origen de los catálogos. Consultar la hoja PERSONA del diccionario **no implica usar microdatos de personas**.

La unidad es el registro de vivienda; la pregunta TIC se refiere al equipamiento del hogar. El universo aplicable comprende viviendas particulares (tipos 1–6) con personas presentes (ocupación 0/1; el código 0 indica sin jefe). Los vacíos fuera del universo no significan ausencia de Internet.

La tasa principal usa **Sí / universo aplicable**, conservando «Sin especificar» en el denominador. Cada tabla ofrece también el número de respuestas determinadas (Sí/No) y la tasa entre ellas. Esto permite comparar fijo y móvil sobre la misma base y evaluar la sensibilidad a la no respuesta.

Se mantiene el indicador combinado oficial. Las diferencias frente a una regla lógica conservadora se exponen sin corregir las fuentes. Las asociaciones y los valores atípicos son descriptivos; no prueban causalidad ni errores del dato.

## Resultados y archivos generados

```text
notebooks/EDA_Internet_LaPaz.ipynb     Notebook ejecutado, tablas, interpretación y gráficos
outputs/resumen_eda.json             Resultados y huellas de las fuentes utilizadas
outputs/integridad_originales.json   Verificación de los seis originales al rehacer el EDA
outputs/tablas/                      Tablas agregadas, catálogos y auditorías
outputs/graficos/                    Nueve gráficos PNG
outputs/datos/                      Copia comprimida local de La Paz (excluida de Git)
```

El notebook comprueba totales, denominadores, códigos, geografía, duplicados y existencia de los nueve gráficos. Verifica al finalizar que Vivienda, el diccionario y el cuestionario conservan sus huellas SHA-256. Cada ejecución regenera las salidas a partir de los originales.

## Git y microdatos grandes

`Base de datos CSV/` no estaba versionada y queda excluida mediante `.gitignore`, al igual que `outputs/datos/`, `.venv/`, `__pycache__/`, `.ipynb_checkpoints/` y `*.pyc`. Tras clonar, los integrantes deben colocar localmente la carpeta censal: Git no la descarga. No fuerce su inclusión con `git add -f`.

Las tablas agregadas y gráficos sí son archivos pequeños revisables. No se requiere commit, push ni Pull Request para ejecutar el análisis.
