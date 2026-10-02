# H3_3: árbol de regresión de conectividad municipal

Responsable: Cristopher Iori Lazcano Gutierrez. Materia: Minería de Datos,
UNIFRANZ, gestión 2026-2. Docente: Enrique Alejandro Laurel Cossio.

## Qué se hizo

Se adaptó la práctica de árboles de regresión a los datos del proyecto de acceso
a Internet en La Paz. La variable objetivo es el porcentaje municipal de
viviendas con acceso declarado a Internet, con escala de 0 a 100. La unidad de
análisis de H3_3 es el municipio. Los modelos de H3_1 y H3_2 siguen clasificando
viviendas según tengan o no Internet.

Se descargaron las fuentes oficiales del INE y se verificaron mediante SHA-256.
Vivienda, diccionario y cuestionario coinciden con los archivos que utilizó el
grupo. Se aplicaron los mismos filtros del universo TIC y se comprobaron las
tasas y denominadores contra el EDA ya publicado. Las respuestas «Sin
especificar» siguen en el denominador de la tasa principal.

Se agregaron seis predictores municipales: porcentaje de viviendas urbanas, con
computadora, con celular, con electricidad de red pública, con tres o más
habitaciones y promedio de personas. La categoría ocho o más habitaciones se
interpreta como categoría abierta, sin calcular un promedio exacto de cuartos.
Las variables de Internet y sus derivados se excluyeron de los predictores.

## Archivos propios

- `notebooks/H3_3_Arbol_Regresion_Internet_LaPaz.ipynb`: notebook ejecutado, tablas,
  gráficos, interpretación, ruta de decisión y conclusiones.
- `src/decision_tree_regression.py`: agregación municipal, entrenamiento,
  evaluación, explicación y exportación.
- `scripts/entrenar_arbol_regresion.py`: ejecución desde terminal.
- `scripts/descargar_datos_censo.py`: descarga oficial y verificación de fuentes.
- `outputs/h3_3/`: tabla de 87 municipios, calidad, partición, grilla,
  comparación, predicciones, importancias, reglas completas, cuatro gráficos,
  métricas y pipeline entrenado.
- `tests/test_decision_tree_regression.py`: comprobaciones de denominadores,
  geografía y separación del entrenamiento.

Los notebooks, el código, la presentación y los artefactos anteriores del grupo
se conservaron. Las salidas nuevas tienen carpeta propia. Los originales
censales y su ZIP quedan excluidos de Git.

## Cómo reproducir

Desde la raíz del repositorio, con Python 3.12:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/descargar_datos_censo.py
python scripts/entrenar_arbol_regresion.py
```

También se puede abrir el notebook y ejecutar todas las celdas. Para guardar de
nuevo el notebook con sus salidas:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/H3_3_Arbol_Regresion_Internet_LaPaz.ipynb
```

La primera carga recorre Vivienda en bloques. Las siguientes reutilizan la copia
departamental local. `--forzar-fuente` vuelve a leer el CSV original. La grilla
explora 608 combinaciones con validación cruzada de cinco particiones solo sobre
los 60 municipios de entrenamiento. Los 27 de prueba quedan reservados para la
evaluación final. La semilla es 777.

## Resultados de la ejecución

La fuente contiene 1.334.496 registros de La Paz, de los cuales 1.069.838
pertenecen al universo TIC. Se agregaron 87 municipios. El acceso departamental
fue 69,95 %. Los predictores municipales no tienen valores faltantes en esta
ejecución.

| Modelo | MAE de prueba (pp) | RMSE de prueba (pp) | MSE de prueba (pp²) | R² de prueba |
| --- | ---: | ---: | ---: | ---: |
| Promedio de entrenamiento | 9,472 | 11,847 | 140,357 | −0,650 |
| Árbol inicial, profundidad 2 | 5,101 | 6,297 | 39,647 | 0,534 |
| Árbol ajustado por validación cruzada | 4,692 | 6,049 | 36,590 | 0,570 |

El ajuste elegido por MSE de validación utiliza `max_depth=7`,
`min_samples_split=7` y `min_samples_leaf=1`; tiene 17 hojas. Su MSE medio de
validación fue 51,166 pp². El predictor con mayor importancia fue la proporción
de viviendas con teléfono celular, con 89,58 % de la reducción de error
atribuida por el árbol. Esto expresa una asociación municipal, sin demostrar
causalidad.

El MAE del árbol ajustado se redujo aproximadamente 50,5 % frente a predecir el
promedio de entrenamiento. Esta cifra describe una mejora predictiva en la
partición usada, sin indicar un aumento real de conectividad. El MAE de
entrenamiento fue 1,591 pp, frente a 4,692 pp en prueba: la diferencia advierte
posible sobreajuste. Con 87 municipios, el resultado es sensible a la partición
y requiere validación externa antes de un uso operativo.

## Explicación breve para exposición

«Nuestro proyecto estudia el acceso a Internet en La Paz. En H3_3 cambiamos la
unidad del modelo a municipios para estimar una tasa numérica. Usamos las mismas
viviendas aplicables y el mismo denominador del EDA. Entrenamos con 60 municipios
y reservamos 27 para prueba. El árbol ajustado obtuvo un error absoluto medio de
4,69 puntos porcentuales; la referencia que predice el promedio obtuvo 9,47.
La tenencia de celular fue la variable más utilizada por el árbol. Hay diferencia
entre el error de entrenamiento y prueba, así que todavía existe riesgo de
sobreajuste. Son asociaciones del Censo 2024, sin evidencia causal ni de cambios
futuros.»

## Relación con la consigna del docente

El ejemplo H3_3 enseña regresión con `insurance.csv`; aquí se aplicó el mismo tipo
de modelo al proyecto propio. El repositorio del docente no publica una entrega
independiente para ese notebook.

La evaluación grupal publicada el 30 de septiembre exige diez respuestas con
evidencia hasta modelado de CRISP-DM y resultados de clasificación. H3_1 y H3_2
conservan esa evidencia. H3_3 es una práctica complementaria y sus métricas
numéricas no sustituyen accuracy, F1 y matriz de confusión del proyecto de
clasificación. La rúbrica anterior de dos diapositivas y tres minutos corresponde
a la presentación del EDA.

## Rama de trabajo

El desarrollo está en `rama-cris`, que incorpora como base los avances publicados
del grupo. Subir `rama-cris` actualiza exclusivamente esa rama. Para incorporarlo
a la rama principal hay que revisar y fusionar un pull request hacia la rama que
el grupo defina. La rama predeterminada del repositorio es actualmente
`rama-alex`; `git push` no fusiona automáticamente con ella ni con `main`.

## Fuentes

- [INE: descarga oficial CSV del Censo 2024](https://cpv2024.ine.gob.bo/index.php/principal/descargas/).
- [Docente: H3_3, árboles de decisión para regresión](https://github.com/ealaurel/MINERIA_DATOS_2026_2/blob/main/h3_3_Arboles_de_decisi%C3%B3n_regresion.ipynb).
- [Docente: evaluación grupal del proyecto](https://github.com/ealaurel/MINERIA_DATOS_2026_2/blob/main/evaluacion1_grupal_proyecto_crisp_dm.md).
