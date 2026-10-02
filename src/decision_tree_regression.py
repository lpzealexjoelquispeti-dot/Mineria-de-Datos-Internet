"""H3_3: regresión del porcentaje municipal de acceso a Internet en La Paz.

Reutiliza las fuentes, el universo TIC y la agregación del proyecto. Cada
observación del modelo es un municipio; H3_1 y H3_2 mantienen su análisis
de clasificación por vivienda. Los artefactos de H3_3 tienen carpeta propia.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

from src.mining import summarize_groups
from src.pipeline import load_lapaz, read_dictionary, sha256

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TARGET = "pct_algun"
RANDOM_STATE = 777
TEST_SIZE = 0.30
FEATURE_LABELS = {
    "pct_urbanas": "Viviendas urbanas (%)",
    "pct_computadora": "Con computadora, laptop o tablet (%)",
    "pct_celular": "Con teléfono celular (%)",
    "pct_red_electrica": "Con electricidad de red pública (%)",
    "pct_3_o_mas_habitaciones": "Con 3 o más habitaciones (%)",
    "promedio_personas": "Personas por vivienda (promedio)",
}
FEATURES = list(FEATURE_LABELS)
SOURCE_VARIABLES = ["urbrur", "v19c_compu", "v19d_celular", "v09_energia", "v13_habitac", "tot_pers"]
SOURCE_PAGE = "https://cpv2024.ine.gob.bo/index.php/principal/descargas/"
TEACHER_NOTEBOOK = "https://github.com/ealaurel/MINERIA_DATOS_2026_2/blob/main/h3_3_Arboles_de_decisi%C3%B3n_regresion.ipynb"


@dataclass
class MunicipalDataset:
    table: pd.DataFrame
    quality: pd.DataFrame
    audit: dict[str, Any]


def agregar_municipios(data: pd.DataFrame, catalog: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agrega las viviendas aplicables sin convertir no respuesta en ausencia."""
    applicable = data["v01_tipoviv"].isin(["1", "2", "3", "4", "5", "6"]) & data["v02_condocup"].isin(["0", "1"])
    universe = data.loc[applicable].copy()
    if universe.empty:
        raise ValueError("No hay viviendas dentro del universo TIC.")
    if not universe["v19e_f"].isin(["1", "2", "9"]).all():
        raise ValueError("El target censal contiene códigos vacíos o fuera de catálogo.")
    universe["municipio_codigo"] = universe["idep"].str.lstrip("0") + universe["iprov"] + universe["imun"]
    universe["municipio"] = universe["municipio_codigo"].map(catalog)
    if universe["municipio"].isna().any():
        raise ValueError("Hay códigos municipales sin correspondencia en el diccionario.")

    counts = summarize_groups(universe, ["municipio_codigo", "municipio"])
    table = counts[["universo", "algun_si", "algun_no", "algun_sin_especificar", TARGET, "pct_sin_especificar"]].copy()
    grouped = universe.groupby(["municipio_codigo", "municipio"], observed=True)
    binary = {
        "pct_urbanas": ("urbrur", "1"),
        "pct_computadora": ("v19c_compu", "1"),
        "pct_celular": ("v19d_celular", "1"),
        "pct_red_electrica": ("v09_energia", "1"),
    }
    for output, (source, code) in binary.items():
        table[output] = grouped[source].agg(lambda values: values.eq(code).sum()) / table["universo"] * 100

    rooms = pd.to_numeric(universe["v13_habitac"], errors="coerce")
    # 8 significa 8 o más; el indicador evita tratar esa categoría como promedio exacto.
    universe["tres_o_mas"] = rooms.between(3, 8)
    people = pd.to_numeric(universe["tot_pers"], errors="coerce")
    universe["personas_validas"] = people.where(people.between(1, 9999))
    grouped = universe.groupby(["municipio_codigo", "municipio"], observed=True)
    table["pct_3_o_mas_habitaciones"] = grouped["tres_o_mas"].sum() / table["universo"] * 100
    table["promedio_personas"] = grouped["personas_validas"].mean()

    quality = []
    valid_codes = {
        "urbrur": {"1", "2"}, "v19c_compu": {"1", "2", "9"},
        "v19d_celular": {"1", "2", "9"}, "v09_energia": {"1", "2", "3", "4", "5"},
        "v13_habitac": {str(value) for value in range(1, 9)},
    }
    for variable, codes in valid_codes.items():
        values = universe[variable]
        invalid = ~values.isin(codes)
        if invalid.any():
            raise ValueError(f"Hay {int(invalid.sum())} códigos fuera de catálogo en {variable}.")
        quality.append({"variable": variable, "registros": len(values),
                        "sin_especificar": int(values.eq("9").sum()),
                        "invalidos_o_vacios": int(invalid.sum())})
    quality.append({"variable": "tot_pers", "registros": len(people), "sin_especificar": 0,
                    "invalidos_o_vacios": int(universe["personas_validas"].isna().sum())})
    table = table.reset_index().sort_values("municipio_codigo").reset_index(drop=True)
    if not table[TARGET].between(0, 100).all() or table["municipio_codigo"].duplicated().any():
        raise ValueError("La tabla municipal tiene tasas inválidas o códigos duplicados.")
    return table, pd.DataFrame(quality)


def preparar_dataset(root: Path, prefer_cache: bool = True) -> MunicipalDataset:
    """Carga las mismas fuentes del grupo y reconcilia las tasas con su EDA."""
    source_dir = root / "Base de datos CSV"
    source = source_dir / "Vivienda_CPV-2024.csv"
    dictionary_path = source_dir / "Diccionario de variables CPV 2024.xlsx"
    expected = json.loads((root / "outputs/resumen_eda.json").read_text(encoding="utf-8"))
    sources = {}
    for path in (source, dictionary_path):
        actual = sha256(path)
        if actual != expected["fuentes"][path.name]["sha256"]:
            raise ValueError(f"{path.name} difiere de la fuente utilizada por el grupo.")
        sources[path.name] = {"sha256": actual, "bytes": path.stat().st_size}
    dictionary = read_dictionary(dictionary_path)
    codes = [code for code, label in dictionary["dep_res_cod"]["categorias"].items() if label == "La Paz"]
    if len(codes) != 1:
        raise ValueError("No se identifica unívocamente La Paz en el diccionario.")
    cached = root / "outputs/datos/vivienda_lapaz_seleccion.csv.gz"
    data, national, _ = load_lapaz(source, cached, codes[0].zfill(2), prefer_cache and cached.is_file())
    table, quality = agregar_municipios(data, dictionary["mun_res_cod"]["categorias"])
    original = pd.read_csv(root / "outputs/tablas/municipios.csv", dtype={"municipio_codigo": str})
    check = table.merge(original[["municipio_codigo", "universo", TARGET]], on="municipio_codigo", suffixes=("", "_eda"), validate="one_to_one", how="outer")
    if len(check) != len(table) or check[["universo_eda", f"{TARGET}_eda"]].isna().any().any():
        raise ValueError("La tabla municipal no concilia con los municipios del EDA.")
    if not np.array_equal(check["universo"], check["universo_eda"]) or not np.allclose(check[TARGET], check[f"{TARGET}_eda"], atol=1e-10, rtol=0):
        raise ValueError("Los denominadores o las tasas difieren del EDA del grupo.")
    audit = {
        "fuente": SOURCE_PAGE, "fuentes_verificadas": sources,
        "registros_nacionales": int(national), "registros_lapaz": len(data),
        "universo_tic": int(table["universo"].sum()), "municipios": len(table),
        "target_sin_especificar": int(table["algun_sin_especificar"].sum()),
        "tasa_departamental_pct": float(table["algun_si"].sum() / table["universo"].sum() * 100),
        "tasas_y_denominadores_coinciden_con_eda": True,
        "faltantes_municipales": {col: int(table[col].isna().sum()) for col in FEATURES},
    }
    return MunicipalDataset(table=table, quality=quality, audit=audit)


def dividir_municipios(table: pd.DataFrame):
    """Reserva municipios completos, con semilla y sin estratificar una tasa continua."""
    X, y = table[FEATURES], table[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def crear_pipeline(**parameters: Any) -> Pipeline:
    return Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("modelo", DecisionTreeRegressor(random_state=RANDOM_STATE, **parameters)),
    ])


def buscar_modelo(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    grid = {"modelo__max_depth": list(range(1, 20)),
            "modelo__min_samples_split": list(range(2, 10)),
            "modelo__min_samples_leaf": list(range(1, 5))}
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(crear_pipeline(), grid, cv=cv, scoring="neg_mean_squared_error",
                          refit=True, n_jobs=1, return_train_score=True, error_score="raise")
    search.fit(X_train, y_train)
    return search


def evaluar(model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    return {"mse_pp2": float(mse), "rmse_pp": float(np.sqrt(mse)),
            "mae_pp": float(mean_absolute_error(y, predictions)), "r2": float(r2_score(y, predictions))}


def explicar_prediccion(model: Pipeline, row: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Recorre las condiciones exactas de la predicción de un municipio."""
    if len(row) != 1:
        raise ValueError("Seleccione exactamente un municipio para explicar la ruta.")
    values = model.named_steps["imputador"].transform(row[FEATURES])
    tree = model.named_steps["modelo"]
    leaf = int(tree.apply(values)[0])
    path = tree.decision_path(values).indices
    steps = []
    for node in path:
        if node == leaf:
            continue
        feature = int(tree.tree_.feature[node])
        threshold = float(tree.tree_.threshold[node])
        value = float(values[0, feature])
        steps.append({"variable": FEATURE_LABELS[FEATURES[feature]], "valor_municipal": value,
                      "condicion": "≤" if value <= threshold else ">", "umbral": threshold})
    return pd.DataFrame(steps), float(tree.predict(values)[0])


def entrenar(dataset: MunicipalDataset) -> dict[str, Any]:
    X_train, X_test, y_train, y_test = dividir_municipios(dataset.table)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    initial = crear_pipeline(max_depth=2).fit(X_train, y_train)
    search = buscar_modelo(X_train, y_train)
    final = search.best_estimator_
    # La selección queda cerrada antes de calcular cualquiera de las métricas de prueba.
    models = {"Promedio de entrenamiento": baseline, "Árbol inicial (profundidad 2)": initial, "Árbol ajustado por CV": final}
    comparison = []
    for name, model in models.items():
        cv_mse = -cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=cv)
        row = {"modelo": name, "mse_cv_pp2": float(cv_mse.mean()), "desviacion_mse_cv_pp2": float(cv_mse.std())}
        row.update({f"train_{key}": value for key, value in evaluar(model, X_train, y_train).items()})
        row.update({f"test_{key}": value for key, value in evaluar(model, X_test, y_test).items()})
        comparison.append(row)
    metrics = pd.DataFrame(comparison)
    predictions = dataset.table.loc[X_test.index, ["municipio_codigo", "municipio", "universo", TARGET]].copy()
    predictions = predictions.rename(columns={TARGET: "real_pct"})
    predictions["prediccion_pct"] = final.predict(X_test)
    predictions["baseline_pct"] = baseline.predict(X_test)
    predictions["error_pp"] = predictions["prediccion_pct"] - predictions["real_pct"]
    predictions["error_absoluto_pp"] = predictions["error_pp"].abs()
    predictions = predictions.sort_values("error_absoluto_pp", ascending=False)
    tree = final.named_steps["modelo"]
    importance = pd.DataFrame({"variable": FEATURES, "descripcion": list(FEATURE_LABELS.values()), "importancia": tree.feature_importances_}).sort_values("importancia", ascending=False)
    grid_table = pd.DataFrame(search.cv_results_)[["param_modelo__max_depth", "param_modelo__min_samples_split", "param_modelo__min_samples_leaf", "mean_test_score", "std_test_score", "rank_test_score"]].copy()
    grid_table["mse_cv_pp2"] = -grid_table.pop("mean_test_score")
    grid_table = grid_table.sort_values("rank_test_score")
    best = {name.replace("modelo__", ""): int(value) for name, value in search.best_params_.items()}
    split = dataset.table[["municipio_codigo", "municipio", "universo", TARGET]].copy()
    split["conjunto"] = np.where(split.index.isin(X_train.index), "train", "test")
    summary = {
        "objetivo": "Estimar el porcentaje municipal de viviendas con acceso declarado a Internet en La Paz, Censo 2024.",
        "unidad_analisis": "Municipio; cada municipio tiene igual peso en entrenamiento y evaluación.",
        "target": {"nombre": TARGET, "unidad": "porcentaje (escala 0 a 100)",
                   "formula": "100 × viviendas con v19e_f = 1 / universo TIC municipal", "sin_especificar": "Se conserva en el denominador; no se recodifica como No."},
        "fuentes": dataset.audit, "referencia_docente": TEACHER_NOTEBOOK,
        "predictores": FEATURE_LABELS, "particion": {"train": len(X_train), "test": len(X_test), "test_size": TEST_SIZE, "random_state": RANDOM_STATE, "sin_municipios_compartidos": not bool(set(X_train.index) & set(X_test.index))},
        "validacion_cruzada": {"folds": 5, "criterio": "MSE mínimo en train", "combinaciones": len(grid_table), "mejor_mse_cv_pp2": float(-search.best_score_), "mejores_parametros": best},
        "metricas": metrics.to_dict(orient="records"),
        "estructura_arbol": {"profundidad": int(tree.get_depth()), "hojas": int(tree.get_n_leaves()), "nodos": int(tree.tree_.node_count)},
        "variables_importantes": importance.to_dict(orient="records"),
        "versiones": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        "limitaciones": [
            "Solo 87 municipios; las métricas dependen de la partición y no prueban desempeño fuera de La Paz.",
            "Los datos son de 2024: se estima una tasa contemporánea, no una tendencia ni un pronóstico futuro.",
            "Las asociaciones municipales no describen efectos causales ni decisiones individuales de viviendas.",
            "Las tasas conservan Sin especificar en el denominador; no representan el acceso de quienes no respondieron.",
            "Las importancias corresponden a este árbol y pueden favorecer variables con más puntos de corte.",
        ],
    }
    return {"resumen": summary, "modelo": final, "baseline": baseline, "modelo_inicial": initial,
            "busqueda": search, "comparacion": metrics, "predicciones": predictions,
            "importancias": importance, "grid": grid_table, "particion": split,
            "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


def visualizar(result: dict[str, Any], folder: Path) -> dict[str, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "figure.dpi": 120})
    tree = result["modelo"].named_steps["modelo"]
    names = [FEATURE_LABELS[feature].replace(" (%)", "\n(%)") for feature in FEATURES]
    figures = {}
    depth = min(tree.get_depth(), 3)
    fig, ax = plt.subplots(figsize=(22, 11), layout="constrained")
    plot_tree(tree, feature_names=names, filled=True, rounded=True, max_depth=depth, precision=2, fontsize=9, ax=ax)
    ax.set_title("Árbol de regresión: acceso municipal a Internet (%)\n" + ("Vista hasta el nivel 3; reglas completas en el archivo de texto" if tree.get_depth() > 3 else "Cada hoja estima el porcentaje de viviendas con Internet"), pad=18)
    figures["arbol"] = folder / "arbol_regresion.png"
    fig.savefig(figures["arbol"], bbox_inches="tight"); plt.close(fig)
    imp = result["importancias"].sort_values("importancia")
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.barh(imp["descripcion"], imp["importancia"], color="#267f81")
    ax.set(title="Importancia de los predictores municipales", xlabel="Fracción de reducción de error atribuida por el árbol", xlim=(0, 1))
    figures["importancias"] = folder / "importancias_regresion.png"
    fig.savefig(figures["importancias"], bbox_inches="tight"); plt.close(fig)
    predictions = result["predicciones"]
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    ax.scatter(predictions["real_pct"], predictions["prediccion_pct"], color="#267f81", s=45)
    ax.plot([0, 100], [0, 100], linestyle="--", color="#8c4b32", label="Predicción igual al valor real")
    ax.set(title="Municipios de prueba: porcentaje real y estimado", xlabel="Acceso real (%)", ylabel="Acceso estimado (%)", xlim=(0, 100), ylim=(0, 100))
    ax.legend(fontsize=9)
    figures["predicciones"] = folder / "real_vs_estimado.png"
    fig.savefig(figures["predicciones"], bbox_inches="tight"); plt.close(fig)
    comparison = result["comparacion"]
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    x = np.arange(len(comparison))
    ax.bar(x - .18, comparison["train_mae_pp"], width=.36, label="Entrenamiento", color="#82bcbc")
    ax.bar(x + .18, comparison["test_mae_pp"], width=.36, label="Prueba", color="#267f81")
    ax.set_xticks(x, ["Promedio", "Árbol inicial", "Árbol ajustado"])
    ax.set(title="Error absoluto medio por municipio", ylabel="MAE (puntos porcentuales)")
    ax.legend()
    figures["errores"] = folder / "comparacion_mae.png"
    fig.savefig(figures["errores"], bbox_inches="tight"); plt.close(fig)
    return figures


def guardar_resultados(dataset: MunicipalDataset, result: dict[str, Any], root: Path) -> dict[str, Path]:
    folder = root / "outputs/h3_3"
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in {"datos_municipales": dataset.table, "calidad_fuente": dataset.quality,
                        "comparacion_modelos": result["comparacion"], "predicciones_test": result["predicciones"],
                        "importancias": result["importancias"], "gridsearch": result["grid"], "particion": result["particion"]}.items():
        table.to_csv(folder / f"{name}.csv", index=False, encoding="utf-8-sig")
    paths = visualizar(result, folder / "graficos")
    rules = export_text(result["modelo"].named_steps["modelo"], feature_names=list(FEATURE_LABELS.values()), decimals=3, max_depth=20)
    (folder / "reglas_arbol.txt").write_text(rules, encoding="utf-8")
    joblib.dump(result["modelo"], folder / "arbol_regresion.joblib")
    (folder / "metricas.json").write_text(json.dumps(result["resumen"], ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return paths
