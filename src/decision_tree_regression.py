"""H3_3: predicción territorial sobre el universo TIC municipal del EDA.

No se imputa antes de agregar: tasas de predictores entre respuestas determinadas,
medias entre valores válidos. La imputación municipal se ajusta dentro de train/CV.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, plot_tree

from src.logistic_regression import seleccionar_variables as limpiar_microdatos
from src.mining import summarize_groups
from src.pipeline import load_lapaz, read_dictionary

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TARGET = "PORCENTAJE_ACCESO_INTERNET"
RANDOM_STATE = 777
TEST_SIZE = 0.20
CV_FOLDS = 5
SCORING = "neg_mean_squared_error"
MODEL_FEATURES = [
    "pct_urbano", "pct_con_energia", "pct_computadora", "pct_celular",
    "promedio_habitaciones", "promedio_personas",
]
FORBIDDEN_FEATURES = {
    "v19e_inetfijo", "v19f_inetmovil", "v19e_f", "TIENE_ACCESO_INTERNET",
    "fijo_si", "fijo_no", "movil_si", "movil_no", "algun_si", "algun_no",
    "pct_fijo", "pct_movil", "pct_algun", "pct_algun_determinados",
    "pct_sin_internet", "pct_sin_especificar", "brecha_movil_fijo_pp",
    "municipio_codigo", "municipio", TARGET,
}
FEATURE_DESCRIPTIONS = {
    "pct_urbano": "Porcentaje de viviendas urbanas del universo TIC",
    "pct_con_energia": "Porcentaje con electricidad entre respuestas determinadas",
    "pct_computadora": "Porcentaje con computadora/laptop/tablet entre respuestas determinadas",
    "pct_celular": "Porcentaje con teléfono celular entre respuestas determinadas",
    "promedio_habitaciones": "Media del código de habitaciones válido (8 = ocho o más)",
    "promedio_personas": "Media de personas por vivienda con valor válido",
}
REFERENCE_URL = (
    "https://github.com/ealaurel/MINERIA_DATOS_2026_2/blob/main/"
    "h3_3_Arboles_de_decisi%C3%B3n_regresion.ipynb"
)


@dataclass(frozen=True)
class MunicipalDataset:
    dataset: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    audit: dict[str, Any]


def seleccionar_predictores(dataset: pd.DataFrame) -> pd.DataFrame:
    """Lista permitida cerrada; excluye identificadores y todo agregado de Internet."""
    if FORBIDDEN_FEATURES.intersection(MODEL_FEATURES):
        raise ValueError("Predictores con fuga de información")
    missing = set(MODEL_FEATURES).difference(dataset.columns)
    if missing:
        raise ValueError(f"Faltan predictores: {sorted(missing)}")
    X = dataset.loc[:, MODEL_FEATURES].astype(float).copy()
    if np.isinf(X.to_numpy()).any():
        raise ValueError("Predictores infinitos")
    return X


def validar_target_eda(dataset: pd.DataFrame, eda: pd.DataFrame) -> dict[str, Any]:
    """Correspondencia uno a uno y tolerancia absoluta de 1e-8 pp."""
    for table in (dataset, eda):
        if table["municipio_codigo"].isna().any() or table["municipio_codigo"].duplicated().any():
            raise ValueError("Municipios ausentes o duplicados")
    left = dataset.assign(municipio_codigo=dataset["municipio_codigo"].astype(str))
    right = eda.assign(municipio_codigo=eda["municipio_codigo"].astype(str))
    merged = left.merge(right, on="municipio_codigo", how="outer", validate="one_to_one",
                        suffixes=("", "_eda"), indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("Los municipios no coinciden con el EDA")
    target = pd.to_numeric(merged[TARGET], errors="coerce")
    reference = pd.to_numeric(merged["pct_algun"], errors="coerce")
    if not target.between(0, 100).all() or not reference.between(0, 100).all():
        raise ValueError("Target ausente o fuera del rango 0–100")
    if not np.allclose(target, reference, atol=1e-8, rtol=0):
        raise ValueError("Target distinto de pct_algun del EDA; revisar metodología")
    if not merged["municipio"].eq(merged["municipio_eda"]).all():
        raise ValueError("Nombres municipales distintos del EDA")
    if "universo_eda" in merged and not merged["universo"].eq(merged["universo_eda"]).all():
        raise ValueError("Denominadores municipales distintos del EDA")
    return {"coincide": True, "tolerancia_pp": 1e-8,
            "max_diferencia_pp": float((target - reference).abs().max())}


def _code(dictionary: dict[str, Any], variable: str, label: str) -> str:
    codes = [code for code, text in dictionary[variable]["categorias"].items()
             if text.casefold() == label.casefold()]
    if len(codes) != 1:
        raise ValueError(f"Categoría oficial ambigua: {variable}, {label}")
    return codes[0]


def construir_dataset_municipal(
    data: pd.DataFrame, dictionary: dict[str, Any], eda: pd.DataFrame,
) -> MunicipalDataset:
    """Agrega viviendas reales sin excluir Internet indeterminado del universo."""
    applicable = data["v01_tipoviv"].isin(["1", "2", "3", "4", "5", "6"]) & data[
        "v02_condocup"].isin(["0", "1"])
    universe = data.loc[applicable].copy()
    universe["municipio_codigo"] = (
        universe["idep"].str.lstrip("0") + universe["iprov"] + universe["imun"]
    )
    universe["municipio"] = universe["municipio_codigo"].map(
        dictionary["mun_res_cod"]["categorias"])
    if universe["municipio"].isna().any() or universe.empty:
        raise ValueError("Universo vacío o códigos municipales sin correspondencia oficial")
    cleaned = limpiar_microdatos(universe)
    urban_code = _code(dictionary, "urbrur", "Urbana")
    no_energy = _code(dictionary, "v09_energia", "No tiene")
    energy_catalog = dictionary["v09_energia"]["categorias"]
    energy_yes = [code for code, label in energy_catalog.items()
                  if code != no_energy and "sin especificar" not in label.casefold()
                  and "no aplica" not in label.casefold()]
    rows = pd.DataFrame(index=universe.index)
    rows["municipio_codigo"] = universe["municipio_codigo"]
    rows["pct_urbano"] = universe["urbrur"].map({
        code: float(code == urban_code) * 100
        for code in dictionary["urbrur"]["categorias"]})
    rows["pct_con_energia"] = universe["v09_energia"].map(
        {**dict.fromkeys(energy_yes, 100.0), no_energy: 0.0})
    for source, feature in (("v19c_compu", "pct_computadora"), ("v19d_celular", "pct_celular")):
        rows[feature] = cleaned[source].map({
            _code(dictionary, source, "Sí"): 100.0,
            _code(dictionary, source, "No"): 0.0,
        })
    rows["promedio_habitaciones"] = cleaned["v13_habitac"]
    rows["promedio_personas"] = cleaned["tot_pers"]
    grouped = rows.groupby("municipio_codigo", sort=True)
    features = grouped[MODEL_FEATURES].mean()
    targets = summarize_groups(universe, ["municipio_codigo", "municipio"]).reset_index()
    targets = targets.rename(columns={"pct_algun": TARGET})
    targets = targets[["municipio_codigo", "municipio", "universo", TARGET]]
    validation = validar_target_eda(targets, eda)
    dataset = targets.merge(features, on="municipio_codigo", validate="one_to_one")
    dataset = dataset[["municipio_codigo", "municipio", *MODEL_FEATURES, TARGET]].sort_values(
        "municipio_codigo").reset_index(drop=True)
    X = seleccionar_predictores(dataset)
    y = dataset[TARGET].astype(float)
    counts = grouped[MODEL_FEATURES].count().add_prefix("validos_").reset_index()
    counts = counts.merge(targets[["municipio_codigo", "municipio", "universo"]],
                          on="municipio_codigo", validate="one_to_one")
    audit = {
        "registros_lapaz": int(len(data)), "universo_tic": int(len(universe)),
        "municipios": int(len(dataset)), "municipios_duplicados": 0, "target_nan": 0,
        "validacion_eda": validation,
        "energia_codigos_con_disponibilidad": energy_yes,
        "energia_codigo_ausencia": no_energy,
        "denominadores": (
            "Target: todas las viviendas del universo TIC, incluido Internet sin especificar. "
            "Porcentajes predictores: respuestas determinadas de cada variable; "
            "9/vacíos no se recodifican como No. Medias: valores válidos según H3_1/H3_2. "
            "No se imputa antes de agregar ni antes de separar train/test."
        ),
        "conteos_validos_municipales": counts.to_dict(orient="records"),
        "faltantes_municipales": {c: int(X[c].isna().sum()) for c in MODEL_FEATURES},
        "estadisticos_target": {"minimo": float(y.min()), "maximo": float(y.max()),
                                "promedio": float(y.mean()), "mediana": float(y.median())},
    }
    return MunicipalDataset(dataset, X, y, audit)


def preparar_dataset_modelo(project_root: Path, prefer_cache: bool = True) -> MunicipalDataset:
    source_dir = project_root / "Base de datos CSV"
    dictionary = read_dictionary(source_dir / "Diccionario de variables CPV 2024.xlsx")
    cached = project_root / "outputs/datos/vivienda_lapaz_seleccion.csv.gz"
    use_cache = prefer_cache and cached.is_file()
    data, national, _ = load_lapaz(source_dir / "Vivienda_CPV-2024.csv", cached,
                                   _code(dictionary, "dep_res_cod", "La Paz").zfill(2), use_cache)
    eda = pd.read_csv(project_root / "outputs/tablas/municipios.csv",
                      dtype={"municipio_codigo": str})
    prepared = construir_dataset_municipal(data, dictionary, eda)
    prepared.audit.update({"registros_nacionales": int(national),
                           "fuente_datos": str(cached.relative_to(project_root)) if use_cache
                           else "Base de datos CSV/Vivienda_CPV-2024.csv"})
    return prepared


def dividir_train_test(X: pd.DataFrame, y: pd.Series) -> tuple:
    """80/20 reproducible, sin estratificación de un target continuo."""
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def crear_pipeline() -> Pipeline:
    return Pipeline([("imputador", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("modelo", DecisionTreeRegressor(random_state=RANDOM_STATE))])


def crear_grid() -> dict[str, list[int]]:
    return {"modelo__max_depth": list(range(1, 20)),
            "modelo__min_samples_split": list(range(2, 10)),
            "modelo__min_samples_leaf": list(range(1, 5))}


def buscar_hiperparametros(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    if len(X_train) < CV_FOLDS:
        raise ValueError("Se requieren al menos cinco municipios train para CV=5")
    search = GridSearchCV(crear_pipeline(), crear_grid(), cv=CV_FOLDS, scoring=SCORING,
                          n_jobs=-1, return_train_score=True, error_score="raise")
    search.fit(X_train, y_train)
    return search


def calcular_metricas(y: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y, predictions))
    return {"mse": mse, "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y, predictions)),
            "r2": float(r2_score(y, predictions))}


def evaluar_modelo(model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    predictions = model.predict(X)
    return {"metricas": calcular_metricas(y, predictions), "predicciones": predictions}


def obtener_importancias(pipeline: Pipeline) -> pd.DataFrame:
    model = pipeline.named_steps["modelo"]
    values = model.feature_importances_
    # Un árbol sin divisiones tiene importancias cero; no se inventan contribuciones.
    expected = 1.0 if model.tree_.node_count > 1 else 0.0
    if not np.isclose(values.sum(), expected, atol=1e-8):
        raise AssertionError("Importancias inconsistentes con la estructura del árbol")
    return pd.DataFrame({"variable": MODEL_FEATURES, "importancia": values}).sort_values(
        "importancia", ascending=False, kind="stable")


def crear_predicciones(dataset: pd.DataFrame, y_test: pd.Series, predictions: np.ndarray) -> pd.DataFrame:
    table = dataset.loc[y_test.index, ["municipio_codigo", "municipio"]].copy()
    table["valor_real"] = y_test.to_numpy()
    table["valor_predicho"] = predictions
    table["error"] = table["valor_real"] - table["valor_predicho"]
    table["error_absoluto"] = table["error"].abs()
    return table.sort_values(["error_absoluto", "municipio_codigo"], ascending=[False, True])


def guardar_graficos(pipeline: Pipeline, predictions: pd.DataFrame, graphics_dir: Path) -> None:
    graphics_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(pipeline.named_steps["modelo"], feature_names=MODEL_FEATURES,
              max_depth=3, filled=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title("Árbol de regresión — niveles 0 a 3 (target en porcentaje)")
    fig.tight_layout()
    fig.savefig(graphics_dir / "arbol_regresion_niveles_0_3.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(predictions["valor_real"], predictions["valor_predicho"], color="#167f87")
    ax.plot([0, 100], [0, 100], "--", color="#b26128", label="Predicción ideal: y = x")
    ax.set(xlabel="Acceso real (%)", ylabel="Acceso predicho (%)", xlim=(0, 100), ylim=(0, 100),
           title="Real vs. predicho — municipios test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(graphics_dir / "regresion_real_vs_predicho.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(predictions["valor_predicho"], predictions["error"], color="#167f87")
    ax.axhline(0, linestyle="--", color="#b26128")
    ax.set(xlabel="Acceso predicho (%)", ylabel="Residuo: real − predicho (pp)",
           title="Residuos — municipios test")
    fig.tight_layout()
    fig.savefig(graphics_dir / "residuos_arbol_regresion.png", dpi=160)
    plt.close(fig)


def resultado_serializable(results: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in results.items() if key != "_runtime"}


def ejecutar_entrenamiento(project_root: Path, prefer_cache: bool = True,
                           save_outputs: bool = True, informar: bool = False) -> dict[str, Any]:
    prepared = preparar_dataset_modelo(project_root, prefer_cache)
    X_train, X_test, y_train, y_test = dividir_train_test(prepared.X, prepared.y)
    if len(y_test) < 2:
        raise ValueError("Se requieren al menos dos municipios test para evaluar R²")
    if informar:
        print(json.dumps({"antes_del_entrenamiento": {
            "municipios": len(prepared.dataset), "train": len(X_train), "test": len(X_test),
            "duplicados": prepared.audit["municipios_duplicados"],
            "target_nan": prepared.audit["target_nan"],
            "estadisticos_target": prepared.audit["estadisticos_target"],
        }}, ensure_ascii=False), file=sys.stderr)
    base = crear_pipeline().fit(X_train, y_train)
    base_metrics = evaluar_modelo(base, X_train, y_train)["metricas"]
    search = buscar_hiperparametros(X_train, y_train)
    pipeline = search.best_estimator_
    train = evaluar_modelo(pipeline, X_train, y_train)
    test = evaluar_modelo(pipeline, X_test, y_test)
    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
    baseline = evaluar_modelo(dummy, X_test, y_test)["metricas"]
    importance = obtener_importancias(pipeline)
    predictions = crear_predicciones(prepared.dataset, y_test, test["predicciones"])
    model = pipeline.named_steps["modelo"]
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results["params"] = cv_results["params"].map(lambda p: json.dumps(p, sort_keys=True))
    best = search.best_index_
    cv_folds = [-float(search.cv_results_[f"split{i}_test_score"][best]) for i in range(CV_FOLDS)]
    split = prepared.dataset[["municipio_codigo", "municipio"]].copy()
    split["particion"] = np.where(split.index.isin(X_train.index), "train", "test")
    comparison = pd.DataFrame([{"modelo": "DummyRegressor", **baseline},
                              {"modelo": "DecisionTreeRegressor", **test["metricas"]}])
    delta = {key: test["metricas"][key] - baseline[key] for key in baseline}
    params = {k.removeprefix("modelo__"): int(v) for k, v in search.best_params_.items()}
    top_errors = predictions.head(5).to_dict(orient="records")
    improved = test["metricas"]["mse"] < baseline["mse"]
    interpretation = (
        f"Se predijo el porcentaje municipal de viviendas con Internet mediante {', '.join(MODEL_FEATURES)}. "
        f"CV seleccionó {params}. En test, la diferencia absoluta promedio fue "
        f"{test['metricas']['mae']:.3f} puntos porcentuales y RMSE={test['metricas']['rmse']:.3f} pp. "
        f"R²={test['metricas']['r2']:.4f}; "
        + ("el desempeño es inferior a predecir la media del propio test. " if test['metricas']['r2'] < 0
           else "la variabilidad explicada debe interpretarse con el pequeño test municipal. ")
        + f"El árbol {'superó' if improved else 'no superó'} al baseline según MSE "
        f"(árbol={test['metricas']['mse']:.3f}; baseline={baseline['mse']:.3f} pp²). "
        f"La mayor importancia corresponde a {importance.iloc[0]['variable']} "
        f"({importance.iloc[0]['importancia']:.4f}). Los mayores errores se observan en "
        f"{', '.join(p['municipio'] for p in top_errors)}. Las importancias describen "
        "contribuciones a la predicción; no permiten atribuir efectos causales."
    )
    artifacts = {
        "dataset": "outputs/modelado/dataset_regresion_municipal.csv",
        "metricas": "outputs/modelado/metricas_arbol_regresion.json",
        "predicciones": "outputs/modelado/predicciones_arbol_regresion_test.csv",
        "importancia": "outputs/modelado/importancia_variables_arbol_regresion.csv",
        "gridsearch": "outputs/modelado/gridsearch_arbol_regresion.csv",
        "comparacion": "outputs/modelado/comparacion_baseline_arbol_regresion.csv",
        "particion": "outputs/modelado/particion_arbol_regresion.csv",
        "modelo": "outputs/modelos/arbol_regresion.joblib",
        "grafico_arbol": "outputs/graficos/arbol_regresion_niveles_0_3.png",
        "grafico_predicciones": "outputs/graficos/regresion_real_vs_predicho.png",
        "grafico_residuos": "outputs/graficos/residuos_arbol_regresion.png",
    }
    results = {
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "objetivo": "Predecir el porcentaje de viviendas con acceso a Internet de cada municipio de La Paz a partir de características censales agregadas.",
        "unidad_analisis": "Municipio", "referencia_academica": REFERENCE_URL,
        "target": {"nombre": TARGET, "indicador_eda": "pct_algun", "unidad": "porcentaje (0–100)",
                   "definicion": "v19e_f=1 / universo TIC municipal × 100; incluye sin especificar en el denominador",
                   "nota_leakage": "Solo se utilizan seis predictores permitidos; Internet, sus agregados, target e identificadores quedan fuera de X."},
        "variables_utilizadas": [{"variable": c, "descripcion": FEATURE_DESCRIPTIONS[c]} for c in MODEL_FEATURES],
        "registros": prepared.audit,
        "particion": {"train": len(X_train), "test": len(X_test), "test_size": TEST_SIZE,
                      "random_state": RANDOM_STATE, "estratificada": False,
                      "nota": "Se mantiene la convención 777 y 80/20 del proyecto frente a 22 y 70/30 del docente; sin stratify.",
                      "municipios_train": split.loc[X_train.index].to_dict(orient="records"),
                      "municipios_test": split.loc[X_test.index].to_dict(orient="records")},
        "modelo_base": {"parametros": {"random_state": RANDOM_STATE}, "metricas_train": base_metrics,
                        "nota": "Árbol base ilustrativo; test se evalúa solo con el modelo seleccionado mediante CV."},
        "gridsearch": {"cv": CV_FOLDS, "scoring": SCORING, "n_jobs": -1,
                       "best_params_": {k: int(v) for k, v in search.best_params_.items()},
                       "mejores_parametros": params, "best_score_": float(search.best_score_),
                       "mejor_mse_cv": -float(search.best_score_),
                       "std_mse_cv": float(search.cv_results_["std_test_score"][best]),
                       "mse_folds": cv_folds, "combinaciones_evaluadas": len(cv_results),
                       "ajustes_cv": len(cv_results) * CV_FOLDS, "reajustes_finales": 1,
                       "valores_evaluados": crear_grid(),
                       "estrategia": "Producto cartesiano completo; KFold=5 sin shuffle; imputación dentro de cada fold; solo train."},
        "metricas": {"train": train["metricas"], "test": test["metricas"]},
        "unidades_metricas": {"mae": "pp", "rmse": "pp", "mse": "pp²", "r2": "adimensional"},
        "baseline": {"modelo": "DummyRegressor", "strategy": "mean", "media_train": float(y_train.mean()),
                     "metricas_test": baseline},
        "comparacion_baseline": comparison.to_dict(orient="records"),
        "diferencia_arbol_menos_baseline": delta,
        "importancia_variables": importance.to_dict(orient="records"),
        "suma_importancias": float(importance["importancia"].sum()),
        "estructura_arbol": {"profundidad": int(model.tree_.max_depth), "nodos": int(model.tree_.node_count),
                             "hojas": int(model.get_n_leaves()), "visualizacion_max_depth": 3},
        "real_vs_predicho": predictions.to_dict(orient="records"),
        "residuos": {"definicion": "real - predicho", "promedio": float(predictions["error"].mean()),
                     "minimo": float(predictions["error"].min()), "maximo": float(predictions["error"].max()),
                     "mae": float(predictions["error_absoluto"].mean())},
        "municipios_mayor_error": top_errors, "interpretacion": interpretation,
        "limitaciones": [
            f"Solo {len(prepared.dataset)} municipios, frente a modelos individuales con muchas más viviendas; test contiene {len(X_test)} municipios.",
            "Una única partición y cinco folds producen una evaluación sensible a los territorios disponibles; la desviación CV no es un intervalo de confianza.",
            "Cada municipio pesa una vez, sin ponderar por número de viviendas; posibles dependencias espaciales no se modelan.",
            "Promedio de habitaciones usa códigos: 8 representa ocho o más, por lo que es una aproximación inferior en esa categoría.",
            "Los porcentajes de equipamiento se calculan entre respuestas determinadas; se conservan conteos válidos para auditar no respuesta.",
            "Acceso declarado en el censo 2024; no mide calidad del servicio ni identifica causalidad o efectos individuales (riesgo de falacia ecológica).",
            "No se garantiza generalización a otros departamentos, años, países ni condiciones futuras.",
        ], "artefactos": artifacts,
    }
    if save_outputs:
        for key, frame in (("dataset", prepared.dataset), ("predicciones", predictions),
                           ("importancia", importance), ("gridsearch", cv_results),
                           ("comparacion", comparison), ("particion", split)):
            path = project_root / artifacts[key]
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False, encoding="utf-8")
        model_path = project_root / artifacts["modelo"]
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path, compress=3)
        guardar_graficos(pipeline, predictions, project_root / "outputs/graficos")
        (project_root / artifacts["metricas"]).write_text(
            json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    results["_runtime"] = {"pipeline": pipeline, "base": base, "baseline": dummy,
                           "prepared": prepared, "X_train": X_train, "X_test": X_test,
                           "y_train": y_train, "y_test": y_test, "search": search}
    return results
