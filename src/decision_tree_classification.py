"""Árbol de decisión para clasificar acceso a Internet en La Paz.

H3_2 reutiliza de H3_1 el universo censal, el target, la selección y limpieza
de predictores y la partición train/test. La búsqueda de hiperparámetros se
realiza exclusivamente sobre train y el conjunto test se reserva para la
evaluación final.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

from src.logistic_regression import (
    CATEGORICAL_FEATURES,
    INTERNET_VARIABLES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TARGET_SOURCE,
    TEST_SIZE,
    dividir_train_test,
    preparar_dataset_modelo,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend definido antes de pyplot)


CV_FOLDS = 5
SCORING = "roc_auc"
MAX_DEPTH_VALUES = list(range(1, 16))
MIN_SAMPLES_SPLIT_VALUES = list(range(2, 10))
MIN_SAMPLES_LEAF_VALUES = list(range(1, 6))


def crear_pipeline(memory: str | None = None) -> Pipeline:
    """Crea el preprocesamiento solicitado y un árbol reproducible sin escalado."""

    leaked = INTERNET_VARIABLES.intersection(MODEL_FEATURES)
    if leaked or TARGET in MODEL_FEATURES:
        raise ValueError(f"Predictores con fuga de información: {sorted(leaked)}")

    categorical = Pipeline(
        steps=[
            ("imputador", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric = Pipeline(steps=[("imputador", SimpleImputer(strategy="median"))])
    preprocessor = ColumnTransformer(
        transformers=[
            ("categoricas", categorical, CATEGORICAL_FEATURES),
            ("numericas", numeric, NUMERIC_FEATURES),
        ],
        sparse_threshold=1.0,
    )
    model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    return Pipeline(
        steps=[("preprocesador", preprocessor), ("modelo", model)],
        memory=memory,
    )


def crear_grid_profundidad() -> dict[str, list[int]]:
    """Primera etapa: explora cada profundidad exigida por la actividad."""

    return {
        "modelo__max_depth": MAX_DEPTH_VALUES,
        "modelo__min_samples_split": [2],
        "modelo__min_samples_leaf": [1],
    }


def crear_grid_regularizacion(best_depth: int) -> dict[str, list[int]]:
    """Segunda etapa: cruza todas las restricciones de nodos en la mejor profundidad."""

    return {
        "modelo__max_depth": [int(best_depth)],
        "modelo__min_samples_split": MIN_SAMPLES_SPLIT_VALUES,
        "modelo__min_samples_leaf": MIN_SAMPLES_LEAF_VALUES,
    }


def _cv() -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def _gridsearch(pipeline: Pipeline, param_grid: dict[str, list[int]]) -> GridSearchCV:
    splits = param_grid["modelo__min_samples_split"]
    if min(splits) < 2:
        raise ValueError("min_samples_split debe ser mayor o igual a 2")
    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=SCORING,
        cv=_cv(),
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        error_score="raise",
    )


def _cv_results(search: GridSearchCV, stage: str) -> pd.DataFrame:
    table = pd.DataFrame(search.cv_results_).copy()
    keep = [
        "params",
        "param_modelo__max_depth",
        "param_modelo__min_samples_split",
        "param_modelo__min_samples_leaf",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
        "mean_train_score",
        "std_train_score",
        "mean_fit_time",
        "std_fit_time",
    ]
    table = table[keep]
    table.insert(0, "etapa", stage)
    table["params"] = table["params"].map(
        lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
    )
    return table.sort_values(["rank_test_score", "mean_fit_time"])


def buscar_hiperparametros(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[dict[str, int], float, pd.DataFrame]:
    """Ejecuta una búsqueda en dos etapas sin muestrear el conjunto de entrenamiento.

    El diseño reduce 600 combinaciones cartesianas a 55 candidatos: primero
    evalúa las profundidades 1--15 y, en la mejor, cruza todos los valores
    solicitados de ``min_samples_split`` y ``min_samples_leaf``. Con CV=5 son
    275 ajustes. El cache temporal evita repetir el preprocesamiento idéntico.
    """

    with tempfile.TemporaryDirectory(prefix="h32-gridsearch-") as cache_dir:
        depth_search = _gridsearch(
            crear_pipeline(memory=cache_dir), crear_grid_profundidad()
        )
        depth_search.fit(X_train, y_train)
        best_depth = int(depth_search.best_params_["modelo__max_depth"])

        regularization_search = _gridsearch(
            crear_pipeline(memory=cache_dir),
            crear_grid_regularizacion(best_depth),
        )
        regularization_search.fit(X_train, y_train)

        search_results = pd.concat(
            [
                _cv_results(depth_search, "1_profundidad"),
                _cv_results(regularization_search, "2_regularizacion"),
            ],
            ignore_index=True,
        )
        best_params = {
            key.removeprefix("modelo__"): int(value)
            for key, value in regularization_search.best_params_.items()
        }
        best_score = float(regularization_search.best_score_)

    return best_params, best_score, search_results


def entrenar_modelo_final(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict[str, int],
) -> Pipeline:
    pipeline = crear_pipeline()
    pipeline.set_params(
        **{f"modelo__{key}": value for key, value in best_params.items()}
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def _metricas(y_true: pd.Series, predictions: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def _matriz(y_true: pd.Series, predictions: np.ndarray) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "matriz": [[tn, fp], [fn, tp]],
    }


def evaluar_modelo(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evalúa con probabilidades; ROC-AUC nunca se calcula con clases predichas."""

    probabilities = pipeline.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype("int8")
    return {
        "metricas": _metricas(y, predictions, probabilities),
        "matriz_confusion": _matriz(y, predictions),
        "classification_report": classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["Sin Internet", "Con Internet"],
            output_dict=True,
            zero_division=0,
        ),
        "predicciones": predictions,
        "probabilidades": probabilities,
    }


def seleccionar_umbral_roc(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict[str, int],
) -> dict[str, Any]:
    """Selecciona el umbral por Youden usando probabilidades out-of-fold de train."""

    estimator = crear_pipeline()
    estimator.set_params(
        **{f"modelo__{key}": value for key, value in best_params.items()}
    )
    probabilities = cross_val_predict(
        estimator,
        X_train,
        y_train,
        cv=_cv(),
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, probabilities)
    finite = np.isfinite(thresholds)
    valid_indices = np.flatnonzero(finite)
    if not len(valid_indices):
        raise ValueError("No se encontró un umbral ROC finito")
    youden = tpr[finite] - fpr[finite]
    selected_index = int(valid_indices[int(np.argmax(youden))])
    threshold = float(thresholds[selected_index])
    predictions = (probabilities >= threshold).astype("int8")
    return {
        "umbral": threshold,
        "criterio": "Máximo índice de Youden (TPR - FPR) sobre probabilidades out-of-fold de train",
        "origen": "train, validación cruzada estratificada de 5 folds",
        "indice_youden": float(tpr[selected_index] - fpr[selected_index]),
        "tpr_seleccion": float(tpr[selected_index]),
        "fpr_seleccion": float(fpr[selected_index]),
        "metricas_oof_train": _metricas(y_train, predictions, probabilities),
    }


def _roc_points(y_true: pd.Series, probabilities: np.ndarray) -> list[dict[str, float | None]]:
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    return [
        {
            "fpr": float(x),
            "tpr": float(y),
            "threshold": None if np.isinf(t) else float(t),
        }
        for x, y, t in zip(fpr, tpr, thresholds, strict=True)
    ]


def obtener_importancias(
    pipeline: Pipeline,
    dictionary: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve importancias transformadas y agregadas por predictor original."""

    preprocessor: ColumnTransformer = pipeline.named_steps["preprocesador"]
    raw_names = [str(value) for value in preprocessor.get_feature_names_out()]
    importances = pipeline.named_steps["modelo"].feature_importances_
    rows: list[dict[str, Any]] = []

    for raw_name, importance in zip(raw_names, importances, strict=True):
        if raw_name.startswith("categoricas__"):
            feature = raw_name.removeprefix("categoricas__")
            variable = next(
                item for item in CATEGORICAL_FEATURES if feature.startswith(f"{item}_")
            )
            category_code = feature.removeprefix(f"{variable}_")
            category = (
                dictionary[variable]["categorias"].get(category_code, category_code)
                if dictionary is not None
                else category_code
            )
        elif raw_name.startswith("numericas__"):
            feature = raw_name.removeprefix("numericas__")
            variable = feature
            category_code = None
            category = None
        else:
            raise ValueError(f"Nombre transformado no reconocido: {raw_name}")
        rows.append(
            {
                "feature": feature,
                "variable_original": variable,
                "categoria_codigo": category_code,
                "categoria": category,
                "importancia": float(importance),
            }
        )

    transformed = pd.DataFrame(rows).sort_values("importancia", ascending=False)
    aggregated = (
        transformed.groupby("variable_original", as_index=False)["importancia"]
        .sum()
        .sort_values("importancia", ascending=False)
    )
    if not np.isclose(aggregated["importancia"].sum(), 1.0, atol=1e-8):
        raise AssertionError("Las importancias agregadas no suman aproximadamente 1")
    return transformed, aggregated


def guardar_visualizacion_arbol(pipeline: Pipeline, path: Path) -> None:
    """Guarda solo los niveles 0--3; el estimador serializado conserva el árbol completo."""

    path.parent.mkdir(parents=True, exist_ok=True)
    feature_names = [
        str(name).replace("categoricas__", "").replace("numericas__", "")
        for name in pipeline.named_steps["preprocesador"].get_feature_names_out()
    ]
    figure, axis = plt.subplots(figsize=(24, 12))
    plot_tree(
        pipeline.named_steps["modelo"],
        feature_names=feature_names,
        class_names=["Sin Internet", "Con Internet"],
        filled=True,
        rounded=True,
        proportion=True,
        max_depth=3,
        fontsize=7,
        ax=axis,
    )
    axis.set_title("Árbol de decisión — niveles 0 a 3 (modelo completo no truncado)")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plain_params(params: dict[str, Any]) -> dict[str, int]:
    return {key.removeprefix("modelo__"): int(value) for key, value in params.items()}


def _comparison(
    project_root: Path,
    tree_metrics: dict[str, float],
    index_test: pd.Index,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    logistic_path = project_root / "outputs" / "modelado" / "metricas_regresion_logistica.json"
    logistic_predictions_path = project_root / "outputs" / "modelado" / "predicciones_test.csv.gz"
    if not logistic_path.is_file() or not logistic_predictions_path.is_file():
        raise FileNotFoundError(
            "Faltan artefactos de H3_1. Ejecute python scripts/entrenar_regresion_logistica.py"
        )

    logistic = json.loads(logistic_path.read_text(encoding="utf-8"))
    logistic_indices = pd.read_csv(
        logistic_predictions_path,
        usecols=["indice_registro_lapaz"],
    )["indice_registro_lapaz"].to_numpy()
    same_indices = np.array_equal(logistic_indices, index_test.to_numpy())
    verification = {
        "mismos_registros": int(logistic["registros"]["registros_target_valido"])
        == int(len(index_test) + logistic["particion"]["train"]),
        "mismos_indices_test_y_orden": bool(same_indices),
        "mismo_random_state": int(logistic["particion"]["random_state"])
        == RANDOM_STATE,
        "mismo_test_size": float(logistic["particion"]["test_size"]) == TEST_SIZE,
        "mismo_target": logistic["target"]["nombre"] == TARGET,
        "misma_variable_fuente": logistic["target"]["variable_fuente"] == TARGET_SOURCE,
    }
    if not all(verification.values()):
        raise AssertionError(f"La comparación H3_1/H3_2 no es homogénea: {verification}")

    rows = []
    for model, metrics in (
        ("Regresión Logística", logistic["metricas"]["test"]),
        ("Árbol de Decisión", tree_metrics),
    ):
        rows.append(
            {
                "modelo": model,
                **{key: float(metrics[key]) for key in ("accuracy", "precision", "recall", "f1", "roc_auc")},
            }
        )
    return pd.DataFrame(rows), verification


def _interpretation(train: dict[str, float], test: dict[str, float]) -> str:
    auc_gap = train["roc_auc"] - test["roc_auc"]
    accuracy_gap = train["accuracy"] - test["accuracy"]
    if auc_gap > 0.03 or accuracy_gap > 0.03:
        return (
            "Las métricas de entrenamiento superan de forma apreciable a las de prueba; "
            "la diferencia es compatible con sobreajuste y debe considerarse al interpretar el árbol."
        )
    if test["roc_auc"] < 0.65:
        return (
            "Train y test son cercanos, pero la discriminación es limitada con los predictores disponibles."
        )
    return (
        "Las métricas de train y test son cercanas; no aparece una brecha importante de generalización. "
        "El resultado describe asociaciones predictivas y no efectos causales."
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    raise TypeError(f"Tipo no serializable: {type(value)!r}")


def resultado_serializable(results: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in results.items() if key != "_runtime"}


def ejecutar_entrenamiento(
    project_root: Path,
    prefer_cache: bool = True,
    save_outputs: bool = True,
) -> dict[str, Any]:
    """Ejecuta H3_2 de extremo a extremo y guarda artefactos reproducibles."""

    prepared = preparar_dataset_modelo(project_root, prefer_cache=prefer_cache)
    X_train, X_test, y_train, y_test, index_train, index_test = dividir_train_test(
        prepared.X, prepared.y, prepared.indices
    )

    best_params, best_score, search_results = buscar_hiperparametros(X_train, y_train)
    pipeline = entrenar_modelo_final(X_train, y_train, best_params)
    threshold_selection = seleccionar_umbral_roc(X_train, y_train, best_params)
    roc_threshold = float(threshold_selection["umbral"])

    train_standard = evaluar_modelo(pipeline, X_train, y_train, threshold=0.5)
    test_standard = evaluar_modelo(pipeline, X_test, y_test, threshold=0.5)
    train_roc = evaluar_modelo(pipeline, X_train, y_train, threshold=roc_threshold)
    test_roc = evaluar_modelo(pipeline, X_test, y_test, threshold=roc_threshold)
    transformed_importance, aggregated_importance = obtener_importancias(
        pipeline, prepared.dictionary
    )

    model: DecisionTreeClassifier = pipeline.named_steps["modelo"]
    structure = {
        "profundidad": int(model.tree_.max_depth),
        "nodos": int(model.tree_.node_count),
        "hojas": int(model.get_n_leaves()),
        "visualizacion_max_depth": 3,
        "nota_visualizacion": (
            "La figura muestra solo los niveles 0 a 3; el modelo entrenado y serializado conserva el árbol completo."
        ),
    }
    comparison, split_verification = _comparison(
        project_root, test_standard["metricas"], index_test
    )

    train_distribution = y_train.value_counts().sort_index()
    test_distribution = y_test.value_counts().sort_index()
    artifacts = {
        "metricas": "outputs/modelado/metricas_arbol_clasificacion.json",
        "matriz_confusion": "outputs/modelado/matriz_confusion_arbol_test.csv",
        "importancia_transformada": "outputs/modelado/importancia_variables_arbol.csv",
        "importancia_agregada": "outputs/modelado/importancia_variables_arbol_agregada.csv",
        "predicciones": "outputs/modelado/predicciones_arbol_test.csv.gz",
        "gridsearch": "outputs/modelado/gridsearch_arbol_clasificacion.csv",
        "comparacion": "outputs/modelado/comparacion_modelos_clasificacion.csv",
        "modelo": "outputs/modelos/arbol_clasificacion.joblib",
        "grafico_arbol": "outputs/graficos/arbol_clasificacion_niveles_0_3.png",
    }
    results: dict[str, Any] = {
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "objetivo": (
            "Predecir si una vivienda u hogar del departamento de La Paz tiene acceso a Internet "
            "mediante un árbol de decisión de clasificación."
        ),
        "target": {
            "nombre": TARGET,
            "variable_fuente": TARGET_SOURCE,
            "descripcion_fuente": prepared.dictionary[TARGET_SOURCE]["descripcion"],
            "codificacion": {"0": "No tiene acceso", "1": "Tiene acceso"},
            "clase_positiva": 1,
            "nota_leakage": (
                "v19e_inetfijo, v19f_inetmovil, v19e_f, el target y sus transformaciones "
                "no se usan como predictores."
            ),
        },
        "variables_utilizadas": prepared.audit["variables"],
        "registros": {
            key: value
            for key, value in prepared.audit.items()
            if key not in {"variables", "distribucion_clases"}
        },
        "distribucion_clases_total": prepared.audit["distribucion_clases"],
        "particion": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "estratificada": True,
            "train": int(len(y_train)),
            "test": int(len(y_test)),
            "distribucion_train": {
                "0_sin_internet": int(train_distribution.get(0, 0)),
                "1_con_internet": int(train_distribution.get(1, 0)),
            },
            "distribucion_test": {
                "0_sin_internet": int(test_distribution.get(0, 0)),
                "1_con_internet": int(test_distribution.get(1, 0)),
            },
        },
        "gridsearch": {
            "cv": CV_FOLDS,
            "scoring": SCORING,
            "n_jobs": -1,
            "mejores_parametros": best_params,
            "mejor_roc_auc_cv": best_score,
            "best_params_": best_params,
            "best_score_": best_score,
            "valores_evaluados": {
                "max_depth": MAX_DEPTH_VALUES,
                "min_samples_split": MIN_SAMPLES_SPLIT_VALUES,
                "min_samples_leaf": MIN_SAMPLES_LEAF_VALUES,
            },
            "combinaciones_evaluadas": int(len(search_results)),
            "ajustes_cv": int(len(search_results) * CV_FOLDS),
            "estrategia": (
                "Búsqueda en dos etapas sobre todo X_train: profundidades 1-15; después cruce "
                "2-9 × 1-5 para min_samples_split y min_samples_leaf en la mejor profundidad."
            ),
            "motivo_reduccion": (
                "El producto cartesiano completo requeriría 3.000 ajustes sobre 831.472 registros de train; "
                "la estrategia conserva todos los valores exigidos, evita muestreo y requiere 275 ajustes."
            ),
        },
        "umbral_estandar": 0.5,
        "metricas": {
            "train": train_standard["metricas"],
            "test": test_standard["metricas"],
        },
        "matriz_confusion_test": test_standard["matriz_confusion"],
        "classification_report_test": test_standard["classification_report"],
        "roc": {
            "auc_test": test_standard["metricas"]["roc_auc"],
            "puntos": _roc_points(y_test, test_standard["probabilidades"]),
        },
        "seleccion_umbral_roc": {
            **threshold_selection,
            "metricas_train": train_roc["metricas"],
            "metricas_test": test_roc["metricas"],
            "matriz_confusion_test": test_roc["matriz_confusion"],
        },
        "importancia_variables": {
            "transformadas": transformed_importance.to_dict(orient="records"),
            "agregadas": aggregated_importance.to_dict(orient="records"),
            "suma_agregada": float(aggregated_importance["importancia"].sum()),
        },
        "estructura_arbol": structure,
        "comparacion_modelos": comparison.to_dict(orient="records"),
        "verificacion_comparabilidad": split_verification,
        "interpretacion": _interpretation(
            train_standard["metricas"], test_standard["metricas"]
        ),
        "artefactos": artifacts,
    }

    if save_outputs:
        modeling_dir = project_root / "outputs" / "modelado"
        models_dir = project_root / "outputs" / "modelos"
        graphics_dir = project_root / "outputs" / "graficos"
        modeling_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
        graphics_dir.mkdir(parents=True, exist_ok=True)

        transformed_importance.to_csv(
            modeling_dir / "importancia_variables_arbol.csv",
            index=False,
            encoding="utf-8",
        )
        aggregated_importance.to_csv(
            modeling_dir / "importancia_variables_arbol_agregada.csv",
            index=False,
            encoding="utf-8",
        )
        search_results.to_csv(
            modeling_dir / "gridsearch_arbol_clasificacion.csv",
            index=False,
            encoding="utf-8",
        )
        comparison.to_csv(
            modeling_dir / "comparacion_modelos_clasificacion.csv",
            index=False,
            encoding="utf-8",
        )
        pd.DataFrame(
            test_standard["matriz_confusion"]["matriz"],
            index=["real_0_sin_internet", "real_1_con_internet"],
            columns=["predicho_0_sin_internet", "predicho_1_con_internet"],
        ).to_csv(
            modeling_dir / "matriz_confusion_arbol_test.csv",
            encoding="utf-8",
        )
        pd.DataFrame(
            {
                "indice_registro_lapaz": index_test.to_numpy(),
                "valor_real": y_test.to_numpy(),
                "prediccion_umbral_0_5": test_standard["predicciones"],
                "probabilidad_acceso_internet": test_standard["probabilidades"],
                "prediccion_umbral_roc": test_roc["predicciones"],
                "umbral_roc": roc_threshold,
            }
        ).to_csv(
            modeling_dir / "predicciones_arbol_test.csv.gz",
            index=False,
            encoding="utf-8",
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )
        joblib.dump(pipeline, models_dir / "arbol_clasificacion.joblib", compress=3)
        guardar_visualizacion_arbol(
            pipeline, graphics_dir / "arbol_clasificacion_niveles_0_3.png"
        )
        (modeling_dir / "metricas_arbol_clasificacion.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2, default=_json_value),
            encoding="utf-8",
        )

    results["_runtime"] = {
        "pipeline": pipeline,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "index_train": index_train,
        "index_test": index_test,
        "gridsearch_resultados": search_results,
        "importancia_transformada": transformed_importance,
        "importancia_agregada": aggregated_importance,
    }
    return results
