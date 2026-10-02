"""Primera iteración de regresión logística para acceso a Internet en La Paz.

El módulo reutiliza la lectura y el universo TIC definidos por ``src.pipeline``.
La variable oficial ``v19e_f`` se usa exclusivamente para construir el target;
ninguna variable de Internet participa como predictor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.pipeline import load_lapaz, read_dictionary


TARGET = "TIENE_ACCESO_INTERNET"
TARGET_SOURCE = "v19e_f"
RANDOM_STATE = 777
TEST_SIZE = 0.20

INTERNET_VARIABLES = {
    "v19e_inetfijo",
    "v19f_inetmovil",
    "v19e_f",
}

CATEGORICAL_FEATURES = [
    "urbrur",
    "v01_tipoviv",
    "v09_energia",
    "v19c_compu",
    "v19d_celular",
]
NUMERIC_FEATURES = ["v13_habitac", "tot_pers"]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

FEATURE_REASONS = {
    "urbrur": (
        "La disponibilidad de infraestructura y cobertura digital puede diferir "
        "entre áreas urbanas y rurales."
    ),
    "v01_tipoviv": (
        "El tipo de vivienda aproxima condiciones residenciales asociadas con la "
        "posibilidad de contratar o instalar conectividad."
    ),
    "v09_energia": (
        "La fuente de energía eléctrica representa una condición básica para usar "
        "equipos y servicios digitales."
    ),
    "v19c_compu": (
        "La disponibilidad de computadora, laptop o tablet está asociada con la "
        "capacidad de uso de servicios digitales, sin revelar directamente el target."
    ),
    "v19d_celular": (
        "La tenencia de teléfono celular está asociada con la posibilidad de usar "
        "servicios digitales, pero no equivale a declarar Internet móvil."
    ),
    "v13_habitac": (
        "La cantidad de habitaciones describe el tamaño físico de la vivienda y "
        "puede aproximar diferencias socioeconómicas."
    ),
    "tot_pers": (
        "El tamaño del hogar puede relacionarse con la demanda compartida y la "
        "capacidad de sostener un servicio de Internet."
    ),
}


@dataclass(frozen=True)
class PreparedDataset:
    X: pd.DataFrame
    y: pd.Series
    indices: pd.Index
    dictionary: dict[str, dict[str, Any]]
    audit: dict[str, Any]


def crear_target(values: pd.Series) -> pd.Series:
    """Codifica la variable oficial combinada: 1=acceso, 0=sin acceso.

    Los códigos distintos de 1 y 2, incluido 9=Sin especificar, permanecen
    ausentes y por tanto no se convierten en la clase negativa.
    """

    return values.map({"1": 1, "2": 0}).astype("Int8")


def seleccionar_variables(data: pd.DataFrame) -> pd.DataFrame:
    """Selecciona y limpia predictores sin convertir categorías en magnitudes."""

    leaked = INTERNET_VARIABLES.intersection(MODEL_FEATURES)
    if leaked:
        raise ValueError(f"Predictores con fuga de información: {sorted(leaked)}")

    missing = set(MODEL_FEATURES).difference(data.columns)
    if missing:
        raise ValueError(f"Faltan variables explicativas: {sorted(missing)}")

    result = data[MODEL_FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        result[column] = result[column].astype("object").replace("", np.nan)

    # El diccionario define 9 como "Sin especificar" en estas variables.
    for column in ("v19c_compu", "v19d_celular"):
        result[column] = result[column].replace("9", np.nan)

    for column in NUMERIC_FEATURES:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result.loc[~result["v13_habitac"].between(1, 8), "v13_habitac"] = np.nan
    result.loc[~result["tot_pers"].between(0, 9999), "tot_pers"] = np.nan
    return result


def _variable_table(
    dictionary: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for variable in MODEL_FEATURES:
        item = dictionary[variable]
        rows.append(
            {
                "variable": variable,
                "descripcion": str(item["descripcion"]),
                "tipo": "Categórica" if variable in CATEGORICAL_FEATURES else "Numérica",
                "motivo": FEATURE_REASONS[variable],
            }
        )
    return rows


def preparar_dataset_modelo(
    project_root: Path,
    prefer_cache: bool = True,
) -> PreparedDataset:
    """Carga La Paz, conserva el universo TIC y elimina targets indeterminados."""

    source_dir = project_root / "Base de datos CSV"
    source = source_dir / "Vivienda_CPV-2024.csv"
    dictionary_path = source_dir / "Diccionario de variables CPV 2024.xlsx"
    cached = project_root / "outputs" / "datos" / "vivienda_lapaz_seleccion.csv.gz"

    dictionary = read_dictionary(dictionary_path)
    department_codes = [
        code
        for code, label in dictionary["dep_res_cod"]["categorias"].items()
        if label == "La Paz"
    ]
    if len(department_codes) != 1:
        raise ValueError("No se pudo identificar unívocamente el código de La Paz")

    use_cache = prefer_cache and cached.is_file()
    data, national_records, _ = load_lapaz(
        source=source,
        cached=cached,
        department_code=department_codes[0].zfill(2),
        from_cache=use_cache,
    )

    applicable = data["v01_tipoviv"].isin(["1", "2", "3", "4", "5", "6"]) & data[
        "v02_condocup"
    ].isin(["0", "1"])
    raw_target = data[TARGET_SOURCE]
    target = crear_target(raw_target)
    valid = applicable & target.notna()

    unspecified = applicable & raw_target.eq("9")
    invalid_or_empty = applicable & ~raw_target.isin(["1", "2", "9"])
    outside_universe = ~applicable

    X = seleccionar_variables(data.loc[valid])
    y = target.loc[valid].astype("int8").rename(TARGET)
    if not set(y.unique()).issubset({0, 1}) or y.nunique() != 2:
        raise ValueError("El target no contiene exactamente las clases binarias 0 y 1")

    class_counts = y.value_counts().sort_index()
    audit = {
        "fuente_datos": (
            "outputs/datos/vivienda_lapaz_seleccion.csv.gz (derivado reproducible)"
            if use_cache
            else "Base de datos CSV/Vivienda_CPV-2024.csv"
        ),
        "registros_nacionales": int(national_records),
        "registros_lapaz": int(len(data)),
        "universo_tic": int(applicable.sum()),
        "registros_target_valido": int(valid.sum()),
        "registros_excluidos": int((~valid).sum()),
        "motivos_exclusion": {
            "fuera_universo_tic": int(outside_universe.sum()),
            "target_sin_especificar_codigo_9": int(unspecified.sum()),
            "target_vacio_o_fuera_catalogo": int(invalid_or_empty.sum()),
        },
        "distribucion_clases": {
            "0_sin_internet": int(class_counts.get(0, 0)),
            "1_con_internet": int(class_counts.get(1, 0)),
            "porcentaje_clase_positiva": float(y.mean() * 100),
        },
        "variables": _variable_table(dictionary),
    }
    if audit["registros_excluidos"] != sum(audit["motivos_exclusion"].values()):
        raise AssertionError("Los motivos de exclusión no concilian con el total")

    return PreparedDataset(X=X, y=y, indices=X.index, dictionary=dictionary, audit=audit)


def crear_pipeline() -> Pipeline:
    """Construye un preprocesamiento reproducible y el modelo base sin pesos."""

    categorical = Pipeline(
        steps=[
            ("imputador", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=True,
                ),
            ),
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
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    return Pipeline(steps=[("preprocesador", preprocessor), ("modelo", model)])


def dividir_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    indices: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Index, pd.Index]:
    """Aplica la partición 80/20 estratificada solicitada."""

    return train_test_split(
        X,
        y,
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def entrenar_modelo(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluar_modelo(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, Any]:
    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)[:, 1]
    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return {
        "metricas": {
            "accuracy": float(accuracy_score(y, predictions)),
            "precision": float(precision_score(y, predictions, zero_division=0)),
            "recall": float(recall_score(y, predictions, zero_division=0)),
            "f1": float(f1_score(y, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probabilities)),
        },
        "matriz_confusion": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matriz": [[tn, fp], [fn, tp]],
        },
        "classification_report": classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["Sin Internet", "Con Internet"],
            output_dict=True,
            zero_division=0,
        ),
        "predicciones": predictions.astype("int8"),
        "probabilidades": probabilities,
    }


def _feature_metadata(
    pipeline: Pipeline,
    dictionary: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocesador"]
    feature_names = preprocessor.get_feature_names_out()
    metadata: dict[str, dict[str, Any]] = {}

    encoder: OneHotEncoder = preprocessor.named_transformers_["categoricas"].named_steps[
        "one_hot"
    ]
    for variable, categories, drop_index in zip(
        CATEGORICAL_FEATURES,
        encoder.categories_,
        encoder.drop_idx_,
        strict=True,
    ):
        reference_code = str(categories[int(drop_index)])
        reference = dictionary[variable]["categorias"].get(reference_code, reference_code)
        for category in categories:
            code = str(category)
            if code == reference_code:
                continue
            raw_name = f"categoricas__{variable}_{code}"
            category_label = dictionary[variable]["categorias"].get(code, code)
            metadata[raw_name] = {
                "variable_original": variable,
                "categoria_codigo": code,
                "categoria": category_label,
                "descripcion": (
                    f"{dictionary[variable]['descripcion']}: {category_label} "
                    f"(referencia: {reference})"
                ),
            }

    for variable in NUMERIC_FEATURES:
        raw_name = f"numericas__{variable}"
        metadata[raw_name] = {
            "variable_original": variable,
            "categoria_codigo": None,
            "categoria": None,
            "descripcion": f"{dictionary[variable]['descripcion']} (incremento de una unidad)",
        }

    missing = set(feature_names).difference(metadata)
    if missing:
        raise ValueError(f"No se pudo describir las variables transformadas: {sorted(missing)}")
    return metadata


def obtener_coeficientes(
    pipeline: Pipeline,
    dictionary: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Extrae coeficientes y odds ratios del modelo sklearn ajustado."""

    names = pipeline.named_steps["preprocesador"].get_feature_names_out()
    coefficients = pipeline.named_steps["modelo"].coef_[0]
    metadata = _feature_metadata(pipeline, dictionary)
    rows = []
    for name, coefficient in zip(names, coefficients, strict=True):
        rows.append(
            {
                "feature": str(name),
                **metadata[str(name)],
                "coeficiente": float(coefficient),
                "odds_ratio": float(np.exp(coefficient)),
                "impacto_absoluto": float(abs(coefficient)),
            }
        )
    return pd.DataFrame(rows).sort_values("impacto_absoluto", ascending=False)


def obtener_odds_ratios(
    pipeline: Pipeline,
    dictionary: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    return obtener_coeficientes(pipeline, dictionary)[
        ["feature", "descripcion", "coeficiente", "odds_ratio"]
    ]


def ajustar_statsmodels(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[Any, pd.DataFrame]:
    """Ajusta ``statsmodels.api.Logit`` sobre el diseño aprendido en train."""

    transformed = pipeline.named_steps["preprocesador"].transform(X_train)
    if sparse.issparse(transformed):
        transformed = transformed.toarray()
    feature_names = list(pipeline.named_steps["preprocesador"].get_feature_names_out())
    design = pd.DataFrame(transformed, columns=feature_names, index=X_train.index)
    design = sm.add_constant(design, has_constant="add")
    model = sm.Logit(y_train.astype(float), design)
    result = model.fit(method="lbfgs", maxiter=200, disp=False)

    confidence = result.conf_int()
    inference = pd.DataFrame(
        {
            "feature": result.params.index,
            "coeficiente_statsmodels": result.params.to_numpy(),
            "p_value": result.pvalues.to_numpy(),
            "ci_95_inferior": confidence[0].to_numpy(),
            "ci_95_superior": confidence[1].to_numpy(),
        }
    )
    inference["odds_ratio_statsmodels"] = np.exp(
        inference["coeficiente_statsmodels"]
    )
    inference["or_ci_95_inferior"] = np.exp(inference["ci_95_inferior"])
    inference["or_ci_95_superior"] = np.exp(inference["ci_95_superior"])
    return result, inference


def _downsample_roc(
    y_true: pd.Series,
    probabilities: np.ndarray,
    max_points: int = 201,
) -> list[dict[str, float]]:
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    if len(fpr) > max_points:
        selected = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
        fpr, tpr, thresholds = fpr[selected], tpr[selected], thresholds[selected]
    return [
        {
            "fpr": float(x),
            "tpr": float(y),
            "threshold": None if np.isinf(t) else float(t),
        }
        for x, y, t in zip(fpr, tpr, thresholds, strict=True)
    ]


def _consistency_interpretation(
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> str:
    auc_gap = train_metrics["roc_auc"] - test_metrics["roc_auc"]
    accuracy_gap = train_metrics["accuracy"] - test_metrics["accuracy"]
    if auc_gap > 0.03 or accuracy_gap > 0.03:
        return (
            "El rendimiento de entrenamiento supera de forma apreciable al de prueba; "
            "existe una señal de posible sobreajuste que debe revisarse."
        )
    if test_metrics["roc_auc"] < 0.65:
        return (
            "Train y test son similares, pero la discriminación es limitada; la primera "
            "iteración puede estar subajustada o requerir mejores predictores."
        )
    return (
        "Las métricas de train y test son cercanas, lo que indica un comportamiento "
        "consistente sin una señal importante de sobreajuste."
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


def ejecutar_entrenamiento(
    project_root: Path,
    prefer_cache: bool = True,
    save_outputs: bool = True,
) -> dict[str, Any]:
    """Ejecuta el flujo completo y opcionalmente guarda todos los artefactos."""

    prepared = preparar_dataset_modelo(project_root, prefer_cache=prefer_cache)
    X_train, X_test, y_train, y_test, index_train, index_test = dividir_train_test(
        prepared.X, prepared.y, prepared.indices
    )

    pipeline = entrenar_modelo(crear_pipeline(), X_train, y_train)
    train_evaluation = evaluar_modelo(pipeline, X_train, y_train)
    test_evaluation = evaluar_modelo(pipeline, X_test, y_test)

    coefficients = obtener_coeficientes(pipeline, prepared.dictionary)
    stats_result, inference = ajustar_statsmodels(pipeline, X_train, y_train)
    coefficients = coefficients.merge(inference, on="feature", how="left")

    majority_class = int(y_train.mode().iloc[0])
    baseline_predictions = np.full(len(y_test), majority_class, dtype="int8")
    baseline_accuracy = float(accuracy_score(y_test, baseline_predictions))

    train_distribution = y_train.value_counts().sort_index()
    test_distribution = y_test.value_counts().sort_index()
    train_metrics = train_evaluation["metricas"]
    test_metrics = test_evaluation["metricas"]
    interpretation = _consistency_interpretation(train_metrics, test_metrics)

    significant = inference.loc[
        inference["feature"].ne("const") & inference["p_value"].lt(0.05)
    ].copy()
    significant = significant.merge(
        coefficients[["feature", "descripcion"]], on="feature", how="left"
    ).sort_values("p_value")

    top_coefficients = coefficients.head(10).copy()
    results: dict[str, Any] = {
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "objetivo": (
            "Estimar la probabilidad de que una vivienda u hogar del departamento "
            "de La Paz tenga acceso a Internet a partir de características censales."
        ),
        "target": {
            "nombre": TARGET,
            "variable_fuente": TARGET_SOURCE,
            "descripcion_fuente": prepared.dictionary[TARGET_SOURCE]["descripcion"],
            "codificacion": {"0": "No tiene acceso", "1": "Tiene acceso"},
            "clase_positiva": 1,
            "nota_leakage": (
                "Las variables de Internet fijo y móvil no se utilizan como predictores "
                "para evitar fuga de información (data leakage)."
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
        "metricas": {"train": train_metrics, "test": test_metrics},
        "matriz_confusion_test": test_evaluation["matriz_confusion"],
        "classification_report_test": test_evaluation["classification_report"],
        "baseline": {
            "descripcion": "Predecir siempre la clase mayoritaria observada en train",
            "clase_mayoritaria": majority_class,
            "accuracy_test": baseline_accuracy,
        },
        "umbral_principal": 0.5,
        "roc": {
            "auc_test": test_metrics["roc_auc"],
            "puntos": _downsample_roc(y_test, test_evaluation["probabilidades"]),
        },
        "coeficientes_principales": top_coefficients.to_dict(orient="records"),
        "statsmodels": {
            "metodo": "statsmodels.api.Logit sobre X_train preprocesado",
            "convergio": bool(stats_result.mle_retvals.get("converged", False)),
            "pseudo_r2_mcfadden": float(stats_result.prsquared),
            "aic": float(stats_result.aic),
            "bic": float(stats_result.bic),
            "variables_significativas_0_05": significant.to_dict(orient="records"),
        },
        "interpretacion_primera_iteracion": interpretation,
    }

    if save_outputs:
        modeling_dir = project_root / "outputs" / "modelado"
        models_dir = project_root / "outputs" / "modelos"
        modeling_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = modeling_dir / "metricas_regresion_logistica.json"
        metrics_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2, default=_json_value),
            encoding="utf-8",
        )

        matrix = test_evaluation["matriz_confusion"]["matriz"]
        pd.DataFrame(
            matrix,
            index=["real_0_sin_internet", "real_1_con_internet"],
            columns=["predicho_0_sin_internet", "predicho_1_con_internet"],
        ).to_csv(modeling_dir / "matriz_confusion_test.csv", encoding="utf-8")

        coefficients.to_csv(
            modeling_dir / "coeficientes_logisticos.csv",
            index=False,
            encoding="utf-8",
        )

        predictions = pd.DataFrame(
            {
                "indice_registro_lapaz": index_test.to_numpy(),
                "valor_real": y_test.to_numpy(),
                "prediccion_umbral_0_5": test_evaluation["predicciones"],
                "probabilidad_acceso_internet": test_evaluation["probabilidades"],
            }
        )
        predictions.to_csv(
            modeling_dir / "predicciones_test.csv.gz",
            index=False,
            encoding="utf-8",
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )

        (modeling_dir / "resumen_statsmodels.txt").write_text(
            stats_result.summary().as_text(), encoding="utf-8"
        )
        joblib.dump(pipeline, models_dir / "regresion_logistica.joblib", compress=3)

    # Los vectores solo se exponen al notebook, no se serializan en el JSON.
    results["_runtime"] = {
        "pipeline": pipeline,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "index_train": index_train,
        "index_test": index_test,
        "coeficientes": coefficients,
        "statsmodels_result": stats_result,
    }
    return results


def resultado_serializable(results: dict[str, Any]) -> dict[str, Any]:
    """Retira objetos de ejecución para imprimir o reutilizar el resumen."""

    return {key: value for key, value in results.items() if key != "_runtime"}
