from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.decision_tree_classification import (
    CATEGORICAL_FEATURES,
    INTERNET_VARIABLES,
    MODEL_FEATURES,
    crear_grid_regularizacion,
    crear_grid_profundidad,
    crear_pipeline,
    evaluar_modelo,
    obtener_importancias,
)
from src.logistic_regression import crear_target, dividir_train_test


def synthetic_dataset(rows: int = 120) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(777)
    frame = pd.DataFrame(
        {
            "urbrur": rng.choice(["1", "2"], rows),
            "v01_tipoviv": rng.choice(["1", "2", "3"], rows),
            "v09_energia": rng.choice(["1", "2", None], rows),
            "v19c_compu": rng.choice(["1", "2", None], rows),
            "v19d_celular": rng.choice(["1", "2", None], rows),
            "v13_habitac": rng.choice([1.0, 2.0, 3.0, np.nan], rows),
            "tot_pers": rng.choice([1.0, 2.0, 4.0, np.nan], rows),
        }
    )
    signal = (
        frame["urbrur"].eq("1").astype(int)
        + frame["v19c_compu"].eq("1").astype(int)
        + rng.integers(0, 2, rows)
    )
    target = signal.ge(2).astype("int8").rename("TIENE_ACCESO_INTERNET")
    return frame, target


def test_target_binary_and_predictors_do_not_leak() -> None:
    target = crear_target(pd.Series(["1", "2", "9", ""]))
    assert set(target.dropna().astype(int)) == {0, 1}
    assert set(MODEL_FEATURES).isdisjoint(INTERNET_VARIABLES)
    assert "TIENE_ACCESO_INTERNET" not in MODEL_FEATURES
    assert CATEGORICAL_FEATURES == [
        "urbrur",
        "v01_tipoviv",
        "v09_energia",
        "v19c_compu",
        "v19d_celular",
    ]


def test_split_is_reproducible() -> None:
    X, y = synthetic_dataset()
    indices = X.index
    first = dividir_train_test(X, y, indices)
    second = dividir_train_test(X, y, indices)
    assert first[4].equals(second[4])
    assert first[5].equals(second[5])
    assert first[3].value_counts().to_dict() == second[3].value_counts().to_dict()


def test_grid_never_uses_invalid_min_samples_split() -> None:
    depth_grid = crear_grid_profundidad()
    grid = crear_grid_regularizacion(best_depth=4)
    assert depth_grid["modelo__max_depth"] == list(range(1, 16))
    assert min(grid["modelo__min_samples_split"]) >= 2
    assert grid["modelo__min_samples_split"] == list(range(2, 10))
    assert grid["modelo__min_samples_leaf"] == list(range(1, 6))


def test_pipeline_probabilities_metrics_confusion_importance_and_serialization(tmp_path) -> None:
    X, y = synthetic_dataset()
    pipeline = crear_pipeline()
    pipeline.set_params(modelo__max_depth=4, modelo__min_samples_split=2)
    pipeline.fit(X, y)

    probabilities = pipeline.predict_proba(X)
    assert probabilities.shape == (len(X), 2)
    assert np.allclose(probabilities.sum(axis=1), 1)

    evaluation = evaluar_modelo(pipeline, X, y)
    assert all(0 <= value <= 1 for value in evaluation["metricas"].values())
    assert np.asarray(evaluation["matriz_confusion"]["matriz"]).shape == (2, 2)

    transformed, aggregated = obtener_importancias(pipeline)
    assert np.isclose(transformed["importancia"].sum(), 1)
    assert np.isclose(aggregated["importancia"].sum(), 1)
    assert set(aggregated["variable_original"]).issubset(MODEL_FEATURES)

    model_path = tmp_path / "tree.joblib"
    joblib.dump(pipeline, model_path)
    restored = joblib.load(model_path)
    assert np.allclose(restored.predict_proba(X), probabilities)
