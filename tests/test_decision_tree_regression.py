"""Verificación estadística de H3_3 usando municipios reales del proyecto."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import ParameterGrid

from src import decision_tree_regression as regression

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def prepared():
    return regression.preparar_dataset_modelo(ROOT)


@pytest.fixture(scope="module")
def run(prepared, tmp_path_factory):
    destination = tmp_path_factory.mktemp("h33")
    real_search = regression.buscar_hiperparametros
    expected = regression.dividir_train_test(prepared.X, prepared.y)
    observed = []

    def train_only(X, y):
        pd.testing.assert_frame_equal(X, expected[0])
        pd.testing.assert_series_equal(y, expected[2])
        assert X.index.intersection(expected[1].index).empty
        observed.append(X.index.tolist())
        return real_search(X, y)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(regression, "preparar_dataset_modelo", lambda *args: prepared)
        patch.setattr(regression, "buscar_hiperparametros", train_only)
        result = regression.ejecutar_entrenamiento(destination)
    assert len(observed) == 1
    return result, destination


def test_one_numeric_continuous_target_per_municipality(prepared):
    table = prepared.dataset
    assert len(table) == len(table.municipio_codigo.unique())
    assert not table.municipio_codigo.duplicated().any()
    assert pd.api.types.is_float_dtype(table[regression.TARGET])
    assert table[regression.TARGET].between(0, 100).all()
    assert not table[regression.TARGET].isna().any()
    assert table[regression.TARGET].nunique() > 2
    assert table.municipio_codigo.tolist() == sorted(table.municipio_codigo.tolist())


def test_target_and_denominators_match_eda(prepared):
    eda = pd.read_csv(ROOT / "outputs/tablas/municipios.csv", dtype={"municipio_codigo": str})
    audit = regression.validar_target_eda(prepared.dataset, eda)
    assert audit["coincide"]
    assert audit["max_diferencia_pp"] < 1e-8
    counts = pd.DataFrame(prepared.audit["conteos_validos_municipales"])
    merged = counts.merge(eda, on="municipio_codigo", suffixes=("", "_eda"))
    assert np.array_equal(merged.universo, merged.universo_eda)
    # Contraste independiente del porcentaje, incluidos los indeterminados.
    actual = prepared.dataset.set_index("municipio_codigo")[regression.TARGET]
    expected = eda.set_index("municipio_codigo").eval("algun_si / universo * 100")
    assert np.allclose(actual.sort_index(), expected.sort_index(), atol=1e-8, rtol=0)


@pytest.mark.parametrize("damage", ["value", "duplicate", "missing", "nan", "range", "name"])
def test_invalid_target_or_geography_stops(damage, prepared):
    dataset = prepared.dataset.copy()
    eda = pd.read_csv(ROOT / "outputs/tablas/municipios.csv", dtype={"municipio_codigo": str})
    if damage == "value":
        dataset.loc[0, regression.TARGET] += .01
    elif damage == "duplicate":
        dataset.loc[1, "municipio_codigo"] = dataset.loc[0, "municipio_codigo"]
    elif damage == "missing":
        dataset = dataset.iloc[1:]
    elif damage == "nan":
        dataset.loc[0, regression.TARGET] = np.nan
    elif damage == "range":
        dataset.loc[0, regression.TARGET] = 101
    else:
        dataset.loc[0, "municipio"] = "Nombre inconsistente"
    with pytest.raises(ValueError):
        regression.validar_target_eda(dataset, eda)


def test_internet_and_identifiers_never_selected(prepared):
    dataset = prepared.dataset.copy()
    for variable in regression.FORBIDDEN_FEATURES:
        if variable not in dataset:
            dataset[variable] = 99
    X = regression.seleccionar_predictores(dataset)
    assert X.columns.tolist() == regression.MODEL_FEATURES
    assert set(X).isdisjoint(regression.FORBIDDEN_FEATURES)
    pd.testing.assert_frame_equal(X, prepared.X)


def test_leaking_feature_configuration_rejected(monkeypatch, prepared):
    monkeypatch.setattr(regression, "MODEL_FEATURES", [*regression.MODEL_FEATURES, "pct_algun"])
    with pytest.raises(ValueError, match="fuga"):
        regression.seleccionar_predictores(prepared.dataset)


def test_dictionary_energy_and_valid_counts(prepared):
    assert set(prepared.audit["energia_codigos_con_disponibilidad"]) == {"1", "2", "3", "4"}
    assert prepared.audit["energia_codigo_ausencia"] == "5"
    counts = pd.DataFrame(prepared.audit["conteos_validos_municipales"])
    for column in counts.filter(like="validos_"):
        assert (counts[column] <= counts.universo).all()
    assert (counts.validos_pct_computadora < counts.universo).any()
    assert (counts.validos_pct_celular < counts.universo).any()


def test_reproducible_split_without_stratify(monkeypatch, prepared):
    original_split = regression.train_test_split
    calls = []

    def spy(*args, **kwargs):
        assert "stratify" not in kwargs
        calls.append(kwargs)
        return original_split(*args, **kwargs)

    monkeypatch.setattr(regression, "train_test_split", spy)
    first = regression.dividir_train_test(prepared.X, prepared.y)
    second = regression.dividir_train_test(prepared.X, prepared.y)
    for a, b in zip(first, second, strict=True):
        assert a.equals(b)
    assert first[0].index.intersection(first[1].index).empty
    assert len(first[0]) + len(first[1]) == len(prepared.dataset)
    assert calls[0] == {"test_size": .2, "random_state": 777}


def test_full_grid_valid_and_train_only(run):
    result, _ = run
    grid = regression.crear_grid()
    assert grid["modelo__max_depth"] == list(range(1, 20))
    assert grid["modelo__min_samples_split"] == list(range(2, 10))
    assert grid["modelo__min_samples_leaf"] == list(range(1, 5))
    assert len(ParameterGrid(grid)) == 608
    runtime = result["_runtime"]
    search = runtime["search"]
    assert search.cv == 5 and search.n_jobs == -1
    assert search.scoring == "neg_mean_squared_error"
    assert len(search.cv_results_["params"]) == 608
    assert result["gridsearch"]["ajustes_cv"] == 3040
    assert result["gridsearch"]["mejor_mse_cv"] == pytest.approx(np.mean(result["gridsearch"]["mse_folds"]))
    assert result["gridsearch"]["std_mse_cv"] == pytest.approx(np.std(result["gridsearch"]["mse_folds"]))


def test_pipeline_predictions_median_importance_and_serialization(run):
    result, destination = run
    runtime = result["_runtime"]
    pipeline = runtime["pipeline"]
    assert list(pipeline.named_steps) == ["imputador", "modelo"]
    assert pipeline.named_steps["modelo"].__class__.__name__ == "DecisionTreeRegressor"
    assert np.allclose(pipeline.named_steps["imputador"].statistics_, runtime["X_train"].median())
    predictions = pipeline.predict(runtime["X_test"])
    assert np.isfinite(predictions).all()
    assert np.issubdtype(predictions.dtype, np.number)
    assert result["suma_importancias"] == pytest.approx(1)
    restored = joblib.load(destination / result["artefactos"]["modelo"])
    assert np.allclose(restored.predict(runtime["X_test"]), predictions)
    assert set(pipeline.feature_names_in_) == set(regression.MODEL_FEATURES)


def test_metrics_nonnegative_errors_and_preserve_negative_r2(prepared, run):
    result, _ = run
    for metrics in [*result["metricas"].values(), result["baseline"]["metricas_test"]]:
        assert all(metrics[key] >= 0 for key in ("mse", "mae", "rmse"))
        assert metrics["rmse"] ** 2 == pytest.approx(metrics["mse"])
        assert set(metrics) == {"mse", "rmse", "mae", "r2"}
    bad = regression.calcular_metricas(prepared.y, np.full(len(prepared.y), 100.0))
    assert bad["r2"] < 0


def test_predictions_correct_municipalities_and_baseline(run):
    result, destination = run
    runtime = result["_runtime"]
    predictions = pd.read_csv(destination / result["artefactos"]["predicciones"], dtype={"municipio_codigo": str})
    expected = runtime["prepared"].dataset.loc[runtime["y_test"].index].set_index("municipio_codigo")
    assert set(predictions.municipio_codigo) == set(expected.index)
    aligned = predictions.set_index("municipio_codigo").loc[expected.index]
    assert np.allclose(aligned.valor_real, expected[regression.TARGET])
    assert np.allclose(aligned.valor_predicho, runtime["pipeline"].predict(runtime["X_test"]))
    assert np.allclose(predictions.error, predictions.valor_real - predictions.valor_predicho)
    assert np.allclose(predictions.error_absoluto, predictions.error.abs())
    assert result["baseline"]["media_train"] == pytest.approx(runtime["y_train"].mean())
    assert np.allclose(runtime["baseline"].predict(runtime["X_test"]), runtime["y_train"].mean())
    split = pd.read_csv(destination / result["artefactos"]["particion"], dtype={"municipio_codigo": str})
    assert not split.municipio_codigo.duplicated().any()
    assert set(split.query("particion == 'test'").municipio_codigo) == set(predictions.municipio_codigo)


def test_all_artifacts_exist_and_no_individual_microdata(run):
    result, destination = run
    for path in result["artefactos"].values():
        assert (destination / path).is_file() and (destination / path).stat().st_size > 0
    table = pd.read_csv(destination / result["artefactos"]["dataset"])
    assert list(table) == ["municipio_codigo", "municipio", *regression.MODEL_FEATURES, regression.TARGET]
    assert len(table) == result["registros"]["municipios"]


def test_insufficient_cv_data_rejected(prepared):
    with pytest.raises(ValueError, match="cinco"):
        regression.buscar_hiperparametros(prepared.X.iloc[:4], prepared.y.iloc[:4])


def test_aggregation_never_uses_internet_to_construct_features():
    from src.pipeline import VARIABLES, read_dictionary
    raw = pd.read_csv(ROOT / "outputs/datos/vivienda_lapaz_seleccion.csv.gz", sep=";",
                      nrows=1000, usecols=VARIABLES, dtype="string", keep_default_na=False)
    dictionary = read_dictionary(ROOT / "Base de datos CSV/Diccionario de variables CPV 2024.xlsx")

    def reference(frame):
        valid = frame.v01_tipoviv.isin(["1", "2", "3", "4", "5", "6"]) & frame.v02_condocup.isin(["0", "1"])
        selected = frame.loc[valid].copy()
        selected["municipio_codigo"] = selected.idep.str.lstrip("0") + selected.iprov + selected.imun
        groups = selected.groupby("municipio_codigo")
        table = groups.size().to_frame("universo")
        table["pct_algun"] = groups.v19e_f.apply(lambda x: x.eq("1").sum() / len(x) * 100)
        table["municipio"] = table.index.map(dictionary["mun_res_cod"]["categorias"])
        return table.reset_index()

    first = regression.construir_dataset_municipal(raw, dictionary, reference(raw))
    changed = raw.copy()
    for variable in ["v19e_inetfijo", "v19f_inetmovil", "v19e_f"]:
        changed[variable] = "2"
    second = regression.construir_dataset_municipal(changed, dictionary, reference(changed))
    pd.testing.assert_frame_equal(first.X, second.X)
    assert second.y.eq(0).all()
    assert first.y.gt(0).any()


def test_aggregation_preserves_missing_equipment_and_numeric_cleaning():
    from src.pipeline import VARIABLES, read_dictionary
    raw = pd.read_csv(ROOT / "outputs/datos/vivienda_lapaz_seleccion.csv.gz", sep=";",
                      nrows=1000, usecols=VARIABLES, dtype="string", keep_default_na=False)
    raw = raw.loc[raw.v01_tipoviv.isin(["1", "2", "3", "4", "5", "6"]) & raw.v02_condocup.isin(["0", "1"])].copy()
    dictionary = read_dictionary(ROOT / "Base de datos CSV/Diccionario de variables CPV 2024.xlsx")
    # Se alteran valores de registros reales solo para probar la limpieza, sin añadir observaciones.
    index = raw.index[:4]
    raw.loc[index, "v09_energia"] = ["2", "3", "4", "5"]
    raw.loc[index, "v19c_compu"] = ["1", "2", "9", ""]
    raw.loc[index, "v19d_celular"] = ["1", "2", "9", ""]
    raw.loc[index, "v13_habitac"] = ["1", "8", "9", ""]
    raw.loc[index, "tot_pers"] = ["0", "9999", "10000", ""]
    raw["municipio_codigo"] = raw.idep.str.lstrip("0") + raw.iprov + raw.imun
    groups = raw.groupby("municipio_codigo")
    eda = groups.size().to_frame("universo")
    eda["pct_algun"] = groups.v19e_f.apply(lambda x: x.eq("1").sum() / len(x) * 100)
    eda["municipio"] = eda.index.map(dictionary["mun_res_cod"]["categorias"])
    built = regression.construir_dataset_municipal(raw, dictionary, eda.reset_index())
    selected_code = raw.loc[index[0], "municipio_codigo"]
    group = raw.loc[raw.municipio_codigo.eq(selected_code)]
    row = built.dataset.set_index("municipio_codigo").loc[selected_code]
    expected_compu = group.v19c_compu.eq("1").sum() / group.v19c_compu.isin(["1", "2"]).sum() * 100
    assert row.pct_computadora == pytest.approx(expected_compu)
    assert row.pct_con_energia == pytest.approx(group.v09_energia.isin(["1", "2", "3", "4"]).mean() * 100)
    rooms = pd.to_numeric(group.v13_habitac, errors="coerce")
    people = pd.to_numeric(group.tot_pers, errors="coerce")
    assert row.promedio_habitaciones == pytest.approx(rooms.loc[rooms.between(1, 8)].mean())
    assert row.promedio_personas == pytest.approx(people.loc[people.between(0, 9999)].mean())
