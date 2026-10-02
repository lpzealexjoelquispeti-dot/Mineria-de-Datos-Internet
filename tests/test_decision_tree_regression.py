"""Verificación estadística de H3_3 usando municipios reales del proyecto."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.tree import export_text

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

    def train_only(X, y, cv=None):
        pd.testing.assert_frame_equal(X, regression.seleccionar_predictores(prepared.dataset.loc[expected[0].index], list(X.columns)))
        pd.testing.assert_series_equal(y, expected[2])
        assert X.index.intersection(expected[1].index).empty
        observed.append((X.index.tolist(), cv, list(cv.split(X, y))))
        return real_search(X, y, cv=cv)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(regression, "preparar_dataset_modelo", lambda *args: prepared)
        patch.setattr(regression, "buscar_hiperparametros", train_only)
        result = regression.ejecutar_entrenamiento(destination)
    assert len(observed) == 2
    assert observed[0][0] == observed[1][0]
    assert observed[0][1] is observed[1][1]
    for (at, av), (bt, bv) in zip(observed[0][2], observed[1][2], strict=True):
        assert np.array_equal(at, bt) and np.array_equal(av, bv)
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
    assert isinstance(search.cv, KFold) and search.n_jobs == -1
    assert search.cv.n_splits == 5 and search.cv.shuffle is True
    assert search.cv.random_state == 777
    assert result["gridsearch"]["configuracion_cv"] == {
        "tipo": "KFold", "n_splits": 5, "shuffle": True, "random_state": 777}
    assert result["gridsearch"]["ajustes_cv_totales"] == 6080
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
    assert set(pipeline.feature_names_in_) == set(result["seleccion_features"]["variables"])


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
    assert list(table) == ["municipio_codigo", "municipio", *regression.DATASET_FEATURES, regression.TARGET]
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
    pd.testing.assert_frame_equal(
        first.dataset[regression.DATASET_FEATURES], second.dataset[regression.DATASET_FEATURES])
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


def test_room_percentage_validity_and_open_ended_category(prepared):
    assert prepared.dataset.pct_3_o_mas_habitaciones.between(0, 100).all()
    from src.pipeline import VARIABLES, read_dictionary
    raw = pd.read_csv(ROOT / "outputs/datos/vivienda_lapaz_seleccion.csv.gz", sep=";",
                      nrows=10000, usecols=VARIABLES, dtype="string", keep_default_na=False)
    raw = raw.loc[raw.v01_tipoviv.isin(["1", "2", "3", "4", "5", "6"]) & raw.v02_condocup.isin(["0", "1"])].copy()
    geography = raw.idep + raw.iprov + raw.imun
    code = geography.value_counts().index[0]
    raw = raw.loc[geography.eq(code)].iloc[:6].copy()
    assert len(raw) == 6
    assert (raw[["idep", "iprov", "imun"]].nunique() == 1).all()
    dictionary = read_dictionary(ROOT / "Base de datos CSV/Diccionario de variables CPV 2024.xlsx")
    code = raw.iloc[0].idep.lstrip("0") + raw.iloc[0].iprov + raw.iloc[0].imun
    eda = pd.DataFrame([{"municipio_codigo": code, "municipio": dictionary["mun_res_cod"]["categorias"][code],
                         "universo": len(raw), "pct_algun": raw.v19e_f.eq("1").mean() * 100}])
    raw["v13_habitac"] = ["1", "2", "3", "8", "9", ""]
    first = regression.construir_dataset_municipal(raw, dictionary, eda)
    assert first.dataset.pct_3_o_mas_habitaciones.iloc[0] == pytest.approx(50)
    assert first.audit["conteos_validos_municipales"][0]["validos_pct_3_o_mas_habitaciones"] == 4
    raw["v13_habitac"] = ["9", "", "0", "99", "-1", "3.5"]
    second = regression.construir_dataset_municipal(raw, dictionary, eda)
    assert pd.isna(second.dataset.pct_3_o_mas_habitaciones.iloc[0])
    assert second.audit["conteos_validos_municipales"][0]["validos_pct_3_o_mas_habitaciones"] == 0


def test_selection_uses_cv_only_and_documented_tie():
    rows = [{"variante": "A", "best_mse_cv": 20, "metricas_test": {"mse": 0}},
            {"variante": "B", "best_mse_cv": 19, "metricas_test": {"mse": 99999}}]
    assert regression.seleccionar_variante(rows)["variante"] == "B"
    rows[0]["metricas_test"], rows[1]["metricas_test"] = rows[1]["metricas_test"], rows[0]["metricas_test"]
    assert regression.seleccionar_variante(rows)["variante"] == "B"
    rows[1]["best_mse_cv"] = 20 + 1e-7
    selection = regression.seleccionar_variante(rows)
    assert selection["variante"] == "B" and selection["empate_equivalente"]
    rows[1]["best_mse_cv"] = 21
    assert regression.seleccionar_variante(rows)["variante"] == "A"


def test_selection_precedes_single_final_tree_test_evaluation(monkeypatch, prepared, tmp_path):
    selected = []
    evaluated = []
    actual_select, actual_evaluate = regression.seleccionar_variante, regression.evaluar_modelo
    test_index = regression.dividir_train_test(prepared.X, prepared.y)[1].index

    def select(rows):
        assert all("metricas_test" not in row for row in rows)
        selected.append(True)
        return actual_select(rows)

    def evaluate(model, X, y):
        if X.index.equals(test_index):
            assert selected == [True]
            if hasattr(model, "named_steps"):
                evaluated.append(model)
        return actual_evaluate(model, X, y)

    # Espías vigilan el orden del flujo y una sola evaluación test del árbol.
    monkeypatch.setattr(regression, "preparar_dataset_modelo", lambda *args: prepared)
    monkeypatch.setattr(regression, "seleccionar_variante", select)
    monkeypatch.setattr(regression, "evaluar_modelo", evaluate)
    regression.ejecutar_entrenamiento(tmp_path, save_outputs=False)
    assert len(evaluated) == 1


def test_sha256_deterministic_changed_content_and_previous_manifest(tmp_path):
    from src.pipeline import sha256
    from src.census_sources import verificar_archivos
    path = tmp_path / "tiny.csv"
    path.write_bytes(b"col\n1\n")
    first = verificar_archivos({"tiny.csv": path}, root=tmp_path)
    assert first["tiny.csv"]["sha256"] == sha256(path) == sha256(path)
    assert first["tiny.csv"]["bytes"] == 6
    assert verificar_archivos({"tiny.csv": path}, first, tmp_path) == first
    path.write_bytes(b"col\n2\n")  # Mismo tamaño, distinto contenido.
    assert sha256(path) != first["tiny.csv"]["sha256"]
    with pytest.raises(ValueError, match="manifiesto previo"):
        verificar_archivos({"tiny.csv": path}, first, tmp_path)


def test_source_verifier_checks_eda_and_does_not_overwrite_changed_source(tmp_path):
    import json
    from src.census_sources import SOURCE_NAMES, MANIFEST_PATH, verificar_fuentes_censo
    source = tmp_path / "Base de datos CSV"
    source.mkdir()
    for name in SOURCE_NAMES:
        (source / name).write_bytes(b"small fixture")
    first = verificar_fuentes_censo(tmp_path)
    manifest = tmp_path / MANIFEST_PATH
    original = manifest.read_bytes()
    assert verificar_fuentes_censo(tmp_path) == first
    manifest.unlink()
    (tmp_path / "outputs/resumen_eda.json").write_text(json.dumps({"fuentes": first}))
    (source / SOURCE_NAMES[0]).write_bytes(b"changed source")
    with pytest.raises(ValueError, match="resumen_eda"):
        verificar_fuentes_censo(tmp_path)
    assert not manifest.exists()
    manifest.write_bytes(original)
    with pytest.raises(ValueError, match="fuentes_censo_sha256"):
        verificar_fuentes_censo(tmp_path)
    assert manifest.read_bytes() == original


def test_exported_rules_match_serialized_final_model_and_features(run):
    result, destination = run
    restored = joblib.load(destination / result["artefactos"]["modelo"])
    model = restored.named_steps["modelo"]
    features = result["seleccion_features"]["variables"]
    assert list(restored.feature_names_in_) == features
    expected = export_text(model, feature_names=features, decimals=6, max_depth=max(1, model.get_depth()))
    assert (destination / result["artefactos"]["reglas"]).read_text() == expected
    runtime = result["_runtime"]
    assert runtime["pipeline"] is runtime["search"].best_estimator_
    row = runtime["X_test"].iloc[:1]
    explanation = regression.explicar_prediccion(restored, row)
    assert explanation["prediccion_final"] == pytest.approx(restored.predict(row)[0])
    tree = model.tree_
    node = 0
    for step in explanation["ruta"]:
        assert step["nodo"] == node
        assert step["feature"] == features[tree.feature[node]]
        assert step["umbral"] == tree.threshold[node]
        node = tree.children_left[node] if step["condicion"] == "<=" else tree.children_right[node]
    assert node == explanation["hoja"]


def test_saved_cv_comparison_and_fold_tables_match_results(run):
    import json
    result, destination = run
    comparison = pd.read_csv(destination / result["artefactos"]["comparacion_features"])
    assert comparison.variante.tolist() == ["A", "B"]
    assert comparison.seleccionada.sum() == 1
    for row, expected in zip(comparison.to_dict("records"), result["comparacion_features"], strict=True):
        assert json.loads(row["variables"]) == expected["variables"]
        assert json.loads(row["best_params"]) == expected["best_params"]
        assert row["best_mse_cv"] == pytest.approx(expected["best_mse_cv"])
        assert row["std_mse_cv"] == pytest.approx(expected["std_mse_cv"])
    folds = pd.read_csv(destination / result["artefactos"]["folds_cv"])
    assert folds.fold.tolist() == list(range(1, 6))
    assert np.allclose(folds.mse_validacion, result["gridsearch"]["mse_folds"])
    grid = pd.read_csv(destination / result["artefactos"]["gridsearch"])
    assert grid.groupby("variante").size().to_dict() == {"A": 608, "B": 608}
