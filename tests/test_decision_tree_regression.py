"""Verifica denominadores, geografía y aislamiento de entrenamiento en H3_3."""

import numpy as np
import pandas as pd
import pytest

from src.decision_tree_regression import (
    FEATURES, TARGET, agregar_municipios, crear_pipeline, dividir_municipios,
)


def viviendas():
    return pd.DataFrame({
        "idep": ["02"] * 5, "iprov": ["01"] * 5, "imun": ["01"] * 5,
        "v01_tipoviv": ["1", "1", "1", "1", "7"],
        "v02_condocup": ["1", "1", "1", "2", "1"],
        "v19e_f": ["1", "2", "9", "1", "1"],
        "v19e_inetfijo": ["1", "2", "9", "1", "1"],
        "v19f_inetmovil": ["2", "2", "9", "2", "2"],
        "urbrur": ["1", "2", "2", "1", "1"],
        "v19c_compu": ["1", "2", "9", "1", "1"],
        "v19d_celular": ["1", "1", "9", "1", "1"],
        "v09_energia": ["1", "5", "3", "1", "1"],
        "v13_habitac": ["1", "3", "8", "1", "1"],
        "tot_pers": ["2", "4", "3", "1", "1"],
    })


def test_unknown_response_stays_in_denominator_and_inapplicable_rows_are_excluded():
    table, quality = agregar_municipios(viviendas(), {"20101": "La Paz"})
    row = table.iloc[0]
    assert row["universo"] == 3
    assert row["algun_si"] == row["algun_no"] == row["algun_sin_especificar"] == 1
    assert row[TARGET] == pytest.approx(100 / 3)
    assert row["pct_computadora"] == pytest.approx(100 / 3)
    # La categoría 8 (ocho o más) sí pertenece al indicador de tres o más.
    assert row["pct_3_o_mas_habitaciones"] == pytest.approx(200 / 3)
    assert row["promedio_personas"] == 3
    assert quality.set_index("variable").loc["v19c_compu", "sin_especificar"] == 1


def test_unmapped_geography_and_invalid_targets_do_not_disappear_in_aggregation():
    with pytest.raises(ValueError, match="sin correspondencia"):
        agregar_municipios(viviendas(), {})
    bad = viviendas()
    bad.loc[0, "v19e_f"] = ""
    with pytest.raises(ValueError, match="target censal"):
        agregar_municipios(bad, {"20101": "La Paz"})


def test_municipal_split_and_imputation_do_not_use_test_information():
    table = pd.DataFrame({name: np.arange(30, dtype=float) for name in FEATURES})
    table[TARGET] = np.arange(30, dtype=float)
    X_train, X_test, y_train, _ = dividir_municipios(table)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert len(X_test) == 9
    original = crear_pipeline(max_depth=2).fit(X_train, y_train)
    stats = original.named_steps["imputador"].statistics_.copy()
    table.loc[X_test.index, FEATURES] = 10_000
    train_again, _, target_again, _ = dividir_municipios(table)
    repeated = crear_pipeline(max_depth=2).fit(train_again, target_again)
    assert np.array_equal(stats, repeated.named_steps["imputador"].statistics_)
    assert TARGET not in FEATURES
