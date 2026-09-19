"""Funciones estadísticas compartidas por el pipeline y el notebook.

Este módulo concentra las reglas de agregación. FastAPI no replica estas
operaciones: únicamente sirve los artefactos pequeños generados por el pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


INTERNET_VARIABLES = {
    "fijo": "v19e_inetfijo",
    "movil": "v19f_inetmovil",
    "algun": "v19e_f",
}


@dataclass(frozen=True)
class IqrResult:
    q1: float
    q3: float
    iqr: float
    limite_inferior: float
    limite_superior: float


def summarize_groups(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Resume acceso TIC con los mismos denominadores usados por el EDA."""

    grouped = data.groupby(columns, observed=True, dropna=False)
    table = grouped.size().rename("universo").to_frame()

    for label, variable in INTERNET_VARIABLES.items():
        counts = (
            data.groupby(columns + [variable], observed=True, dropna=False)
            .size()
            .unstack(variable, fill_value=0)
        )
        for code, suffix in (("1", "si"), ("2", "no"), ("9", "sin_especificar")):
            empty = pd.Series(0, index=table.index, dtype="int64")
            table[f"{label}_{suffix}"] = counts.get(code, empty).reindex(
                table.index, fill_value=0
            )

        table[f"validos_{label}"] = table[f"{label}_si"] + table[f"{label}_no"]
        table[f"pct_{label}"] = table[f"{label}_si"] / table["universo"] * 100
        table[f"pct_{label}_determinados"] = (
            table[f"{label}_si"]
            / table[f"validos_{label}"].replace(0, np.nan)
            * 100
        )

    table["pct_sin_internet"] = table["algun_no"] / table["universo"] * 100
    table["pct_sin_especificar"] = (
        table["algun_sin_especificar"] / table["universo"] * 100
    )
    table["brecha_movil_fijo_pp"] = table["pct_movil"] - table["pct_fijo"]
    return table


def build_comparison(data: pd.DataFrame) -> pd.DataFrame:
    """Construye el resumen departamental fijo, móvil y combinado."""

    labels = {"Fijo": "v19e_inetfijo", "Móvil": "v19f_inetmovil", "Algún internet": "v19e_f"}
    rows: list[dict[str, int | str]] = []
    universe = len(data)
    for label, variable in labels.items():
        series = data[variable]
        rows.append(
            {
                "tipo": label,
                "Sí": int(series.eq("1").sum()),
                "No": int(series.eq("2").sum()),
                "Sin especificar": int(series.eq("9").sum()),
                "universo": universe,
                "validos_si_no": int(series.isin(["1", "2"]).sum()),
            }
        )

    table = pd.DataFrame(rows).set_index("tipo")
    for column in ("Sí", "No", "Sin especificar"):
        table[f"% {column}"] = table[column] / table["universo"] * 100
    table["% Sí entre determinados"] = table["Sí"] / table["validos_si_no"] * 100
    return table


def detect_iqr_outliers(
    table: pd.DataFrame, metric: str = "pct_algun"
) -> tuple[pd.DataFrame, IqrResult]:
    """Aplica la regla de Tukey (1,5 × IQR) sin recortar límites a 0–100."""

    values = table[metric].astype(float)
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    result = IqrResult(
        q1=float(q1),
        q3=float(q3),
        iqr=float(iqr),
        limite_inferior=float(lower),
        limite_superior=float(upper),
    )
    return table[(values < lower) | (values > upper)], result


def cramers_v_2x2(table: pd.DataFrame) -> float:
    """Calcula V de Cramér sobre respuestas determinadas de una tabla 2×2."""

    observed = table[["Sí", "No"]].to_numpy(dtype=float)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    if not (expected > 0).all():
        return 0.0
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    return float(np.sqrt(chi2 / observed.sum()))

