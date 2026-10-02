"""Adaptación de resultados mineros ya calculados a respuestas de API."""

from __future__ import annotations

from typing import Any


METRIC_COLUMNS = {
    "internet": "pct_algun",
    "fijo": "pct_fijo",
    "movil": "pct_movil",
    "sin_internet": "pct_sin_internet",
}


def metric_column(metric: str) -> str:
    return METRIC_COLUMNS[metric]


def sort_territories(
    rows: list[dict[str, Any]], metric: str, order: str, limit: int
) -> list[dict[str, Any]]:
    column = metric_column(metric)
    return sorted(
        rows,
        key=lambda row: float(row[column]),
        reverse=order == "mayor",
    )[:limit]

