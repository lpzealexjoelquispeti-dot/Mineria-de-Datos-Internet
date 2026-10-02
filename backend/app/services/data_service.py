"""Lee y cachea los artefactos pequeños generados por ``src.pipeline``."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.schemas.responses import (
    ArbolClasificacionResponse,
    AreaConectividad,
    AreaResponse,
    CalidadResponse,
    DistribucionItem,
    DistribucionResponse,
    HallazgosResponse,
    Indicador,
    MetadataResponse,
    Outlier,
    OutliersResponse,
    RegresionLogisticaResponse,
    ResumenResponse,
    TerritorioConectividad,
    TerritoriosResponse,
    ValorFaltante,
    VariableMetadata,
)
from app.services.mining_service import metric_column, sort_territories


def _integer(value: Any) -> int:
    return int(float(value))


def _number(value: Any) -> float:
    return float(value)


def _indicator(
    quantity: Any,
    percentage: Any,
    determined: Any | None = None,
) -> Indicador:
    return Indicador(
        cantidad=_integer(quantity),
        porcentaje=_number(percentage),
        porcentaje_determinados=(
            None if determined in (None, "", "nan") else _number(determined)
        ),
    )


class DataService:
    def __init__(self, outputs_dir: Path) -> None:
        self.outputs_dir = outputs_dir
        self.tables_dir = outputs_dir / "tablas"

    @lru_cache(maxsize=1)
    def summary(self) -> dict[str, Any]:
        path = self.outputs_dir / "resumen_eda.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"No existe {path}. Ejecute: python -m src.pipeline"
            )
        return json.loads(path.read_text(encoding="utf-8"))

    @lru_cache(maxsize=1)
    def logistic_summary(self) -> dict[str, Any]:
        path = self.outputs_dir / "modelado" / "metricas_regresion_logistica.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"No existe {path}. Ejecute: python scripts/entrenar_regresion_logistica.py"
            )
        return json.loads(path.read_text(encoding="utf-8"))

    @lru_cache(maxsize=1)
    def decision_tree_summary(self) -> dict[str, Any]:
        path = self.outputs_dir / "modelado" / "metricas_arbol_clasificacion.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"No existe {path}. Ejecute: python scripts/entrenar_arbol_clasificacion.py"
            )
        return json.loads(path.read_text(encoding="utf-8"))

    @lru_cache(maxsize=32)
    def table(self, name: str) -> tuple[dict[str, str], ...]:
        path = self.tables_dir / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"No existe {path}. Ejecute: python -m src.pipeline"
            )
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            return tuple(dict(row) for row in csv.DictReader(file))

    def get_summary(self) -> ResumenResponse:
        summary = self.summary()
        comparison = {row["tipo"]: row for row in summary["comparacion"]}
        fixed = comparison["Fijo"]
        mobile = comparison["Móvil"]
        combined = comparison["Algún internet"]
        universe = _integer(summary["universo_tic"])
        return ResumenResponse(
            total_registros=_integer(summary["registros_lapaz"]),
            registros_validos=universe,
            registros_fuera_universo=_integer(summary["fuera_universo"]),
            internet_fijo=_indicator(
                fixed["Sí"], fixed["% Sí"], fixed["% Sí entre determinados"]
            ),
            internet_movil=_indicator(
                mobile["Sí"], mobile["% Sí"], mobile["% Sí entre determinados"]
            ),
            algun_internet=_indicator(
                combined["Sí"], combined["% Sí"], combined["% Sí entre determinados"]
            ),
            sin_internet=_indicator(combined["No"], combined["% No"]),
            sin_especificar=_indicator(
                combined["Sin especificar"], combined["% Sin especificar"]
            ),
        )

    @staticmethod
    def territory(row: dict[str, Any], kind: str) -> TerritorioConectividad:
        return TerritorioConectividad(
            codigo=str(row[f"{kind}_codigo"]),
            nombre=row.get(kind) or None,
            total=_integer(row["universo"]),
            internet_fijo=_indicator(
                row["fijo_si"], row["pct_fijo"], row["pct_fijo_determinados"]
            ),
            internet_movil=_indicator(
                row["movil_si"], row["pct_movil"], row["pct_movil_determinados"]
            ),
            algun_internet=_indicator(
                row["algun_si"], row["pct_algun"], row["pct_algun_determinados"]
            ),
            sin_internet=_indicator(row["algun_no"], row["pct_sin_internet"]),
            sin_especificar=_indicator(
                row["algun_sin_especificar"], row["pct_sin_especificar"]
            ),
        )

    def get_areas(self) -> AreaResponse:
        summary = self.summary()
        items = []
        for row in summary["areas"]:
            items.append(
                AreaConectividad(
                    area=row["area_label"],
                    total=_integer(row["universo"]),
                    internet_fijo=_indicator(
                        row["fijo_si"], row["pct_fijo"], row["pct_fijo_determinados"]
                    ),
                    internet_movil=_indicator(
                        row["movil_si"], row["pct_movil"], row["pct_movil_determinados"]
                    ),
                    algun_internet=_indicator(
                        row["algun_si"], row["pct_algun"], row["pct_algun_determinados"]
                    ),
                    sin_internet=_indicator(row["algun_no"], row["pct_sin_internet"]),
                    sin_especificar=_indicator(
                        row["algun_sin_especificar"], row["pct_sin_especificar"]
                    ),
                )
            )
        return AreaResponse(
            items=items,
            brecha_urbano_rural_pp=_number(summary["brecha_urbano_rural_pp"]),
            v_cramer=_number(summary["v_cramer"]),
        )

    def get_territories(
        self,
        kind: str,
        limit: int,
        order: str,
        metric: str,
        area: str = "todos",
    ) -> TerritoriosResponse:
        table_name = f"{kind}s"
        rows = list(self.table(table_name))
        if kind == "municipio" and area != "todos":
            rows = list(self.table("municipios_area"))
            expected = "Urbana" if area == "urbana" else "Rural"
            rows = [row for row in rows if row["area_label"] == expected]
        selected = sort_territories(rows, metric, order, limit)
        return TerritoriosResponse(
            items=[self.territory(row, kind) for row in selected],
            total=len(rows),
            limit=limit,
            orden=order,
            metrica=metric,
            area=area,
        )

    def get_outliers(self) -> OutliersResponse:
        summary = self.summary()
        limits = summary["iqr"]
        items = [
            Outlier(
                codigo=str(row["municipio_codigo"]),
                nombre=row["municipio"],
                metrica="Porcentaje de viviendas con algún Internet",
                valor=_number(row["pct_algun"]),
                metodo="Regla de Tukey (1,5 × IQR)",
                q1=_number(limits["Q1"]),
                q3=_number(limits["Q3"]),
                iqr=_number(limits["IQR"]),
                limite_inferior=_number(limits["Límite inferior"]),
                limite_superior=_number(limits["Límite superior"]),
            )
            for row in summary["outliers"]
        ]
        return OutliersResponse(
            descripcion=(
                "Un territorio atípico presenta un valor inusual frente a la distribución "
                "municipal. Es una señal para explorar, no la identificación de un error."
            ),
            items=items,
        )

    def get_distribution(self, metric: str, area: str) -> DistribucionResponse:
        rows = list(self.table("municipios"))
        if area != "todos":
            rows = list(self.table("municipios_area"))
            expected = "Urbana" if area == "urbana" else "Rural"
            rows = [row for row in rows if row["area_label"] == expected]
        column = metric_column(metric)
        values = sorted(_number(row[column]) for row in rows)
        middle = len(values) // 2
        median = (
            values[middle]
            if len(values) % 2
            else (values[middle - 1] + values[middle]) / 2
        )
        return DistribucionResponse(
            metrica=metric,
            area=area,
            items=[
                DistribucionItem(
                    codigo=str(row["municipio_codigo"]),
                    nombre=row["municipio"],
                    valor=_number(row[column]),
                )
                for row in rows
            ],
            mediana=median,
        )

    def get_quality(self) -> CalidadResponse:
        summary = self.summary()
        rows = [
            ValorFaltante(
                variable=row["variable"],
                nulos_pandas=_integer(row["nulos_pandas"]),
                vacios_csv=_integer(row["vacios_csv"]),
                solo_espacios=_integer(row["solo_espacios"]),
                marcadores_nan_null_na=_integer(row["marcadores_NaN_NULL_NA"]),
                sin_especificar=_integer(row["sin_especificar"]),
                no_aplica_codificado=_integer(row["no_aplica_codificado"]),
            )
            for row in self.table("faltantes")
        ]
        comparison = {row["tipo"]: row for row in summary["comparacion"]}
        return CalidadResponse(
            valores_faltantes=rows,
            duplicados={key: _integer(value) for key, value in summary["duplicados"].items()},
            inconsistencias={
                key: _integer(value) for key, value in summary["inconsistencias"].items()
            },
            codigos_sin_especificar={
                "internet_fijo": _integer(comparison["Fijo"]["Sin especificar"]),
                "internet_movil": _integer(comparison["Móvil"]["Sin especificar"]),
                "algun_internet": _integer(
                    comparison["Algún internet"]["Sin especificar"]
                ),
            },
            registros_descartados=_integer(summary["fuera_universo"]),
            motivo_descarte=(
                "Registros fuera del universo TIC por tipo u ocupación de vivienda; "
                "no se clasifican como viviendas sin Internet."
            ),
        )

    def get_metadata(self) -> MetadataResponse:
        summary = self.summary()
        descriptions = [
            VariableMetadata(
                nombre=row["variable"],
                descripcion=row["descripcion"],
                codigos_o_rango=row["codigos_o_rango"],
                fuente=row["fuente"],
            )
            for row in self.table("diccionario_utilizado")
            if row["variable"] in summary["variables_seleccionadas"]
        ]
        summary_path = self.outputs_dir / "resumen_eda.json"
        generated = summary.get("generado_en") or datetime.fromtimestamp(
            summary_path.stat().st_mtime, tz=timezone.utc
        ).isoformat()
        return MetadataResponse(
            fuente=(
                "Microdatos de Vivienda y Diccionario de variables del Censo de "
                "Población y Vivienda 2024 incluidos en el proyecto"
            ),
            censo="Censo de Población y Vivienda 2024",
            departamento="La Paz",
            unidad_analisis="Registro de vivienda dentro del universo TIC aplicable",
            variables_principales=list(summary["variables_seleccionadas"]),
            descripcion_variables=descriptions,
            generado_en=generated,
            proceso=summary.get(
                "proceso",
                "Notebook EDA: lectura en bloques, filtro departamental y agregación censal",
            ),
        )

    def get_findings(self) -> HallazgosResponse:
        summary = self.summary()
        return HallazgosResponse(
            hallazgo_principal=summary["hallazgo_30_segundos"],
            hipotesis=summary["hipotesis"],
            conclusiones_principales=list(summary["conclusiones"]),
        )

    def get_logistic_regression(self) -> RegresionLogisticaResponse:
        """Expone artefactos ya calculados; nunca entrena desde una petición HTTP."""

        return RegresionLogisticaResponse.model_validate(self.logistic_summary())

    def get_decision_tree(self) -> ArbolClasificacionResponse:
        """Expone H3_2 desde artefactos; nunca entrena dentro de FastAPI."""

        return ArbolClasificacionResponse.model_validate(self.decision_tree_summary())


@lru_cache(maxsize=1)
def get_data_service() -> DataService:
    return DataService(settings.outputs_dir)
