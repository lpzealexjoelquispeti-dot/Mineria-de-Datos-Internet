"""Modelos de respuesta documentados por FastAPI."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Indicador(BaseModel):
    cantidad: int = Field(ge=0)
    porcentaje: float = Field(ge=0, le=100)
    porcentaje_determinados: float | None = Field(default=None, ge=0, le=100)


class ResumenResponse(BaseModel):
    total_registros: int = Field(ge=0)
    registros_validos: int = Field(ge=0)
    registros_fuera_universo: int = Field(ge=0)
    internet_fijo: Indicador
    internet_movil: Indicador
    algun_internet: Indicador
    sin_internet: Indicador
    sin_especificar: Indicador


class AreaConectividad(BaseModel):
    area: Literal["Urbana", "Rural"]
    total: int = Field(ge=0)
    internet_fijo: Indicador
    internet_movil: Indicador
    algun_internet: Indicador
    sin_internet: Indicador
    sin_especificar: Indicador


class AreaResponse(BaseModel):
    items: list[AreaConectividad]
    brecha_urbano_rural_pp: float
    v_cramer: float = Field(ge=0, le=1)


class TerritorioConectividad(BaseModel):
    codigo: str
    nombre: str | None
    total: int = Field(ge=0)
    internet_fijo: Indicador
    internet_movil: Indicador
    algun_internet: Indicador
    sin_internet: Indicador
    sin_especificar: Indicador


class TerritoriosResponse(BaseModel):
    items: list[TerritorioConectividad]
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    orden: Literal["mayor", "menor"]
    metrica: Literal["internet", "fijo", "movil", "sin_internet"]
    area: Literal["todos", "urbana", "rural"] = "todos"


class Outlier(BaseModel):
    unidad_geografica: Literal["Municipio"] = "Municipio"
    codigo: str
    nombre: str
    metrica: str
    valor: float
    metodo: str
    q1: float
    q3: float
    iqr: float
    limite_inferior: float
    limite_superior: float


class OutliersResponse(BaseModel):
    descripcion: str
    items: list[Outlier]


class DistribucionItem(BaseModel):
    codigo: str
    nombre: str
    valor: float


class DistribucionResponse(BaseModel):
    metrica: Literal["internet", "fijo", "movil"]
    area: Literal["todos", "urbana", "rural"]
    items: list[DistribucionItem]
    mediana: float


class ValorFaltante(BaseModel):
    variable: str
    nulos_pandas: int
    vacios_csv: int
    solo_espacios: int
    marcadores_nan_null_na: int
    sin_especificar: int
    no_aplica_codificado: int


class CalidadResponse(BaseModel):
    valores_faltantes: list[ValorFaltante]
    duplicados: dict[str, int]
    inconsistencias: dict[str, int]
    codigos_sin_especificar: dict[str, int]
    registros_descartados: int
    motivo_descarte: str


class VariableMetadata(BaseModel):
    nombre: str
    descripcion: str
    codigos_o_rango: str
    fuente: str


class MetadataResponse(BaseModel):
    fuente: str
    censo: str
    departamento: str
    unidad_analisis: str
    variables_principales: list[str]
    descripcion_variables: list[VariableMetadata]
    generado_en: str
    proceso: str


class HallazgosResponse(BaseModel):
    hallazgo_principal: str
    hipotesis: str
    conclusiones_principales: list[str]

