"""Modelos de respuesta documentados por FastAPI."""

from __future__ import annotations

from typing import Any, Literal

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


class RegresionLogisticaResponse(BaseModel):
    objetivo: str
    target: dict[str, Any]
    variables_utilizadas: list[dict[str, str]]
    registros: dict[str, Any]
    distribucion_clases_total: dict[str, Any]
    particion: dict[str, Any]
    metricas: dict[str, dict[str, float]]
    matriz_confusion_test: dict[str, Any]
    baseline: dict[str, Any]
    umbral_principal: float
    roc: dict[str, Any]
    coeficientes_principales: list[dict[str, Any]]
    statsmodels: dict[str, Any]
    interpretacion_primera_iteracion: str


class ArbolClasificacionResponse(BaseModel):
    objetivo: str
    target: dict[str, Any]
    variables_utilizadas: list[dict[str, Any]]
    registros: dict[str, Any]
    distribucion_clases_total: dict[str, Any]
    particion: dict[str, Any]
    gridsearch: dict[str, Any]
    umbral_estandar: float
    metricas: dict[str, dict[str, float]]
    matriz_confusion_test: dict[str, Any]
    classification_report_test: dict[str, Any]
    roc: dict[str, Any]
    seleccion_umbral_roc: dict[str, Any]
    importancia_variables: dict[str, Any]
    estructura_arbol: dict[str, Any]
    comparacion_modelos: list[dict[str, Any]]
    verificacion_comparabilidad: dict[str, bool]
    interpretacion: str
    artefactos: dict[str, str]


class RegressionMetrics(BaseModel):
    mse: float = Field(ge=0)
    rmse: float = Field(ge=0)
    mae: float = Field(ge=0)
    r2: float  # Puede ser negativo.


class MunicipalPrediction(BaseModel):
    municipio_codigo: str
    municipio: str
    valor_real: float = Field(ge=0, le=100)
    valor_predicho: float
    error: float
    error_absoluto: float = Field(ge=0)


class RegressionImportance(BaseModel):
    variable: str
    importancia: float = Field(ge=0, le=1)


class RegressionMetricsDifference(BaseModel):
    mse: float
    rmse: float
    mae: float
    r2: float


class RegressionFeatureSelection(BaseModel):
    variante: Literal["A", "B"]
    variables: list[str]
    criterio: str
    diferencia_mse_b_menos_a: float
    empate_equivalente: bool
    tolerancia_absoluta_pp2: float = Field(ge=0)
    tolerancia_relativa: float = Field(ge=0)
    tolerancia_aplicada_pp2: float = Field(ge=0)
    motivo: str


class RegressionFeatureComparison(BaseModel):
    variante: Literal["A", "B"]
    variables: list[str]
    best_mse_cv: float = Field(ge=0)
    std_mse_cv: float = Field(ge=0)
    best_params: dict[str, int]
    seleccionada: bool


class ArbolRegresionResponse(BaseModel):
    generado_en: str
    objetivo: str
    unidad_analisis: Literal["Municipio"]
    referencia_academica: str
    target: dict[str, str]
    variables_utilizadas: list[dict[str, str]]
    seleccion_features: RegressionFeatureSelection
    comparacion_features: list[RegressionFeatureComparison]
    fuentes_censo_sha256: dict[str, Any]
    registros: dict[str, Any]
    particion: dict[str, Any]
    modelo_base: dict[str, Any]
    gridsearch: dict[str, Any]
    metricas: dict[str, RegressionMetrics]
    unidades_metricas: dict[str, str]
    baseline: dict[str, Any]
    comparacion_baseline: list[dict[str, Any]]
    diferencia_arbol_menos_baseline: RegressionMetricsDifference
    importancia_variables: list[RegressionImportance]
    suma_importancias: float = Field(ge=0, le=1.00000001)
    estructura_arbol: dict[str, int]
    real_vs_predicho: list[MunicipalPrediction]
    residuos: dict[str, Any]
    municipios_mayor_error: list[MunicipalPrediction]
    interpretacion: str
    limitaciones: list[str]
    artefactos: dict[str, str]
