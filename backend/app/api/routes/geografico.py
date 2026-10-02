from typing import Literal

from fastapi import APIRouter, Depends, HTTPException

from app.schemas.responses import (
    ArbolClasificacionResponse,
    ArbolRegresionResponse,
    DistribucionResponse,
    OutliersResponse,
    RegresionLogisticaResponse,
)
from app.services.data_service import DataService, get_data_service


router = APIRouter(prefix="/mineria", tags=["Minería de datos"])


@router.get("/outliers", response_model=OutliersResponse)
def get_outliers(service: DataService = Depends(get_data_service)) -> OutliersResponse:
    return service.get_outliers()


@router.get("/distribucion", response_model=DistribucionResponse)
def get_distribution(
    metrica: Literal["internet", "fijo", "movil"] = "internet",
    area: Literal["todos", "urbana", "rural"] = "todos",
    service: DataService = Depends(get_data_service),
) -> DistribucionResponse:
    return service.get_distribution(metrica, area)


@router.get("/regresion-logistica", response_model=RegresionLogisticaResponse)
def get_logistic_regression(
    service: DataService = Depends(get_data_service),
) -> RegresionLogisticaResponse:
    return service.get_logistic_regression()


@router.get("/arbol-clasificacion", response_model=ArbolClasificacionResponse)
def get_decision_tree(
    service: DataService = Depends(get_data_service),
) -> ArbolClasificacionResponse:
    try:
        return service.get_decision_tree()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.get("/arbol-regresion", response_model=ArbolRegresionResponse)
def get_regression_tree(
    service: DataService = Depends(get_data_service),
) -> ArbolRegresionResponse:
    try:
        return service.get_regression_tree()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
