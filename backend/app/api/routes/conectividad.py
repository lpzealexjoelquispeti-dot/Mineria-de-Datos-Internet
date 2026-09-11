from typing import Literal

from fastapi import APIRouter, Depends, Query

from app.schemas.responses import AreaResponse, TerritoriosResponse
from app.services.data_service import DataService, get_data_service


router = APIRouter(prefix="/conectividad", tags=["Conectividad"])


@router.get("/area", response_model=AreaResponse)
def get_by_area(service: DataService = Depends(get_data_service)) -> AreaResponse:
    return service.get_areas()


@router.get("/municipios", response_model=TerritoriosResponse)
def get_municipalities(
    limit: int = Query(default=10, ge=1, le=100),
    orden: Literal["mayor", "menor"] = "mayor",
    metrica: Literal["internet", "fijo", "movil", "sin_internet"] = "internet",
    area: Literal["todos", "urbana", "rural"] = "todos",
    service: DataService = Depends(get_data_service),
) -> TerritoriosResponse:
    return service.get_territories("municipio", limit, orden, metrica, area)


@router.get("/provincias", response_model=TerritoriosResponse)
def get_provinces(
    limit: int = Query(default=20, ge=1, le=100),
    orden: Literal["mayor", "menor"] = "mayor",
    metrica: Literal["internet", "fijo", "movil", "sin_internet"] = "internet",
    service: DataService = Depends(get_data_service),
) -> TerritoriosResponse:
    return service.get_territories("provincia", limit, orden, metrica)

