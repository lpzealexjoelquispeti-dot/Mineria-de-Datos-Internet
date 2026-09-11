from typing import Literal

from fastapi import APIRouter, Depends

from app.schemas.responses import DistribucionResponse, OutliersResponse
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

