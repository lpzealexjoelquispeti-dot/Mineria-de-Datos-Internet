from fastapi import APIRouter, Depends

from app.schemas.responses import CalidadResponse
from app.services.data_service import DataService, get_data_service


router = APIRouter(tags=["Calidad"])


@router.get("/calidad", response_model=CalidadResponse)
def get_quality(service: DataService = Depends(get_data_service)) -> CalidadResponse:
    return service.get_quality()

