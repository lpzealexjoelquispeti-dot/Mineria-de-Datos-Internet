from fastapi import APIRouter, Depends

from app.schemas.responses import HallazgosResponse, MetadataResponse, ResumenResponse
from app.services.data_service import DataService, get_data_service


router = APIRouter(tags=["Resumen"])


@router.get("/resumen", response_model=ResumenResponse)
def get_summary(service: DataService = Depends(get_data_service)) -> ResumenResponse:
    return service.get_summary()


@router.get("/metadata", response_model=MetadataResponse)
def get_metadata(service: DataService = Depends(get_data_service)) -> MetadataResponse:
    return service.get_metadata()


@router.get("/hallazgos", response_model=HallazgosResponse)
def get_findings(service: DataService = Depends(get_data_service)) -> HallazgosResponse:
    return service.get_findings()

