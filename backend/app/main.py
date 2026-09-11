"""Aplicación FastAPI para explorar resultados procesados del Censo 2024."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import calidad, conectividad, geografico, resumen
from app.core.config import settings


app = FastAPI(
    title=settings.app_name,
    description=(
        "API de solo lectura para los resultados del análisis de Internet fijo y móvil "
        "en el departamento de La Paz."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["Accept", "Content-Type"],
)


@app.get(f"{settings.api_prefix}/health", tags=["Estado"])
def health() -> dict[str, str]:
    return {"status": "ok"}


for route in (resumen.router, conectividad.router, geografico.router, calidad.router):
    app.include_router(route, prefix=settings.api_prefix)

