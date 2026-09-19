import json
from pathlib import Path

import httpx
import pytest

from app.main import app


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def get(path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path)


@pytest.mark.anyio
async def test_health() -> None:
    response = await get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_resumen_uses_processed_results() -> None:
    raw = json.loads(
        (Path(__file__).resolve().parents[2] / "outputs" / "resumen_eda.json").read_text(
            encoding="utf-8"
        )
    )
    response = await get("/api/resumen")
    assert response.status_code == 200
    body = response.json()
    comparison = {row["tipo"]: row for row in raw["comparacion"]}
    assert body["total_registros"] == raw["registros_lapaz"]
    assert body["registros_validos"] == raw["universo_tic"]
    assert body["internet_fijo"]["cantidad"] == comparison["Fijo"]["Sí"]
    assert body["internet_movil"]["cantidad"] == comparison["Móvil"]["Sí"]


@pytest.mark.anyio
async def test_conectividad_area_partitions_universe() -> None:
    raw = json.loads(
        (Path(__file__).resolve().parents[2] / "outputs" / "resumen_eda.json").read_text(
            encoding="utf-8"
        )
    )
    response = await get("/api/conectividad/area")
    assert response.status_code == 200
    body = response.json()
    assert [item["area"] for item in body["items"]] == ["Urbana", "Rural"]
    assert sum(item["total"] for item in body["items"]) == raw["universo_tic"]
    assert body["brecha_urbano_rural_pp"] > 0
