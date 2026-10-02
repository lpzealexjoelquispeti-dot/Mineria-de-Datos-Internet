"""API H3_3: solo artefactos guardados; compatible con endpoints anteriores."""
import json
from pathlib import Path

import httpx
import pytest
from app.main import app
from app.services.data_service import DataService, get_data_service

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def get(path):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        return await client.get(path)


@pytest.mark.anyio
async def test_arbol_regresion_saved_metrics_and_municipal_predictions():
    raw = json.loads((ROOT / "outputs/modelado/metricas_arbol_regresion.json").read_text())
    response = await get("/api/mineria/arbol-regresion")
    assert response.status_code == 200
    body = response.json()
    assert body["seleccion_features"] == raw["seleccion_features"]
    assert body["comparacion_features"] == raw["comparacion_features"]
    assert body["gridsearch"]["configuracion_cv"] == {
        "tipo": "KFold", "n_splits": 5, "shuffle": True, "random_state": 777}
    assert body["seleccion_features"]["variables"] == [v["variable"] for v in body["variables_utilizadas"]]
    assert body["metricas"] == raw["metricas"]
    assert body["real_vs_predicho"] == raw["real_vs_predicho"]
    assert body["unidad_analisis"] == "Municipio"
    assert body["target"]["indicador_eda"] == "pct_algun"
    assert body["registros"]["validacion_eda"]["coincide"]
    assert body["particion"]["train"] + body["particion"]["test"] == body["registros"]["municipios"]
    assert len(body["real_vs_predicho"]) == body["particion"]["test"]
    assert set(body["metricas"]["test"]) == {"mae", "mse", "rmse", "r2"}
    assert body["suma_importancias"] == pytest.approx(1)
    predictors = {item["variable"] for item in body["variables_utilizadas"]}
    assert predictors.isdisjoint({"v19e_f", "pct_algun", "municipio", "municipio_codigo"})


@pytest.mark.anyio
async def test_missing_regression_artifacts_503(tmp_path):
    app.dependency_overrides[get_data_service] = lambda: DataService(tmp_path)
    try:
        response = await get("/api/mineria/arbol-regresion")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503
    assert "entrenar_arbol_regresion.py" in response.json()["detail"]


@pytest.mark.anyio
async def test_endpoint_needs_only_json_and_keeps_negative_r2(tmp_path):
    raw = json.loads((ROOT / "outputs/modelado/metricas_arbol_regresion.json").read_text())
    raw["metricas"]["test"]["r2"] = -0.75
    folder = tmp_path / "modelado"
    folder.mkdir()
    (folder / "metricas_arbol_regresion.json").write_text(json.dumps(raw))
    app.dependency_overrides[get_data_service] = lambda: DataService(tmp_path)
    try:
        response = await get("/api/mineria/arbol-regresion")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["metricas"]["test"]["r2"] == -.75


@pytest.mark.anyio
async def test_three_mining_endpoints_remain_available():
    for path in ["regresion-logistica", "arbol-clasificacion", "arbol-regresion"]:
        response = await get(f"/api/mineria/{path}")
        assert response.status_code == 200
