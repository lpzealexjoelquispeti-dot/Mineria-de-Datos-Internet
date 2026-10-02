import csv
import json
from pathlib import Path

import httpx
import pytest

from app.main import app
from app.services.data_service import DataService, get_data_service


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


@pytest.mark.anyio
async def test_regresion_logistica_uses_saved_results_without_leakage() -> None:
    raw = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "outputs"
            / "modelado"
            / "metricas_regresion_logistica.json"
        ).read_text(encoding="utf-8")
    )
    response = await get("/api/mineria/regresion-logistica")
    assert response.status_code == 200
    body = response.json()
    assert body["target"]["nombre"] == "TIENE_ACCESO_INTERNET"
    assert body["target"]["variable_fuente"] == "v19e_f"
    assert body["metricas"]["test"] == raw["metricas"]["test"]
    assert body["particion"]["train"] + body["particion"]["test"] == body["registros"][
        "registros_target_valido"
    ]
    predictors = {item["variable"] for item in body["variables_utilizadas"]}
    assert predictors.isdisjoint({"v19e_inetfijo", "v19f_inetmovil", "v19e_f"})
    assert body["matriz_confusion_test"]["tn"] + body["matriz_confusion_test"][
        "fp"
    ] + body["matriz_confusion_test"]["fn"] + body["matriz_confusion_test"][
        "tp"
    ] == body["particion"]["test"]


@pytest.mark.anyio
async def test_arbol_clasificacion_uses_saved_real_results() -> None:
    root = Path(__file__).resolve().parents[2]
    raw = json.loads(
        (root / "outputs" / "modelado" / "metricas_arbol_clasificacion.json").read_text(
            encoding="utf-8"
        )
    )
    response = await get("/api/mineria/arbol-clasificacion")
    assert response.status_code == 200
    body = response.json()
    assert body["target"]["nombre"] == "TIENE_ACCESO_INTERNET"
    assert body["metricas"]["test"] == raw["metricas"]["test"]
    assert body["matriz_confusion_test"]["matriz"] == raw["matriz_confusion_test"]["matriz"]
    assert min(body["gridsearch"]["valores_evaluados"]["min_samples_split"]) >= 2
    assert abs(body["importancia_variables"]["suma_agregada"] - 1) < 1e-8
    assert all(body["verificacion_comparabilidad"].values())

    with (root / "outputs" / "modelado" / "comparacion_modelos_clasificacion.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        comparison_csv = list(csv.DictReader(file))
    assert [row["modelo"] for row in body["comparacion_modelos"]] == [
        row["modelo"] for row in comparison_csv
    ]
    for api_row, csv_row in zip(body["comparacion_modelos"], comparison_csv, strict=True):
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert api_row[metric] == pytest.approx(float(csv_row[metric]))


@pytest.mark.anyio
async def test_arbol_clasificacion_missing_artifact_is_handled(tmp_path: Path) -> None:
    app.dependency_overrides[get_data_service] = lambda: DataService(tmp_path)
    try:
        response = await get("/api/mineria/arbol-clasificacion")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503
    assert "entrenar_arbol_clasificacion.py" in response.json()["detail"]
