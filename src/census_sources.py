"""Verificación local de fuentes censales: reutiliza SHA-256 de src.pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.pipeline import sha256

SOURCE_NAMES = ("Vivienda_CPV-2024.csv", "Diccionario de variables CPV 2024.xlsx")
MANIFEST_PATH = Path("outputs/modelado/fuentes_censo_sha256.json")


def verificar_archivos(files: dict[str, Path], previous: dict[str, Any] | None = None,
                       root: Path | None = None) -> dict[str, Any]:
    """Calcula hashes; si existe referencia, exige entrada, tamaño y hash iguales.

    Las rutas son informativas; un traslado del repositorio no cambia identidad.
    No sobrescribe un manifiesto ni acepta cambios silenciosamente.
    """
    result = {}
    for name, path in files.items():
        result[name] = {"ruta": str(path.relative_to(root)) if root else str(path),
                        "bytes": path.stat().st_size, "sha256": sha256(path)}
        if previous is not None:
            expected = previous.get(name)
            if not expected or any(result[name][key] != expected.get(key) for key in ("bytes", "sha256")):
                raise ValueError(f"Fuente censal distinta del manifiesto previo: {name}")
    return result


def verificar_fuentes_censo(project_root: Path, guardar: bool = True) -> dict[str, Any]:
    files = {name: project_root / "Base de datos CSV" / name for name in SOURCE_NAMES}
    current = verificar_archivos(files, root=project_root)
    # Ambas referencias son previas, reales y locales; no se inventan hashes.
    references = (project_root / MANIFEST_PATH, project_root / "outputs/resumen_eda.json")
    for reference in references:
        if reference.is_file():
            expected = json.loads(reference.read_text(encoding="utf-8"))
            if reference.name == "resumen_eda.json":
                expected = expected["fuentes"]
            for name in SOURCE_NAMES:
                if name not in expected or any(current[name][key] != expected[name].get(key)
                                               for key in ("bytes", "sha256")):
                    raise ValueError(f"Fuente censal distinta de {reference.name}: {name}")
    if guardar:
        manifest = project_root / MANIFEST_PATH
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return current
