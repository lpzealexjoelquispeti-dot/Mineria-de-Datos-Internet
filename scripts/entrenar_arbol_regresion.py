"""Regenera exclusivamente los artefactos de H3_3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.decision_tree_regression import entrenar, guardar_resultados, preparar_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="H3_3: árbol de regresión de acceso municipal a Internet.")
    parser.add_argument("--forzar-fuente", action="store_true", help="Releer Vivienda en bloques e ignorar la copia departamental.")
    args = parser.parse_args()
    print("Verificando fuentes y preparando municipios...", flush=True)
    dataset = preparar_dataset(ROOT, prefer_cache=not args.forzar_fuente)
    print(f"{len(dataset.table)} municipios; {dataset.audit['universo_tic']:,} viviendas aplicables.", flush=True)
    result = entrenar(dataset)
    guardar_resultados(dataset, result, ROOT)
    print(json.dumps(result["resumen"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
