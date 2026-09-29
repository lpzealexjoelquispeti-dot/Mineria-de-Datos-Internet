"""Entrena y guarda la primera iteración de regresión logística."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.logistic_regression import ejecutar_entrenamiento, resultado_serializable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena la regresión logística de acceso a Internet en La Paz."
    )
    parser.add_argument(
        "--forzar-fuente",
        action="store_true",
        help=(
            "Ignora el subconjunto departamental derivado y relee el CSV original "
            "de Vivienda en bloques."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = ejecutar_entrenamiento(
        PROJECT_ROOT,
        prefer_cache=not args.forzar_fuente,
        save_outputs=True,
    )
    serializable = resultado_serializable(results)
    summary = {
        "target": serializable["target"],
        "registros": serializable["registros"],
        "particion": serializable["particion"],
        "metricas": serializable["metricas"],
        "matriz_confusion_test": serializable["matriz_confusion_test"],
        "baseline": serializable["baseline"],
        "interpretacion": serializable["interpretacion_primera_iteracion"],
        "salida": "outputs/modelado/metricas_regresion_logistica.json",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
