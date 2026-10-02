"""Entrena y guarda H3_2: árbol de decisión para acceso a Internet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.decision_tree_classification import ejecutar_entrenamiento, resultado_serializable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena el árbol de decisión de acceso a Internet en La Paz."
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
    results = resultado_serializable(
        ejecutar_entrenamiento(
            PROJECT_ROOT,
            prefer_cache=not args.forzar_fuente,
            save_outputs=True,
        )
    )
    summary = {
        "target": results["target"],
        "registros": results["registros"],
        "distribucion_clases": results["distribucion_clases_total"],
        "particion": results["particion"],
        "mejores_hiperparametros": results["gridsearch"]["mejores_parametros"],
        "mejor_roc_auc_cv": results["gridsearch"]["mejor_roc_auc_cv"],
        "metricas_train": results["metricas"]["train"],
        "metricas_test": results["metricas"]["test"],
        "matriz_confusion_test": results["matriz_confusion_test"],
        "umbral_roc": results["seleccion_umbral_roc"],
        "estructura_arbol": results["estructura_arbol"],
        "artefactos": results["artefactos"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
