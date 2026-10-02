"""Verifica fuentes locales frente al EDA/manifiesto previo, sin descargar datos."""
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.census_sources import verificar_fuentes_censo

if __name__ == "__main__":
    print(json.dumps(verificar_fuentes_censo(PROJECT_ROOT), ensure_ascii=False, indent=2))
