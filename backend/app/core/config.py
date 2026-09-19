"""Configuración simple mediante variables de entorno."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Settings:
    app_name: str = "Conectividad Digital en La Paz"
    api_prefix: str = "/api"
    outputs_dir: Path = Path(os.getenv("CENSO_OUTPUTS_DIR", PROJECT_ROOT / "outputs"))
    cors_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",")
        if origin.strip()
    )


settings = Settings()

