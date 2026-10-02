"""Descarga el paquete CSV oficial del INE y extrae las fuentes de Vivienda.

El ZIP y los originales quedan en Base de datos CSV/, excluidos de Git.
No se extrae Persona, que no interviene en el análisis de conectividad.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PAGE = "https://cpv2024.ine.gob.bo/index.php/principal/descargas/"
URL = "https://nimbus.ine.gob.bo/index.php/s/qEmK9gnkGCZ3K7D/download"
ZIP_SHA256 = "9c41c13e655b99bbe4944801953b70678317e3ae9599f35d791e1262ab372160"
FILES = {
    "Vivienda_CPV-2024.csv": "f3cef44bcf103978734e86adf2c8c147a89adcb7146c78f78df7c409107a7aee",
    "Diccionario de variables CPV 2024.xlsx": "20e1f074145b8277ad18a71bc180a3e2f89f0b38fc9760bf88be0bc766e2db0e",
    "Cuestionario censal 2024.pdf": "e304a6d02a0a7c1d63b73f3fab4159e6526e77e545d8545932c5f2002f117059",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    folder = ROOT / "Base de datos CSV"
    folder.mkdir(exist_ok=True)
    archive = folder / "Base de datos CSV.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".zip.part")
        request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        print(f"Fuente oficial: {SOURCE_PAGE}", flush=True)
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as target:
            copied = 0
            next_report = 50 * 1024 * 1024
            while block := response.read(4 * 1024 * 1024):
                target.write(block)
                copied += len(block)
                if copied >= next_report:
                    print(f"Descargados: {copied / 1024**2:.0f} MiB", flush=True)
                    next_report += 50 * 1024 * 1024
        temporary.replace(archive)
    if sha256(archive) != ZIP_SHA256:
        raise ValueError("El ZIP no coincide con la huella publicada por el INE. Revisar la fuente.")
    with zipfile.ZipFile(archive) as package:
        by_name = {Path(item.filename).name: item for item in package.infolist() if not item.is_dir()}
        for name, expected in FILES.items():
            target = folder / name
            if target.exists():
                if sha256(target) != expected:
                    raise ValueError(f"El archivo local {name} difiere de las fuentes del grupo.")
                print(f"Fuente local verificada: {name}", flush=True)
                continue
            if name not in by_name:
                raise FileNotFoundError(f"No se encuentra {name} en el paquete oficial.")
            temporary = target.with_suffix(target.suffix + ".part")
            with package.open(by_name[name]) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            if sha256(temporary) != expected:
                raise ValueError(f"La fuente descargada {name} difiere de la utilizada por el grupo.")
            temporary.replace(target)
            print(f"Extraído y verificado: {name} ({target.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    main()
