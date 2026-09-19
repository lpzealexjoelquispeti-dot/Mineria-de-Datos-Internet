"""Regenera los artefactos pequeños que consume la aplicación web.

Uso desde la raíz:
    python -m src.pipeline

Para desarrollo, si ya existe el subconjunto departamental comprimido:
    python -m src.pipeline --from-cache
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.mining import (
    build_comparison,
    cramers_v_2x2,
    detect_iqr_outliers,
    summarize_groups,
)


VARIABLES = [
    "idep",
    "iprov",
    "imun",
    "i00",
    "urbrur",
    "v01_tipoviv",
    "v02_condocup",
    "v09_energia",
    "v13_habitac",
    "v14_dormit",
    "v19c_compu",
    "v19d_celular",
    "v19e_inetfijo",
    "v19f_inetmovil",
    "v19e_f",
    "tot_pers",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_csv(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as file:
        sample = file.read(65_536)
    dialect = csv.Sniffer().sniff(sample, delimiters=";,\t|")
    header = next(csv.reader(sample.splitlines(), dialect))
    return {"separator": dialect.delimiter, "header": header}


def read_dictionary(path: Path) -> dict[str, dict[str, Any]]:
    sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object, engine="openpyxl")
    dictionary: dict[str, dict[str, Any]] = {}
    for sheet_name, sheet in sheets.items():
        current: str | None = None
        description: Any = None
        for excel_row, row in enumerate(sheet.itertuples(index=False, name=None), start=1):
            key, value = row[1], row[2]
            if key == "Etiqueta":
                description, current = value, None
            elif key == "Nombre":
                current = str(value).lower()
                dictionary[current] = {
                    "descripcion": description,
                    "categorias": {},
                    "hoja": sheet_name,
                    "fila_nombre": excel_row,
                }
            elif current and key == "Tipo":
                dictionary[current]["tipo"] = value
                dictionary[current]["rango"] = str(row[4])
            elif current and pd.notna(key) and re.fullmatch(r"\d+(?:\.0)?", str(key).strip()):
                dictionary[current]["categorias"][str(int(float(key)))] = str(value)
    return dictionary


def load_lapaz(
    source: Path,
    cached: Path,
    department_code: str,
    from_cache: bool,
) -> tuple[pd.DataFrame, int, int]:
    source_info = inspect_csv(source)
    if not set(VARIABLES).issubset(source_info["header"]):
        raise ValueError("Vivienda_CPV-2024.csv no contiene todas las variables requeridas")

    if from_cache:
        if not cached.is_file():
            raise FileNotFoundError(f"No existe el subconjunto local: {cached}")
        data = pd.read_csv(
            cached,
            sep=";",
            encoding="utf-8",
            dtype="string",
            keep_default_na=False,
            low_memory=False,
        )
        # El total nacional no se infiere de la copia departamental.
        structure_path = cached.parents[1] / "tablas" / "estructura.csv"
        structure = pd.read_csv(structure_path)
        national = int(
            structure.loc[structure["medida"].eq("Registros nacionales"), "valor"].iloc[0]
        )
        original_columns = len(source_info["header"])
        return data, national, original_columns

    parts: list[pd.DataFrame] = []
    national = 0
    for chunk in pd.read_csv(
        source,
        sep=source_info["separator"],
        encoding="utf-8-sig",
        encoding_errors="strict",
        usecols=VARIABLES,
        dtype="string",
        keep_default_na=False,
        low_memory=False,
        chunksize=200_000,
    ):
        national += len(chunk)
        parts.append(chunk.loc[chunk["idep"].eq(department_code), VARIABLES].copy())
    data = pd.concat(parts, ignore_index=True)
    cached.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(
        cached,
        sep=";",
        index=False,
        encoding="utf-8",
        compression={"method": "gzip", "compresslevel": 1, "mtime": 0},
    )
    return data, national, len(source_info["header"])


def dictionary_table(
    dictionary: dict[str, dict[str, Any]], variables: list[str]
) -> pd.DataFrame:
    geography_notes = {
        "idep": "Código departamental: correspondencia con dep_res_cod; no definido directamente",
        "iprov": "Componente provincial: idep sin cero + iprov se contrasta con prov_res_cod",
        "imun": "Componente municipal: código provincial + imun se contrasta con mun_res_cod",
        "i00": "Identificador candidato de registro; sin definición directa en el diccionario",
    }
    rows = []
    for variable in variables:
        item = dictionary.get(variable)
        rows.append(
            {
                "variable": variable,
                "descripcion": item["descripcion"] if item else geography_notes[variable],
                "codigos_o_rango": (
                    "; ".join(f"{key}: {label}" for key, label in item["categorias"].items())
                    if item and item["categorias"]
                    else item.get("rango", "") if item else "Ver nota documental"
                ),
                "fuente": (
                    f"{item['hoja']}, fila {item['fila_nombre']}"
                    if item
                    else "Correspondencia auditada / limitación"
                ),
            }
        )
    return pd.DataFrame(rows)


def quality_missing(
    data: pd.DataFrame, dictionary: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    rows = []
    for variable in VARIABLES:
        series = data[variable]
        categories = dictionary.get(variable, {}).get("categorias", {})
        unspecified = [
            key for key, label in categories.items() if "sin especificar" in label.casefold()
        ]
        not_applicable = [
            key for key, label in categories.items() if "no aplica" in label.casefold()
        ]
        measures = {
            "nulos_pandas": int(series.isna().sum()),
            "vacios_csv": int(series.eq("").sum()),
            "solo_espacios": int((series.ne("") & series.str.strip().eq("")).sum()),
            "marcadores_NaN_NULL_NA": int(
                series.str.upper().isin(["NAN", "NULL", "NA", "N/A"]).sum()
            ),
            "sin_especificar": int(series.isin(unspecified).sum()),
            "no_aplica_codificado": int(series.isin(not_applicable).sum()),
        }
        rows.append(
            {
                "variable": variable,
                **measures,
                **{f"{key}_pct": value / len(data) * 100 for key, value in measures.items()},
            }
        )
    return pd.DataFrame(rows)


def catalog_audit(
    data: pd.DataFrame, dictionary: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    rows = []
    for variable in VARIABLES:
        item = dictionary.get(variable)
        if not item:
            continue
        series = data[variable]
        if item["categorias"]:
            invalid = series.ne("") & ~series.isin(item["categorias"])
        else:
            limits = [int(value) for value in re.findall(r"\d+", item["rango"])]
            numeric = pd.to_numeric(series, errors="coerce")
            invalid = series.ne("") & (
                ~numeric.between(*limits) | numeric.isna() | numeric.mod(1).ne(0)
            )
        rows.append(
            {
                "variable": variable,
                "fuera_catalogo": int(invalid.sum()),
                "valores": str(series[invalid].value_counts().to_dict()),
            }
        )
    return pd.DataFrame(rows)


def records(table: pd.DataFrame) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in table.to_dict(orient="records"):
        item: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, np.integer):
                item[key] = int(value)
            elif isinstance(value, np.floating):
                item[key] = float(value)
            elif pd.isna(value):
                item[key] = None
            else:
                item[key] = value
        normalized.append(item)
    return normalized


def format_integer(value: Any) -> str:
    return f"{int(value):,}".replace(",", ".")


def format_decimal(value: Any) -> str:
    return f"{float(value):.2f}".replace(".", ",")


def build_outputs(root: Path, from_cache: bool = False) -> None:
    source_dir = root / "Base de datos CSV"
    output_dir = root / "outputs"
    tables_dir = output_dir / "tablas"
    tables_dir.mkdir(parents=True, exist_ok=True)

    housing = source_dir / "Vivienda_CPV-2024.csv"
    dictionary_path = source_dir / "Diccionario de variables CPV 2024.xlsx"
    questionnaire = source_dir / "Cuestionario censal 2024.pdf"
    cache = output_dir / "datos" / "vivienda_lapaz_seleccion.csv.gz"

    for required in (housing, dictionary_path, questionnaire):
        if not required.is_file():
            raise FileNotFoundError(f"Falta la fuente requerida: {required}")

    dictionary = read_dictionary(dictionary_path)
    department_codes = [
        code for code, label in dictionary["dep_res_cod"]["categorias"].items() if label == "La Paz"
    ]
    if len(department_codes) != 1:
        raise ValueError("No se pudo determinar de forma unívoca el código de La Paz")
    department_code = department_codes[0].zfill(2)
    data, national_records, original_columns = load_lapaz(
        housing, cache, department_code, from_cache
    )

    applicable = data["v01_tipoviv"].isin(["1", "2", "3", "4", "5", "6"]) & data[
        "v02_condocup"
    ].isin(["0", "1"])
    universe = data.loc[applicable].copy()
    universe["area_label"] = universe["urbrur"].map({"1": "Urbana", "2": "Rural"})

    province_codes = data["idep"].str.lstrip("0") + data["iprov"]
    municipality_codes = province_codes + data["imun"]
    province_catalog = dictionary["prov_res_cod"]["categorias"]
    municipality_catalog = dictionary["mun_res_cod"]["categorias"]
    universe["provincia_codigo"] = province_codes.loc[applicable].to_numpy()
    universe["municipio_codigo"] = municipality_codes.loc[applicable].to_numpy()
    universe["provincia"] = universe["provincia_codigo"].map(province_catalog)
    universe["municipio"] = universe["municipio_codigo"].map(municipality_catalog)

    if universe[["provincia", "municipio", "area_label"]].isna().any().any():
        raise ValueError("Existen códigos geográficos sin correspondencia oficial")

    comparison = build_comparison(universe)
    areas = summarize_groups(universe, ["area_label"]).reindex(["Urbana", "Rural"])
    provinces = summarize_groups(universe, ["provincia_codigo", "provincia"]).sort_values(
        "pct_algun", ascending=False
    )
    municipalities = summarize_groups(
        universe, ["municipio_codigo", "municipio"]
    ).sort_values("pct_algun", ascending=False)
    municipalities_area = summarize_groups(
        universe, ["area_label", "municipio_codigo", "municipio"]
    ).sort_values(["area_label", "pct_algun"], ascending=[True, False])

    outliers, iqr = detect_iqr_outliers(municipalities)
    rates = municipalities["pct_algun"].astype(float)
    cross_area = pd.crosstab(universe["area_label"], universe["v19e_f"]).reindex(
        index=["Urbana", "Rural"], columns=["1", "2", "9"], fill_value=0
    )
    cross_area.columns = ["Sí", "No", "Sin especificar"]
    cramer = cramers_v_2x2(cross_area)

    missing = quality_missing(data, dictionary)
    out_of_catalog = catalog_audit(data, dictionary)
    duplicate_counts = {
        "i00_vacio": int(data["i00"].eq("").sum()),
        "i00_repeticiones_adicionales": int(data["i00"].duplicated().sum()),
        "i00_filas_involucradas": int(data["i00"].duplicated(keep=False).sum()),
        "clave_compuesta_repeticiones": int(
            data.duplicated(["idep", "iprov", "imun", "i00"]).sum()
        ),
        "filas_iguales_16_columnas": int(data.duplicated().sum()),
    }

    fixed, mobile, official = (data[column] for column in ("v19e_inetfijo", "v19f_inetmovil", "v19e_f"))
    conservative = pd.Series("Fuera del universo", index=data.index, dtype="string")
    conservative.loc[applicable] = "9"
    conservative.loc[applicable & (fixed.eq("1") | mobile.eq("1"))] = "1"
    conservative.loc[applicable & fixed.eq("2") & mobile.eq("2")] = "2"
    discrepancy = applicable & official.ne(conservative)
    inconsistencies = {
        "codigos_fuera_catalogo": int(out_of_catalog["fuera_catalogo"].sum()),
        "dormitorios_mayores_que_habitaciones": int(
            (
                pd.to_numeric(data["v14_dormit"], errors="coerce")
                > pd.to_numeric(data["v13_habitac"], errors="coerce")
            ).sum()
        ),
        "internet_informado_fuera_universo": int((~applicable & official.ne("")).sum()),
        "particulares_presentes_sin_personas": int(
            (applicable & data["tot_pers"].eq("0")).sum()
        ),
        "combinada_discrepa_regla_conservadora": int(discrepancy.sum()),
        "combinada_contradiccion_si_no_determinada": int(
            (
                applicable
                & (
                    (conservative.eq("1") & official.eq("2"))
                    | (conservative.eq("2") & official.eq("1"))
                )
            ).sum()
        ),
    }

    fixed_row = comparison.loc["Fijo"]
    mobile_row = comparison.loc["Móvil"]
    combined_row = comparison.loc["Algún internet"]
    urban = areas.loc["Urbana"]
    rural = areas.loc["Rural"]
    fixed_mobile_records = int(mobile_row["Sí"] - fixed_row["Sí"])
    fixed_mobile_pp = float(mobile_row["% Sí"] - fixed_row["% Sí"])
    area_gap = float(urban["pct_algun"] - rural["pct_algun"])

    hypothesis = (
        "Creemos que el área de residencia influye en el acceso a Internet fijo o móvil "
        f"porque observamos {format_decimal(urban['pct_algun'])} % de acceso declarado en viviendas urbanas "
        f"frente a {format_decimal(rural['pct_algun'])} % en rurales, una diferencia de {format_decimal(area_gap)} "
        "puntos porcentuales."
    )
    finding = (
        f"En el Censo 2024 analizamos {format_integer(len(universe))} viviendas particulares con personas "
        f"presentes en La Paz. El {format_decimal(mobile_row['% Sí'])} % declara Internet móvil y el "
        f"{format_decimal(fixed_row['% Sí'])} % Internet fijo. La diferencia territorial también es clara: "
        f"algún Internet alcanza al {format_decimal(urban['pct_algun'])} % urbano frente al "
        f"{format_decimal(rural['pct_algun'])} % rural. Esto sugiere una desigualdad de acceso asociada al "
        "área de residencia; no prueba causalidad. Los porcentajes incluyen las respuestas "
        "sin especificar en el denominador."
    )
    outlier_text = "; ".join(
        f"{index[1]} ({format_decimal(row['pct_algun'])} %)" for index, row in outliers.iterrows()
    )
    conclusions = [
        f"De {format_integer(len(data))} registros de La Paz, {format_integer(len(universe))} pertenecen al universo TIC; "
        f"{format_integer((~applicable).sum())} tienen saltos fuera de ese universo y no se clasifican como desconectados.",
        f"Internet móvil ({format_decimal(mobile_row['% Sí'])} %) supera al fijo ({format_decimal(fixed_row['% Sí'])} %) "
        f"en {format_decimal(fixed_mobile_pp)} pp y {format_integer(fixed_mobile_records)} viviendas.",
        f"El indicador combinado oficial registra {format_integer(combined_row['Sí'])} viviendas con acceso "
        f"({format_decimal(combined_row['% Sí'])} %); {format_integer(combined_row['Sin especificar'])} quedan sin "
        f"especificar ({format_decimal(combined_row['% Sin especificar'])} %).",
        f"El acceso urbano ({format_decimal(urban['pct_algun'])} %) excede al rural ({format_decimal(rural['pct_algun'])} %) "
        f"en {format_decimal(area_gap)} pp; la asociación tiene V de Cramér de {format_decimal(cramer)} entre respuestas determinadas.",
        f"Entre {len(municipalities)} municipios, la mediana de acceso es {format_decimal(rates.median())} %. "
        f"{len(outliers)} superan el límite IQR de {format_decimal(iqr.limite_superior)} %: {outlier_text}.",
    ]

    generated_at = datetime.now(timezone.utc).isoformat()
    sources = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in (housing, dictionary_path, questionnaire)
    }
    summary = {
        "fuentes": sources,
        "generado_en": generated_at,
        "proceso": "src.pipeline: lectura en bloques, filtro departamental y agregación censal",
        "registros_nacionales": national_records,
        "columnas_originales": original_columns,
        "variables_seleccionadas": VARIABLES,
        "codigo_lapaz": department_code,
        "registros_lapaz": len(data),
        "universo_tic": len(universe),
        "fuera_universo": int((~applicable).sum()),
        "comparacion": records(comparison.reset_index()),
        "duplicados": duplicate_counts,
        "inconsistencias": inconsistencies,
        "diferencia_movil_fijo_registros": fixed_mobile_records,
        "diferencia_movil_fijo_pp": fixed_mobile_pp,
        "diferencia_relativa_pct": float(fixed_mobile_records / fixed_row["Sí"] * 100),
        "brecha_urbano_rural_pp": area_gap,
        "areas": records(areas.reset_index()),
        "v_cramer": cramer,
        "outliers": records(outliers.reset_index()),
        "iqr": {
            "Q1": iqr.q1,
            "Q3": iqr.q3,
            "IQR": iqr.iqr,
            "Límite inferior": iqr.limite_inferior,
            "Límite superior": iqr.limite_superior,
        },
        "mediana_municipal": float(rates.median()),
        "asimetria_municipal": float(rates.skew()),
        "hipotesis": hypothesis,
        "hallazgo_30_segundos": finding,
        "conclusiones": conclusions,
    }

    geography = (
        universe[["provincia_codigo", "municipio_codigo", "provincia", "municipio"]]
        .drop_duplicates()
        .sort_values("municipio_codigo")
    )
    structure = pd.DataFrame(
        {
            "medida": [
                "Registros nacionales",
                "Registros de La Paz",
                "Variables del CSV original",
                "Variables originales seleccionadas",
            ],
            "valor": [national_records, len(data), original_columns, len(VARIABLES)],
        }
    )
    universe_audit = pd.DataFrame(
        [
            {
                "variable": variable,
                "vacios_dentro_universo": int((applicable & data[variable].eq("")).sum()),
                "vacios_fuera_universo": int((~applicable & data[variable].eq("")).sum()),
                "respuesta_fuera_universo": int((~applicable & data[variable].ne("")).sum()),
                "sin_especificar_dentro": int((applicable & data[variable].eq("9")).sum()),
            }
            for variable in ("v19e_inetfijo", "v19f_inetmovil", "v19e_f")
        ]
    )

    exports = {
        "diccionario_utilizado": dictionary_table(dictionary, VARIABLES),
        "estructura": structure,
        "faltantes": missing,
        "universo": universe_audit,
        "fuera_catalogo": out_of_catalog,
        "correspondencia_geografica": geography,
        "comparacion": comparison.reset_index(),
        "areas": areas.reset_index(),
        "provincias": provinces.reset_index(),
        "municipios": municipalities.reset_index(),
        "municipios_area": municipalities_area.reset_index(),
        "top10": municipalities.head(10).reset_index(),
        "bottom10": municipalities.tail(10).sort_values("pct_algun").reset_index(),
        "outliers": outliers.reset_index(),
        "limites_iqr": pd.DataFrame(
            {
                "medida": ["Q1", "Q3", "IQR", "Límite inferior", "Límite superior"],
                "porcentaje_pp": [
                    iqr.q1,
                    iqr.q3,
                    iqr.iqr,
                    iqr.limite_inferior,
                    iqr.limite_superior,
                ],
            }
        ),
        "distribucion": rates.describe().rename_axis("medida").reset_index(name="porcentaje"),
        "relacion_area": cross_area.reset_index(),
    }
    for name, table in exports.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    (output_dir / "resumen_eda.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if int(areas["universo"].sum()) != len(universe):
        raise AssertionError("La suma por área no coincide con el universo")
    if int(provinces["universo"].sum()) != len(universe):
        raise AssertionError("La suma por provincia no coincide con el universo")
    if int(municipalities["universo"].sum()) != len(universe):
        raise AssertionError("La suma por municipio no coincide con el universo")
    if not np.allclose(
        municipalities[["pct_algun", "pct_sin_internet", "pct_sin_especificar"]].sum(axis=1),
        100,
    ):
        raise AssertionError("Los porcentajes municipales no suman 100")
    if int(out_of_catalog["fuera_catalogo"].sum()) != 0:
        raise AssertionError("Existen códigos fuera de catálogo")

    print(
        f"Resultados regenerados: {len(universe):,} viviendas del universo TIC, "
        f"{len(municipalities)} municipios y {len(provinces)} provincias."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Usa outputs/datos/vivienda_lapaz_seleccion.csv.gz para una regeneración rápida.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build_outputs(root, from_cache=args.from_cache)


if __name__ == "__main__":
    main()
