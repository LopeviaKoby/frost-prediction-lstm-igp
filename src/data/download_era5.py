from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.config import DATA_RAW_DIR, OUTPUTS_LOGS_DIR, PROJECT_ROOT


LOGGER = logging.getLogger("era5_download")
ERA5_ROOT = DATA_RAW_DIR / "era5"
ERA5_MANTARO_ROOT = ERA5_ROOT / "mantaro_hourly"
ERA5_SOUTH_AMERICA_ROOT = ERA5_ROOT / "sudamerica_monthly"
ERA5_MANIFESTS_ROOT = ERA5_ROOT / "_manifests"
ERA5_STATE_PATH = ERA5_ROOT / "_download_state.json"
ERA5_LOG_PATH = OUTPUTS_LOGS_DIR / "era5_download.log"

MANTARO_BASIN_BBOX = [-10, -77, -14, -74]
SOUTH_AMERICA_BBOX = [15, -90, -60, -15]

MANTARO_CORE_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "total_cloud_cover",
    "skin_temperature",
    "surface_sensible_heat_flux",
    "surface_latent_heat_flux",
    "total_precipitation",
    "soil_temperature_level_1",
    "volumetric_soil_water_layer_4",
]

MANTARO_EXTENDED_VARIABLES = MANTARO_CORE_VARIABLES + [
    "mean_sea_level_pressure",
    "evaporation",
    "potential_evaporation",
    "total_sky_direct_solar_radiation_at_surface",
    "surface_net_solar_radiation_clear_sky",
    "runoff",
    "surface_runoff",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
    "leaf_area_index_low_vegetation",
    "leaf_area_index_high_vegetation",
]

SOUTH_AMERICA_MONTHLY_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "sea_surface_temperature",
    "surface_pressure",
    "evaporation",
    "potential_evaporation",
    "total_sky_direct_solar_radiation_at_surface",
    "surface_net_solar_radiation_clear_sky",
]


@dataclass(frozen=True)
class RequestSpec:
    dataset: str
    target_path: Path
    manifest_path: Path
    request: dict
    logical_id: str


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def build_client():
    env_path = PROJECT_ROOT / ".env"
    load_env_file(env_path)

    url = os.getenv("CDSAPI_URL")
    key = os.getenv("CDSAPI_KEY")

    if not url or not key:
        raise RuntimeError(
            "No se encontraron CDSAPI_URL y CDSAPI_KEY. Configura el archivo .env antes de descargar."
        )

    try:
        import cdsapi
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "No se encontro el paquete cdsapi. Instala la dependencia antes de descargar."
        ) from exc

    return cdsapi.Client(url=url, key=key, quiet=False, progress=True)


def ensure_directories() -> None:
    for path in [
        ERA5_ROOT,
        ERA5_MANTARO_ROOT,
        ERA5_SOUTH_AMERICA_ROOT,
        ERA5_MANIFESTS_ROOT,
        OUTPUTS_LOGS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def configure_logging() -> None:
    ensure_directories()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    file_handler = logging.FileHandler(ERA5_LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)
    LOGGER.propagate = False


def load_state() -> dict:
    if not ERA5_STATE_PATH.exists():
        return {"downloads": {}}
    return json.loads(ERA5_STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    ERA5_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def update_state(logical_id: str, status: str, target_path: Path, error: str | None = None) -> None:
    state = load_state()
    state["downloads"][logical_id] = {
        "status": status,
        "target_path": str(target_path),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "error": error,
    }
    save_state(state)


def month_days(year: int, month: int) -> list[str]:
    if month == 2:
        leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        total_days = 29 if leap else 28
    elif month in {4, 6, 9, 11}:
        total_days = 30
    else:
        total_days = 31
    return [f"{day:02d}" for day in range(1, total_days + 1)]


def month_range(start_year: int, start_month: int, end_year: int, end_month: int) -> list[tuple[int, int]]:
    if (end_year, end_month) < (start_year, start_month):
        raise ValueError("La fecha final no puede ser menor que la fecha inicial.")

    pairs: list[tuple[int, int]] = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        pairs.append((year, month))
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
    return pairs


def build_hourly_basin_request(year: int, month: int, variables: list[str]) -> RequestSpec:
    year_dir = ERA5_MANTARO_ROOT / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{year}_{month:02d}.nc"
    target_path = year_dir / filename
    logical_id = f"mantaro_hourly::{year}-{month:02d}"
    manifest_path = ERA5_MANIFESTS_ROOT / f"{logical_id.replace('::', '__')}.json"
    request = {
        "product_type": "reanalysis",
        "variable": variables,
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": month_days(year, month),
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": MANTARO_BASIN_BBOX,
    }
    return RequestSpec(
        dataset="reanalysis-era5-single-levels",
        target_path=target_path,
        manifest_path=manifest_path,
        request=request,
        logical_id=logical_id,
    )


def build_monthly_sa_request(year: int) -> RequestSpec:
    filename = f"{year}.nc"
    target_path = ERA5_SOUTH_AMERICA_ROOT / filename
    logical_id = f"sudamerica_monthly::{year}"
    manifest_path = ERA5_MANIFESTS_ROOT / f"{logical_id.replace('::', '__')}.json"
    request = {
        "product_type": "monthly_averaged_reanalysis",
        "variable": SOUTH_AMERICA_MONTHLY_VARIABLES,
        "year": [str(year)],
        "month": [f"{month:02d}" for month in range(1, 13)],
        "time": "00:00",
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": SOUTH_AMERICA_BBOX,
        "grid": [0.5, 0.5],
    }
    return RequestSpec(
        dataset="reanalysis-era5-single-levels-monthly-means",
        target_path=target_path,
        manifest_path=manifest_path,
        request=request,
        logical_id=logical_id,
    )


def write_manifest(spec: RequestSpec) -> None:
    manifest_payload = {
        "logical_id": spec.logical_id,
        "dataset": spec.dataset,
        "target_path": str(spec.target_path),
        "request": spec.request,
    }
    spec.manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")


def is_valid_file(path: Path, min_bytes: int = 1024) -> bool:
    return path.exists() and path.stat().st_size >= min_bytes


def temporary_path_for(target_path: Path) -> Path:
    return target_path.with_suffix(target_path.suffix + ".part")


def should_skip(spec: RequestSpec) -> bool:
    if is_valid_file(spec.target_path):
        LOGGER.info("Archivo ya presente y valido. Se omite: %s", spec.target_path)
        update_state(spec.logical_id, "skipped_existing", spec.target_path)
        return True
    return False


def retrieve_with_retries(client, spec: RequestSpec, retries: int, retry_wait_seconds: int) -> None:
    temp_path = temporary_path_for(spec.target_path)
    if temp_path.exists():
        LOGGER.warning("Se encontro archivo temporal previo. Se elimina antes de reintentar: %s", temp_path)
        temp_path.unlink()

    for attempt in range(1, retries + 1):
        try:
            LOGGER.info(
                "Descargando %s | intento %s/%s",
                spec.logical_id,
                attempt,
                retries,
            )
            update_state(spec.logical_id, "running", spec.target_path)
            client.retrieve(spec.dataset, spec.request, str(temp_path))

            if not is_valid_file(temp_path):
                raise RuntimeError("El archivo temporal descargado es demasiado pequeno o invalido.")

            temp_path.replace(spec.target_path)
            update_state(spec.logical_id, "completed", spec.target_path)
            LOGGER.info("Descarga completada: %s", spec.target_path)
            return
        except Exception as exc:
            LOGGER.exception("Fallo en %s en el intento %s", spec.logical_id, attempt)
            update_state(spec.logical_id, "failed", spec.target_path, error=str(exc))
            if temp_path.exists():
                temp_path.unlink()

            if attempt == retries:
                raise RuntimeError(f"Error definitivo descargando {spec.logical_id}") from exc

            LOGGER.info("Esperando %s segundos antes del siguiente intento.", retry_wait_seconds)
            time.sleep(retry_wait_seconds)


def parse_year_month(value: str) -> tuple[int, int]:
    year_str, month_str = value.split("-")
    parsed = date(int(year_str), int(month_str), 1)
    return parsed.year, parsed.month


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Descarga robusta y reproducible de ERA5 por bloques."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["mantaro-hourly", "sudamerica-monthly"],
        help="Bloque de descarga a ejecutar.",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Inicio del rango. Formato YYYY-MM para mantaro-hourly, YYYY para sudamerica-monthly.",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="Fin del rango. Formato YYYY-MM para mantaro-hourly, YYYY para sudamerica-monthly.",
    )
    parser.add_argument(
        "--profile",
        choices=["pilot", "core", "extended"],
        default="pilot",
        help="Conjunto de variables para Mantaro.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Numero maximo de reintentos por chunk.",
    )
    parser.add_argument(
        "--retry-wait-seconds",
        type=int,
        default=120,
        help="Espera entre reintentos.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Genera manifests y logging, pero no llama al CDS.",
    )
    return parser


def build_specs(mode: str, start: str, end: str, profile: str) -> list[RequestSpec]:
    if mode == "mantaro-hourly":
        start_year, start_month = parse_year_month(start)
        end_year, end_month = parse_year_month(end)
        if profile in {"pilot", "core"}:
            variables = MANTARO_CORE_VARIABLES
        else:
            variables = MANTARO_EXTENDED_VARIABLES
        return [
            build_hourly_basin_request(year=year, month=month, variables=variables)
            for year, month in month_range(start_year, start_month, end_year, end_month)
        ]

    start_year = int(start)
    end_year = int(end)
    if end_year < start_year:
        raise ValueError("El anio final no puede ser menor que el inicial.")
    return [build_monthly_sa_request(year=year) for year in range(start_year, end_year + 1)]


def execute_downloads(specs: list[RequestSpec], dry_run: bool, retries: int, retry_wait_seconds: int) -> None:
    client = None if dry_run else build_client()

    for spec in specs:
        write_manifest(spec)
        LOGGER.info("Dataset: %s", spec.dataset)
        LOGGER.info("Destino: %s", spec.target_path)
        LOGGER.info("Manifest: %s", spec.manifest_path)

        if dry_run:
            update_state(spec.logical_id, "planned", spec.target_path)
            LOGGER.info("Dry-run activado. No se ejecuta descarga.")
            continue

        if should_skip(spec):
            continue

        retrieve_with_retries(
            client=client,
            spec=spec,
            retries=retries,
            retry_wait_seconds=retry_wait_seconds,
        )

        # Pausa corta para no saturar la API con solicitudes consecutivas.
        time.sleep(5)


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()

    ensure_directories()
    specs = build_specs(
        mode=args.mode,
        start=args.start,
        end=args.end,
        profile=args.profile,
    )
    LOGGER.info("Se prepararon %s chunks para descarga.", len(specs))
    execute_downloads(
        specs=specs,
        dry_run=args.dry_run,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
    )
    LOGGER.info("Proceso ERA5 finalizado.")


if __name__ == "__main__":
    main()
