import os
from pathlib import Path

# Cargar .env manualmente si no se usa python-dotenv
def _load_env():
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

_load_env()

# Configuraciones de API
CDSAPI_URL = os.getenv("CDSAPI_URL")
CDSAPI_KEY = os.getenv("CDSAPI_KEY")

def _parse_bbox(val: str) -> list[int]:
    return [int(x) for x in val.split(",")] if val else []

# Bounding Boxes
ERA5_MANTARO_BBOX = _parse_bbox(os.getenv("ERA5_MANTARO_BBOX", "-10,-77,-14,-74"))
ERA5_SOUTH_AMERICA_BBOX = _parse_bbox(os.getenv("ERA5_SOUTH_AMERICA_BBOX", "15,-90,-60,-15"))

# Reintentos de Descarga
ERA5_MAX_RETRIES = int(os.getenv("ERA5_MAX_RETRIES", "3"))
ERA5_RETRY_WAIT_SECONDS = int(os.getenv("ERA5_RETRY_WAIT_SECONDS", "120"))

# Variables para validación
ERA5_VALIDATION_VERSION = os.getenv("ERA5_VALIDATION_VERSION", "v01")

# Variables Climáticas
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
