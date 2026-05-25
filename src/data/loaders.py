from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.config import DATA_EXTERNAL_DIR, DATA_RAW_DIR, RAW_DATASETS


def _flatten_raw_columns(path: Path, dataframe: pd.DataFrame) -> pd.DataFrame:
    prefix = path.stem.split("_")[0]
    renamed_columns: list[str] = []

    for level_0, level_1 in dataframe.columns:
        if str(level_0).startswith("Unnamed: 0"):
            renamed_columns.append("timestamp")
        else:
            renamed_columns.append(f"{prefix}_{level_1}")

    dataframe = dataframe.copy()
    dataframe.columns = renamed_columns
    dataframe = dataframe[dataframe["timestamp"].ne("time")].copy()
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])

    for column in dataframe.columns:
        if column != "timestamp":
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    return dataframe.set_index("timestamp").sort_index()


def load_hourly_variable_csv(path: Path) -> pd.DataFrame:
    raw_df = pd.read_csv(path, header=[0, 1])
    return _flatten_raw_columns(path=path, dataframe=raw_df)


def resolve_dataset_path(variable_name: str, filename: str, data_dir: Path) -> Path:
    env_var = f"FROST_{variable_name.upper()}_SOURCE_PATH"
    candidates = [data_dir / filename, DATA_EXTERNAL_DIR / filename]

    configured_path = os.getenv(env_var)
    if configured_path:
        candidates.append(Path(configured_path).expanduser())

    for path in candidates:
        if path.exists():
            return path

    searched = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "No se encontro el archivo requerido para "
        f"`{variable_name}`.\nSe buscaron estas rutas:\n{searched}\n"
        f"Tambien puedes definir la variable de entorno `{env_var}`."
    )


def load_raw_hourly_dataset(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = data_dir or DATA_RAW_DIR
    dataframes = []

    for variable_name, filename in RAW_DATASETS.items():
        path = resolve_dataset_path(variable_name=variable_name, filename=filename, data_dir=data_dir)
        dataframe = load_hourly_variable_csv(path)
        # El archivo de presion viene como patm en la cabecera original, pero
        # mantenemos el prefijo de archivo para que el repositorio sea consistente.
        if variable_name == "press":
            dataframe = dataframe.rename(columns=lambda col: col.replace("patm_", "press_"))
        dataframes.append(dataframe)

    merged_df = pd.concat(dataframes, axis=1).sort_index()
    merged_df.index.name = "timestamp"
    return merged_df


def dataset_inventory(dataframe: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(dataframe)),
        "columns": int(dataframe.shape[1]),
        "start": dataframe.index.min(),
        "end": dataframe.index.max(),
        "missing_total": int(dataframe.isna().sum().sum()),
        "rows_with_missing": int(dataframe.isna().any(axis=1).sum()),
        "duplicate_timestamps": int(dataframe.index.duplicated().sum()),
    }
