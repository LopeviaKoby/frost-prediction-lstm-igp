from __future__ import annotations

from pathlib import Path

from src.config import (
    BASELINE_VERSION,
    DATA_INTERIM_DIR,
    DATA_PROCESSED_DIR,
    REPORTS_TABLES_DIR,
)
from src.data.loaders import dataset_inventory, load_raw_hourly_dataset
from src.features.preprocessing import apply_physical_constraints, build_processed_dataset


def build_and_save_datasets() -> dict[str, Path]:
    raw_df = load_raw_hourly_dataset()
    cleaned_df, quality_report = apply_physical_constraints(raw_df)
    processed_df = build_processed_dataset(cleaned_df)

    DATA_INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    interim_path = DATA_INTERIM_DIR / f"frost_hourly_clean_{BASELINE_VERSION}.csv"
    processed_path = DATA_PROCESSED_DIR / f"frost_dataset_{BASELINE_VERSION}.csv"
    quality_report_path = REPORTS_TABLES_DIR / f"data_quality_report_{BASELINE_VERSION}.csv"
    inventory_path = REPORTS_TABLES_DIR / f"dataset_inventory_{BASELINE_VERSION}.md"

    cleaned_df.to_csv(interim_path, index=True)
    processed_df.to_csv(processed_path, index=True)
    quality_report.to_csv(quality_report_path, index=False)

    inventory = dataset_inventory(cleaned_df)
    inventory_path.write_text(
        "\n".join(
            [
                "# Inventario del Dataset v01",
                "",
                f"- Filas: {inventory['rows']}",
                f"- Columnas: {inventory['columns']}",
                f"- Inicio: {inventory['start']}",
                f"- Fin: {inventory['end']}",
                f"- Valores faltantes totales: {inventory['missing_total']}",
                f"- Filas con faltantes: {inventory['rows_with_missing']}",
                f"- Timestamps duplicados: {inventory['duplicate_timestamps']}",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "interim_path": interim_path,
        "processed_path": processed_path,
        "quality_report_path": quality_report_path,
        "inventory_path": inventory_path,
    }


if __name__ == "__main__":
    outputs = build_and_save_datasets()
    for name, path in outputs.items():
        print(f"{name}: {path}")
