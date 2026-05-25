from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import xarray as xr

from src.config import DATA_RAW_DIR, REPORTS_TABLES_DIR


ERA5_VALIDATION_VERSION = "v01"
ERA5_ROOT = DATA_RAW_DIR / "era5"
ERA5_HOURLY_ROOT = ERA5_ROOT / "mantaro_hourly"
ERA5_MONTHLY_ROOT = ERA5_ROOT / "sudamerica_monthly"
VALIDATION_CSV_PATH = REPORTS_TABLES_DIR / f"era5_validation_{ERA5_VALIDATION_VERSION}.csv"
VALIDATION_JSON_PATH = REPORTS_TABLES_DIR / f"era5_validation_{ERA5_VALIDATION_VERSION}.json"
VALIDATION_SUMMARY_CSV_PATH = REPORTS_TABLES_DIR / f"era5_validation_{ERA5_VALIDATION_VERSION}_summary.csv"
VALIDATION_SUMMARY_JSON_PATH = REPORTS_TABLES_DIR / f"era5_validation_{ERA5_VALIDATION_VERSION}_summary.json"


@dataclass(frozen=True)
class ValidationRow:
    block: str
    file_path: str
    member: str
    status: str
    coord_name: str | None
    n_time: int | None
    expected_time: int | None
    start: str | None
    end: str | None
    duplicate_timestamps: int | None
    missing_values: int | None
    variables: str | None
    note: str | None


def _parse_month_from_stem(stem: str) -> tuple[int, int] | None:
    try:
        year_str, month_str = stem.split("_", 1)
        return int(year_str), int(month_str)
    except ValueError:
        return None


def _expected_hours_for_month(year: int, month: int) -> int:
    return int(pd.Period(f"{year:04d}-{month:02d}", freq="M").days_in_month * 24)


def _open_dataset(path: Path) -> xr.Dataset:
    try:
        return xr.open_dataset(path)
    except Exception:
        return xr.open_dataset(path, engine="netcdf4")


def _pick_time_coord(dataset: xr.Dataset) -> str | None:
    for candidate in ("time", "valid_time"):
        if candidate in dataset.coords:
            return candidate
    return None


def _summarize_dataset(
    block: str,
    file_path: Path,
    member: str,
    dataset: xr.Dataset,
    expected_time: int | None,
    flag_missing_values: bool = True,
) -> ValidationRow:
    coord_name = _pick_time_coord(dataset)
    if coord_name is None:
        return ValidationRow(
            block=block,
            file_path=str(file_path),
            member=member,
            status="failed",
            coord_name=None,
            n_time=None,
            expected_time=expected_time,
            start=None,
            end=None,
            duplicate_timestamps=None,
            missing_values=None,
            variables=",".join(sorted(dataset.data_vars)) if dataset.data_vars else None,
            note="No se encontro coordenada temporal (time/valid_time).",
        )

    timestamps = pd.to_datetime(dataset[coord_name].values)
    missing_values = int(sum(dataset[var].isnull().sum().item() for var in dataset.data_vars))
    duplicate_timestamps = int(pd.Index(timestamps).duplicated().sum())
    status = "ok"
    note_parts: list[str] = []

    if expected_time is not None and len(timestamps) != expected_time:
        status = "warning"
        note_parts.append(f"Se esperaban {expected_time} pasos temporales y se encontraron {len(timestamps)}.")
    if duplicate_timestamps:
        status = "warning"
        note_parts.append(f"Se detectaron {duplicate_timestamps} timestamps duplicados.")
    if missing_values and flag_missing_values:
        status = "warning"
        note_parts.append(f"Se detectaron {missing_values} valores faltantes.")

    return ValidationRow(
        block=block,
        file_path=str(file_path),
        member=member,
        status=status,
        coord_name=coord_name,
        n_time=int(len(timestamps)),
        expected_time=expected_time,
        start=pd.Timestamp(timestamps[0]).isoformat(),
        end=pd.Timestamp(timestamps[-1]).isoformat(),
        duplicate_timestamps=duplicate_timestamps,
        missing_values=missing_values,
        variables=",".join(sorted(dataset.data_vars)),
        note=" ".join(note_parts) if note_parts else None,
    )


def validate_hourly_block(root: Path = ERA5_HOURLY_ROOT) -> list[ValidationRow]:
    rows: list[ValidationRow] = []
    for file_path in sorted(root.rglob("*.nc")):
        month_info = _parse_month_from_stem(file_path.stem)
        expected_hours = _expected_hours_for_month(*month_info) if month_info else None

        if zipfile.is_zipfile(file_path):
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(file_path) as archive:
                    members = [name for name in archive.namelist() if name.endswith(".nc")]
                    if len(members) != 2:
                        rows.append(
                            ValidationRow(
                                block="hourly",
                                file_path=str(file_path),
                                member=",".join(members) if members else "",
                                status="warning",
                                coord_name=None,
                                n_time=None,
                                expected_time=expected_hours,
                                start=None,
                                end=None,
                                duplicate_timestamps=None,
                                missing_values=None,
                                variables=None,
                                note=f"Se esperaban 2 NetCDF internos y se encontraron {len(members)}.",
                            )
                        )
                    for member in members:
                        extracted = Path(archive.extract(member, path=tmpdir))
                        ds = _open_dataset(extracted)
                        try:
                            rows.append(
                                _summarize_dataset(
                                    block="hourly",
                                    file_path=file_path,
                                    member=member,
                                    dataset=ds,
                                    expected_time=expected_hours,
                                )
                            )
                        finally:
                            ds.close()
        else:
            ds = _open_dataset(file_path)
            try:
                rows.append(
                    _summarize_dataset(
                        block="hourly",
                        file_path=file_path,
                        member=file_path.name,
                        dataset=ds,
                        expected_time=expected_hours,
                    )
                )
            finally:
                ds.close()
    return rows


def validate_monthly_block(root: Path = ERA5_MONTHLY_ROOT) -> list[ValidationRow]:
    rows: list[ValidationRow] = []
    for file_path in sorted(root.rglob("*.nc")):
        if zipfile.is_zipfile(file_path):
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(file_path) as archive:
                    members = [name for name in archive.namelist() if name.endswith(".nc")]
                    if len(members) != 2:
                        rows.append(
                            ValidationRow(
                                block="monthly",
                                file_path=str(file_path),
                                member=",".join(members) if members else "",
                                status="warning",
                                coord_name=None,
                                n_time=None,
                                expected_time=12,
                                start=None,
                                end=None,
                                duplicate_timestamps=None,
                                missing_values=None,
                                variables=None,
                                note=f"Se esperaban 2 NetCDF internos y se encontraron {len(members)}.",
                            )
                        )
                    for member in members:
                        extracted = Path(archive.extract(member, path=tmpdir))
                        ds = _open_dataset(extracted)
                        try:
                            rows.append(
                                _summarize_dataset(
                                    block="monthly",
                                    file_path=file_path,
                                    member=member,
                                    dataset=ds,
                                    expected_time=12,
                                    flag_missing_values=False,
                                )
                            )
                        finally:
                            ds.close()
        else:
            ds = _open_dataset(file_path)
            try:
                rows.append(
                    _summarize_dataset(
                        block="monthly",
                        file_path=file_path,
                        member=file_path.name,
                        dataset=ds,
                        expected_time=12,
                        flag_missing_values=False,
                    )
                )
            finally:
                ds.close()
    return rows


def build_report(rows: list[ValidationRow]) -> pd.DataFrame:
    return pd.DataFrame([row.__dict__ for row in rows])


def build_summary(report: pd.DataFrame) -> dict[str, object]:
    if report.empty:
        return {
            "rows": 0,
            "ok_rows": 0,
            "warning_rows": 0,
            "failed_rows": 0,
            "hourly_files": 0,
            "monthly_files": 0,
        }

    return {
        "rows": int(len(report)),
        "ok_rows": int((report["status"] == "ok").sum()),
        "warning_rows": int((report["status"] == "warning").sum()),
        "failed_rows": int((report["status"] == "failed").sum()),
        "hourly_files": int((report["block"] == "hourly").sum()),
        "monthly_files": int((report["block"] == "monthly").sum()),
    }


def build_readable_summary(report: pd.DataFrame) -> pd.DataFrame:
    if report.empty:
        return pd.DataFrame(
            columns=[
                "block",
                "file_path",
                "status",
                "members",
                "n_time_min",
                "n_time_max",
                "expected_time",
                "missing_values",
                "duplicate_timestamps",
                "start",
                "end",
                "note",
            ]
        )

    rows: list[dict[str, object]] = []
    for (block, file_path), group in report.groupby(["block", "file_path"], sort=True):
        statuses = set(group["status"].astype(str))
        if "failed" in statuses:
            status = "failed"
        elif "warning" in statuses:
            status = "warning"
        else:
            status = "ok"

        notes = [note for note in group["note"].dropna().astype(str).tolist() if note]
        rows.append(
            {
                "block": block,
                "file_path": file_path,
                "status": status,
                "members": int(len(group)),
                "n_time_min": int(group["n_time"].min()) if group["n_time"].notna().any() else None,
                "n_time_max": int(group["n_time"].max()) if group["n_time"].notna().any() else None,
                "expected_time": int(group["expected_time"].dropna().iloc[0]) if group["expected_time"].notna().any() else None,
                "missing_values": int(group["missing_values"].fillna(0).sum()),
                "duplicate_timestamps": int(group["duplicate_timestamps"].fillna(0).sum()),
                "start": group["start"].dropna().min() if group["start"].notna().any() else None,
                "end": group["end"].dropna().max() if group["end"].notna().any() else None,
                "note": " | ".join(notes) if notes else None,
            }
        )

    return pd.DataFrame(rows).sort_values(["block", "file_path"]).reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validacion ligera de los bloques ERA5 descargados."
    )
    parser.add_argument(
        "--block",
        choices=["hourly", "monthly", "both"],
        default="both",
        help="Bloque ERA5 a validar.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    rows: list[ValidationRow] = []
    if args.block in {"hourly", "both"}:
        rows.extend(validate_hourly_block())
    if args.block in {"monthly", "both"}:
        rows.extend(validate_monthly_block())

    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report(rows)
    readable_report = build_readable_summary(report)
    report.to_csv(VALIDATION_CSV_PATH, index=False)
    readable_report.to_csv(VALIDATION_SUMMARY_CSV_PATH, index=False)
    VALIDATION_JSON_PATH.write_text(
        json.dumps(build_summary(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    VALIDATION_SUMMARY_JSON_PATH.write_text(
        json.dumps(
            {
                "rows": int(len(readable_report)),
                "ok_rows": int((readable_report["status"] == "ok").sum()) if not readable_report.empty else 0,
                "warning_rows": int((readable_report["status"] == "warning").sum()) if not readable_report.empty else 0,
                "failed_rows": int((readable_report["status"] == "failed").sum()) if not readable_report.empty else 0,
                "hourly_files": int((readable_report["block"] == "hourly").sum()) if not readable_report.empty else 0,
                "monthly_files": int((readable_report["block"] == "monthly").sum()) if not readable_report.empty else 0,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = build_summary(report)
    print(f"validation_csv: {VALIDATION_CSV_PATH}")
    print(f"validation_summary_csv: {VALIDATION_SUMMARY_CSV_PATH}")
    print(f"validation_json: {VALIDATION_JSON_PATH}")
    print(f"validation_summary_json: {VALIDATION_SUMMARY_JSON_PATH}")
    print(f"rows: {summary['rows']}")
    print(f"ok_rows: {summary['ok_rows']}")
    print(f"warning_rows: {summary['warning_rows']}")
    print(f"failed_rows: {summary['failed_rows']}")
    print(f"hourly_rows: {summary['hourly_files']}")
    print(f"monthly_rows: {summary['monthly_files']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
