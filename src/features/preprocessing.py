from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.config import (
    EXCLUDED_FEATURE_COLUMNS,
    FORECAST_HORIZON_HOURS,
    FROST_THRESHOLD_C,
    GROUP_LIMITS,
    TRAIN_TARGET_END,
    VALIDATION_TARGET_END,
)


def apply_physical_constraints(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned_df = dataframe.copy()
    quality_rows: list[dict[str, object]] = []

    for prefix, (lower_limit, upper_limit) in GROUP_LIMITS.items():
        matching_columns = [column for column in cleaned_df.columns if column.startswith(f"{prefix}_")]
        for column in matching_columns:
            if column.endswith("_std"):
                # Las desviaciones estandar representan dispersion y no deben
                # evaluarse con el mismo rango fisico que medias/minimos/maximos.
                column_lower_limit, column_upper_limit = 0.0, 200.0
            else:
                column_lower_limit, column_upper_limit = lower_limit, upper_limit

            invalid_mask = (cleaned_df[column] < column_lower_limit) | (
                cleaned_df[column] > column_upper_limit
            )
            invalid_count = int(invalid_mask.sum())
            quality_rows.append(
                {
                    "column": column,
                    "lower_limit": column_lower_limit,
                    "upper_limit": column_upper_limit,
                    "invalid_count": invalid_count,
                    "invalid_pct": round((invalid_count / len(cleaned_df)) * 100, 4),
                }
            )
            cleaned_df.loc[invalid_mask, column] = np.nan

    quality_report = pd.DataFrame(quality_rows).sort_values(
        by=["invalid_count", "column"], ascending=[False, True]
    )
    return cleaned_df, quality_report


def add_temporal_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    enriched_df = dataframe.copy()
    timestamps = enriched_df.index

    enriched_df["hour"] = timestamps.hour
    enriched_df["month"] = timestamps.month
    enriched_df["dayofyear"] = timestamps.dayofyear
    enriched_df["dayofweek"] = timestamps.dayofweek
    enriched_df["is_night"] = timestamps.hour.isin(range(0, 7)).astype(int)
    enriched_df["is_dry_season"] = timestamps.month.isin([5, 6, 7, 8, 9]).astype(int)

    enriched_df["hour_sin"] = np.sin(2 * np.pi * enriched_df["hour"] / 24)
    enriched_df["hour_cos"] = np.cos(2 * np.pi * enriched_df["hour"] / 24)
    enriched_df["dayofyear_sin"] = np.sin(2 * np.pi * enriched_df["dayofyear"] / 365.25)
    enriched_df["dayofyear_cos"] = np.cos(2 * np.pi * enriched_df["dayofyear"] / 365.25)

    return enriched_df


def build_processed_dataset(
    cleaned_hourly_df: pd.DataFrame,
    horizon_hours: int = FORECAST_HORIZON_HOURS,
    frost_threshold_c: float = FROST_THRESHOLD_C,
) -> pd.DataFrame:
    processed_df = add_temporal_features(cleaned_hourly_df)
    processed_df["frost_event_current"] = (
        processed_df["tempsup_min"] <= frost_threshold_c
    ).astype(int)
    processed_df[f"frost_event_t_plus_{horizon_hours}h"] = processed_df["frost_event_current"].shift(
        -horizon_hours
    )
    processed_df["target_timestamp"] = processed_df.index + pd.to_timedelta(horizon_hours, unit="h")

    processed_df = processed_df.dropna(
        subset=[f"frost_event_t_plus_{horizon_hours}h"]
    ).copy()
    processed_df[f"frost_event_t_plus_{horizon_hours}h"] = processed_df[
        f"frost_event_t_plus_{horizon_hours}h"
    ].astype(int)

    return processed_df


def select_feature_columns(columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column not in EXCLUDED_FEATURE_COLUMNS]


def build_target_split_masks(
    processed_df: pd.DataFrame,
    train_target_end: str = TRAIN_TARGET_END,
    validation_target_end: str = VALIDATION_TARGET_END,
) -> dict[str, pd.Series]:
    target_timestamps = pd.to_datetime(processed_df["target_timestamp"])
    train_mask = target_timestamps <= pd.Timestamp(train_target_end)
    validation_mask = (target_timestamps > pd.Timestamp(train_target_end)) & (
        target_timestamps <= pd.Timestamp(validation_target_end)
    )
    test_mask = target_timestamps > pd.Timestamp(validation_target_end)

    return {
        "train": train_mask,
        "validation": validation_mask,
        "test": test_mask,
    }


def impute_features_past_only(
    feature_df: pd.DataFrame,
    train_mask: pd.Series,
    ffill_limit: int = 6,
) -> tuple[pd.DataFrame, pd.Series]:
    imputed_df = feature_df.sort_index().ffill(limit=ffill_limit)
    train_reference = imputed_df.loc[train_mask]
    train_medians = train_reference.median(numeric_only=True)
    imputed_df = imputed_df.fillna(train_medians)
    return imputed_df, train_medians
