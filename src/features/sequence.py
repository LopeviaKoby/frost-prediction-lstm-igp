from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SequenceSplit:
    X: np.ndarray
    y: np.ndarray
    origin_timestamps: pd.DatetimeIndex
    target_timestamps: pd.DatetimeIndex


def create_sequence_split(
    feature_array: np.ndarray,
    target_array: np.ndarray,
    origin_timestamps: pd.DatetimeIndex,
    target_timestamps: pd.DatetimeIndex,
    selected_positions: np.ndarray,
    sequence_length: int,
) -> SequenceSplit:
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    origins: list[pd.Timestamp] = []
    target_times: list[pd.Timestamp] = []

    for position in selected_positions:
        if position < sequence_length - 1:
            continue

        window_start = position - sequence_length + 1
        window = feature_array[window_start : position + 1]

        sequences.append(window)
        targets.append(target_array[position])
        origins.append(origin_timestamps[position])
        target_times.append(target_timestamps[position])

    return SequenceSplit(
        X=np.asarray(sequences, dtype=np.float32),
        y=np.asarray(targets, dtype=np.float32),
        origin_timestamps=pd.DatetimeIndex(origins),
        target_timestamps=pd.DatetimeIndex(target_times),
    )
