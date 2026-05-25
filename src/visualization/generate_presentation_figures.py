from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from src.config import (
    BASELINE_VERSION,
    OUTPUTS_METRICS_DIR,
    OUTPUTS_PREDICTIONS_DIR,
    REPORTS_FIGURES_DIR,
    TARGET_COLUMN,
)


plt.style.use("seaborn-v0_8-whitegrid")

PRESENTATION_PREFIX = f"presentation_{BASELINE_VERSION}"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    processed_df = pd.read_csv(
        f"data/processed/frost_dataset_{BASELINE_VERSION}.csv",
        parse_dates=["timestamp", "target_timestamp"],
    )
    predictions_df = pd.read_csv(
        OUTPUTS_PREDICTIONS_DIR / f"lstm_baseline_{BASELINE_VERSION}_predictions.csv",
        parse_dates=["origin_timestamp", "target_timestamp"],
    )
    metrics = json.loads(
        (OUTPUTS_METRICS_DIR / f"lstm_baseline_{BASELINE_VERSION}_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    return processed_df, predictions_df, metrics


def save_figure(fig: plt.Figure, filename: str) -> Path:
    output_path = REPORTS_FIGURES_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_frost_rate_by_month_and_hour(processed_df: pd.DataFrame) -> Path:
    monthly_rate = processed_df.groupby("month")["frost_event_current"].mean() * 100
    hourly_rate = processed_df.groupby("hour")["frost_event_current"].mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    monthly_rate.plot(kind="bar", color="#1f77b4", ax=axes[0])
    axes[0].set_title("Tasa de helada por mes")
    axes[0].set_xlabel("Mes")
    axes[0].set_ylabel("Porcentaje de horas con helada")
    axes[0].tick_params(axis="x", rotation=0)

    hourly_rate.plot(kind="bar", color="#d62728", ax=axes[1])
    axes[1].set_title("Tasa de helada por hora")
    axes[1].set_xlabel("Hora del dia")
    axes[1].set_ylabel("Porcentaje de horas con helada")
    axes[1].tick_params(axis="x", rotation=0)

    return save_figure(fig, f"{PRESENTATION_PREFIX}_frost_rate_by_month_hour.png")


def plot_frost_heatmap_month_hour(processed_df: pd.DataFrame) -> Path:
    heatmap_df = (
        processed_df.groupby(["month", "hour"])["frost_event_current"]
        .mean()
        .mul(100)
        .unstack(fill_value=0)
        .reindex(index=range(1, 13), columns=range(24), fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(12, 5.5))
    image = ax.imshow(heatmap_df.values, aspect="auto", cmap="YlOrRd")
    ax.set_title("Mapa de calor: incidencia de heladas por mes y hora")
    ax.set_xlabel("Hora del dia")
    ax.set_ylabel("Mes")
    ax.set_xticks(range(24))
    ax.set_yticks(range(12))
    ax.set_yticklabels(heatmap_df.index)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("% de horas con helada")

    return save_figure(fig, f"{PRESENTATION_PREFIX}_frost_heatmap_month_hour.png")


def plot_yearly_frost_counts(processed_df: pd.DataFrame) -> Path:
    yearly_stats = (
        processed_df.assign(year=processed_df["timestamp"].dt.year)
        .groupby("year")["frost_event_current"]
        .agg(["sum", "mean"])
        .rename(columns={"sum": "event_count", "mean": "event_rate"})
    )

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.bar(yearly_stats.index.astype(str), yearly_stats["event_count"], color="#2ca02c")
    ax1.set_ylabel("Numero de horas con helada")
    ax1.set_xlabel("Año")
    ax1.set_title("Evolucion anual de horas con helada")

    ax2 = ax1.twinx()
    ax2.plot(
        yearly_stats.index.astype(str),
        yearly_stats["event_rate"] * 100,
        color="#ff7f0e",
        marker="o",
        linewidth=2,
    )
    ax2.set_ylabel("Tasa anual de helada (%)")

    return save_figure(fig, f"{PRESENTATION_PREFIX}_yearly_frost_counts.png")


def plot_morning_frost_incidence_and_intensity(processed_df: pd.DataFrame) -> Path:
    frost_df = processed_df.loc[processed_df["frost_event_current"] == 1].copy()
    morning_df = frost_df.loc[frost_df["hour"].between(0, 9)]

    summary = (
        morning_df.groupby("hour")
        .agg(
            frost_hours=("frost_event_current", "count"),
            mean_temp=(TARGET_COLUMN, "mean"),
            median_temp=(TARGET_COLUMN, "median"),
        )
        .reindex(range(0, 10), fill_value=0)
    )

    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.bar(summary.index.astype(str), summary["frost_hours"], color="#4c78a8")
    ax1.set_title("Horas de mayor incidencia e intensidad historica de heladas")
    ax1.set_xlabel("Hora del dia")
    ax1.set_ylabel("Numero historico de horas con helada")

    ax2 = ax1.twinx()
    ax2.plot(
        summary.index.astype(str),
        summary["median_temp"],
        color="#e45756",
        marker="o",
        label=f"Mediana {TARGET_COLUMN}",
    )
    ax2.plot(
        summary.index.astype(str),
        summary["mean_temp"],
        color="#72b7b2",
        marker="o",
        linestyle="--",
        label=f"Media {TARGET_COLUMN}",
    )
    ax2.set_ylabel("Temperatura minima durante horas con helada (C)")
    ax2.legend(loc="upper right")

    return save_figure(fig, f"{PRESENTATION_PREFIX}_morning_frost_incidence_intensity.png")


def plot_key_feature_distributions(processed_df: pd.DataFrame) -> Path:
    feature_specs = [
        (TARGET_COLUMN, "Temperatura minima objetivo (C)", "#1f77b4"),
        ("HR_mean", "Humedad relativa media (%)", "#2ca02c"),
        ("radinf_mean", "Radiacion infrarroja media", "#9467bd"),
        ("vel_mean", "Velocidad media del viento (m/s)", "#ff7f0e"),
    ]

    sample_df = processed_df[processed_df["frost_event_current"].isin([0, 1])].copy()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    labels = {0: "No helada", 1: "Helada"}

    for axis, (column, title, color) in zip(axes, feature_specs, strict=True):
        class0 = sample_df.loc[sample_df["frost_event_current"] == 0, column].dropna()
        class1 = sample_df.loc[sample_df["frost_event_current"] == 1, column].dropna()
        axis.hist(class0, bins=40, alpha=0.55, density=True, label=labels[0], color="#9ecae1")
        axis.hist(class1, bins=40, alpha=0.65, density=True, label=labels[1], color=color)
        axis.set_title(title)
        axis.legend()

    fig.suptitle("Variables con mayor contraste entre horas con y sin helada", y=1.02)
    return save_figure(fig, f"{PRESENTATION_PREFIX}_key_feature_distributions.png")


def plot_prediction_timeline_winter_sample(predictions_df: pd.DataFrame) -> Path:
    winter_df = predictions_df.loc[predictions_df["target_timestamp"].dt.month.isin([7, 8])].copy()
    if winter_df.empty:
        sample = predictions_df.iloc[:24 * 21].copy()
    else:
        rolling_hours = 24 * 14
        winter_df = winter_df.sort_values("target_timestamp").reset_index(drop=True)
        winter_df["event_roll"] = winter_df["y_true"].rolling(window=rolling_hours, min_periods=24).sum()
        center_idx = int(winter_df["event_roll"].fillna(0).idxmax())
        start_idx = max(0, center_idx - 24 * 7)
        end_idx = min(len(winter_df), start_idx + 24 * 21)
        sample = winter_df.iloc[start_idx:end_idx].copy()

    fig, ax = plt.subplots(figsize=(12, 4.5))

    ax.plot(sample["target_timestamp"], sample["y_score"], color="#1f77b4", linewidth=1.8, label="Probabilidad LSTM")
    ax.axhline(0.5, color="#444444", linestyle="--", linewidth=1, label="Umbral 0.5")

    frost_points = sample.loc[sample["y_true"] == 1]
    ax.scatter(
        frost_points["target_timestamp"],
        frost_points["y_score"],
        color="#d62728",
        s=28,
        label="Eventos reales de helada",
        zorder=3,
    )

    ax.set_title("Riesgo pronosticado en una ventana de prueba de invierno (julio-agosto)")
    ax.set_xlabel("Tiempo objetivo")
    ax.set_ylabel("Probabilidad pronosticada")
    ax.legend(loc="upper right")

    return save_figure(fig, f"{PRESENTATION_PREFIX}_prediction_timeline_winter_sample.png")


def plot_threshold_tradeoff(predictions_df: pd.DataFrame) -> Path:
    thresholds = np.arange(0.1, 0.91, 0.05)
    rows = []

    for threshold in thresholds:
        preds = (predictions_df["y_score"] >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(predictions_df["y_true"], preds, zero_division=0),
                "recall": recall_score(predictions_df["y_true"], preds, zero_division=0),
                "f1": f1_score(predictions_df["y_true"], preds, zero_division=0),
            }
        )

    tradeoff_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(tradeoff_df["threshold"], tradeoff_df["precision"], marker="o", label="Precision")
    ax.plot(tradeoff_df["threshold"], tradeoff_df["recall"], marker="o", label="Recall")
    ax.plot(tradeoff_df["threshold"], tradeoff_df["f1"], marker="o", label="F1")
    ax.axvline(0.5, color="#444444", linestyle="--", linewidth=1, label="Umbral actual")
    ax.set_title("Trade-off de metricas segun umbral de decision")
    ax.set_xlabel("Umbral")
    ax.set_ylabel("Score")
    ax.legend(loc="best")

    return save_figure(fig, f"{PRESENTATION_PREFIX}_threshold_tradeoff.png")


def plot_model_metric_summary(metrics: dict) -> Path:
    model_metrics = metrics["test_metrics"]
    labels = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    values = [model_metrics[label] for label in labels]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2"]
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_title("Resumen de metricas del baseline LSTM")
    ax.set_ylabel("Score")

    for idx, value in enumerate(values):
        ax.text(idx, value + 0.02, f"{value:.3f}", ha="center", fontsize=9)

    return save_figure(fig, f"{PRESENTATION_PREFIX}_metrics_summary.png")


def main() -> None:
    processed_df, predictions_df, metrics = load_inputs()
    generated = [
        plot_frost_rate_by_month_and_hour(processed_df),
        plot_frost_heatmap_month_hour(processed_df),
        plot_yearly_frost_counts(processed_df),
        plot_morning_frost_incidence_and_intensity(processed_df),
        plot_key_feature_distributions(processed_df),
        plot_prediction_timeline_winter_sample(predictions_df),
        plot_threshold_tradeoff(predictions_df),
        plot_model_metric_summary(metrics),
    ]

    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
