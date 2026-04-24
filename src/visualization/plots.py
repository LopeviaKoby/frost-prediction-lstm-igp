from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay


def plot_training_history(history, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="validation")
    axes[0].set_title("Curva de perdida")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Binary cross-entropy")
    axes[0].legend()

    axes[1].plot(history.history["recall"], label="train")
    axes[1].plot(history.history["val_recall"], label="validation")
    axes[1].set_title("Recall por epoca")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Recall")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrix(y_true, y_pred, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["No helada", "Helada"],
        cmap="Blues",
        ax=axis,
        colorbar=False,
    )
    axis.set_title("Matriz de confusion")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_roc_curve(y_true, y_score, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=axis)
    axis.set_title("Curva ROC")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
