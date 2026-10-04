"""
=============================================================================
AI-Based Tuberculosis (TB) Screening from Chest X-Ray
Model Evaluation Pipeline
=============================================================================
Evaluates the trained model against the unseen held-out test dataset.
Calculates Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score,
and ROC-AUC. Generates Confusion Matrix and ROC Curve plots.
Updates comparison ledger with measured results.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from preprocessing.image_preprocessor import preprocess_image
from utils.helpers import load_json, save_json, scan_dataset

def load_test_dataset() -> Tuple[List[str], List[int]]:
    """
    Loads the held-out test dataset paths and labels saved during training.
    If test_split.json is missing, generates a fresh deterministic test split.
    """
    test_split_path = config.MODELS_DIR / "test_split.json"
    if test_split_path.exists():
        split_info = load_json(test_split_path)
        if split_info and "test_paths" in split_info and "test_labels" in split_info:
            valid_paths = []
            valid_labels = []
            for p, l in zip(split_info["test_paths"], split_info["test_labels"]):
                if Path(p).exists():
                    valid_paths.append(p)
                    valid_labels.append(int(l))
            if valid_paths:
                return valid_paths, valid_labels

    # Fallback: scan dataset and take 20%
    stats = scan_dataset()
    if stats["total_images"] == 0:
        raise ValueError("No images found in dataset/ to evaluate. Run sample_data_generator.py first.")

    tb_files = sorted([str(f) for f in config.TB_DIR.glob("*") if f.suffix.lower().replace(".", "") in config.ALLOWED_EXTENSIONS])
    normal_files = sorted([str(f) for f in config.NORMAL_DIR.glob("*") if f.suffix.lower().replace(".", "") in config.ALLOWED_EXTENSIONS])

    # Take last 20% of each class for evaluation
    tb_test = tb_files[int(len(tb_files) * 0.8):] or tb_files[:2]
    normal_test = normal_files[int(len(normal_files) * 0.8):] or normal_files[:2]

    paths = normal_test + tb_test
    labels = [0] * len(normal_test) + [1] * len(tb_test)
    return paths, labels


def plot_confusion_matrix(cm: np.ndarray, model_name: str) -> str:
    """Renders and saves a styled Seaborn Confusion Matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    fig.patch.set_facecolor("#0e1726")
    ax.set_facecolor("#111c30")

    labels = ["Normal", "TB"]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        linewidths=1.5,
        linecolor="#1e293b",
        ax=ax,
        annot_kws={"size": 14, "weight": "bold", "color": "#0f172a"}
    )

    ax.set_title(f"Confusion Matrix - {model_name}", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Diagnosis", color="#94a3b8", fontsize=10, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Diagnosis (Ground Truth)", color="#94a3b8", fontsize=10, fontweight="bold", labelpad=8)
    ax.tick_params(colors="#94a3b8")

    plt.tight_layout()
    output_path = config.PLOTS_DIR / "confusion_matrix.png"
    fig.savefig(str(output_path), facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return "/static/plots/confusion_matrix.png"


def plot_roc_curve(y_true: List[int], y_prob: List[float], auc_score: float, model_name: str) -> str:
    """Renders and saves the ROC (Receiver Operating Characteristic) curve."""
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=140)
    fig.patch.set_facecolor("#0e1726")
    ax.set_facecolor("#111c30")

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    ax.plot(fpr, tpr, color="#00e5a3", linewidth=2.5,
            label=f"{model_name} (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", linewidth=1.5,
            label="Random Guessing Baseline (AUC = 0.500)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_title(f"ROC Curve - {model_name}", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", color="#94a3b8", fontsize=10)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#94a3b8")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.3, color="#334155")

    plt.tight_layout()
    output_path = config.PLOTS_DIR / "roc_curve.png"
    fig.savefig(str(output_path), facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return "/static/plots/roc_curve.png"


def plot_metrics_bar(metrics: Dict[str, float], model_name: str) -> str:
    """Renders a modern bar chart comparing the primary performance metrics."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    fig.patch.set_facecolor("#0e1726")
    ax.set_facecolor("#111c30")

    keys = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "ROC-AUC"]
    values = [
        metrics.get("accuracy", 0.0),
        metrics.get("precision", 0.0),
        metrics.get("recall", 0.0),
        metrics.get("specificity", 0.0),
        metrics.get("f1_score", 0.0),
        metrics.get("roc_auc", 0.0)
    ]
    colors = ["#00d2ff", "#00e5a3", "#a29bfe", "#fdcb6e", "#ff7675", "#6c5ce7"]

    bars = ax.bar(keys, values, color=colors, width=0.55, edgecolor="#1e293b", linewidth=1.2)
    ax.set_ylim([0, 1.15])
    ax.set_title(f"Clinical Screening Evaluation Metrics - {model_name}", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(colors="#94a3b8", axis="x", rotation=15)
    ax.tick_params(colors="#94a3b8", axis="y")
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#334155")

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.02, f"{h:.2%}",
                ha="center", va="bottom", color="#f8fafc", fontsize=9, fontweight="bold")

    plt.tight_layout()
    output_path = config.PLOTS_DIR / "metrics_bar.png"
    fig.savefig(str(output_path), facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return "/static/plots/metrics_bar.png"


def evaluate_model(model_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Executes model evaluation against the test dataset:
    - Loads test images and true binary labels.
    - Runs inference through the trained model.
    - Computes real metrics: Accuracy, Precision, Recall, Specificity, F1, ROC-AUC.
    - Plots confusion matrix, ROC curve, and metric summary.
    - Saves evaluation results and updates model comparison ledger.
    """
    target_path = Path(model_path) if model_path else config.BEST_MODEL_PATH
    if not target_path.exists():
        raise FileNotFoundError(
            f"Model not found at {target_path}. "
            "Please train the model first by running: python training/train.py"
        )

    print(f"\n=======================================================")
    print(f"Loading Model for Evaluation: {target_path.name}")
    print(f"=======================================================")

    model = tf.keras.models.load_model(str(target_path))

    # Retrieve metadata if available
    metadata = load_json(config.MODEL_METADATA_PATH, {})
    model_name = metadata.get("model_name", "EfficientNetB0")

    # Load unseen test dataset
    test_paths, y_true = load_test_dataset()
    print(f"Evaluating on {len(test_paths)} unseen test images (TB: {sum(y_true)}, Normal: {len(y_true) - sum(y_true)})...")

    # Preprocess test images and predict
    test_images = []
    for p in test_paths:
        img_arr, _ = preprocess_image(p, enhance_contrast=False)
        test_images.append(img_arr)

    X_test = np.array(test_images, dtype=np.float32)
    y_prob_raw = model.predict(X_test, batch_size=config.DEFAULT_BATCH_SIZE, verbose=1)

    # Flatten probability array
    y_prob = [float(p[0]) for p in y_prob_raw]
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]

    # Calculate actual statistical metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5

    # Compute Confusion Matrix (TN, FP, FN, TP)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "total_test_samples": len(test_paths),
        "model_name": model_name,
        "model_file": target_path.name
    }

    # Generate visual artifacts
    cm_plot_url = plot_confusion_matrix(cm, model_name)
    roc_plot_url = plot_roc_curve(y_true, y_prob, auc, model_name)
    bar_plot_url = plot_metrics_bar(metrics, model_name)

    metrics["confusion_matrix_plot"] = cm_plot_url
    metrics["roc_curve_plot"] = roc_plot_url
    metrics["metrics_bar_plot"] = bar_plot_url

    # Save to evaluation_results.json
    save_json(metrics, config.EVALUATION_RESULTS_PATH)

    # Update model comparison ledger
    comparison_ledger = load_json(config.MODEL_COMPARISON_PATH, [])
    # Remove existing entry for this model if present
    comparison_ledger = [item for item in comparison_ledger if item.get("model") != model_name]
    comparison_ledger.append({
        "model": model_name,
        "accuracy": f"{acc * 100:.1f}%",
        "precision": f"{prec * 100:.1f}%",
        "recall": f"{rec * 100:.1f}%",
        "specificity": f"{specificity * 100:.1f}%",
        "f1": f"{f1 * 100:.1f}%",
        "auc": f"{auc:.3f}",
        "raw_accuracy": acc,
        "raw_auc": auc,
        "test_samples": len(test_paths)
    })
    save_json(comparison_ledger, config.MODEL_COMPARISON_PATH)

    print("\nEvaluation Completed Successfully!")
    print(f"  Accuracy    : {acc:.2%}")
    print(f"  Precision   : {prec:.2%}")
    print(f"  Recall/Sens : {rec:.2%}")
    print(f"  Specificity : {specificity:.2%}")
    print(f"  F1-Score    : {f1:.2%}")
    print(f"  ROC-AUC     : {auc:.3f}")
    print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TB Screening Deep Learning Model")
    parser.add_argument("--model", type=str, default=str(config.BEST_MODEL_PATH), help="Path to trained .keras model file")
    args = parser.parse_args()

    evaluate_model(Path(args.model))
