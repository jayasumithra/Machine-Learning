"""
=============================================================================
AI-Based Tuberculosis (TB) Screening from Chest X-Ray
Deep Feature Extraction & Dimensionality Reduction (PCA)
=============================================================================
Leverages the convolutional backbone (e.g. EfficientNetB0) without top
classification layers to extract high-dimensional latent visual representations.
Generates 2D PCA feature space separation plots and numeric vector previews.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from preprocessing.image_preprocessor import preprocess_image
from utils.helpers import scan_dataset

_FEATURE_EXTRACTOR_CACHE = {}

def get_feature_extractor(model_path: Optional[Path] = None, model_name: str = "EfficientNetB0") -> tf.keras.Model:
    """
    Returns a feature extractor model that produces a 1D embedding vector.
    If a trained model exists at model_path, uses its feature representation.
    Otherwise loads the pre-trained EfficientNetB0 backbone.
    """
    cache_key = str(model_path) if model_path and model_path.exists() else model_name
    if cache_key in _FEATURE_EXTRACTOR_CACHE:
        return _FEATURE_EXTRACTOR_CACHE[cache_key]

    if model_path and model_path.exists():
        try:
            full_model = tf.keras.models.load_model(str(model_path))
            # Find the global average pooling layer or dense feature layer
            gap_layer = None
            for layer in reversed(full_model.layers):
                if isinstance(layer, (tf.keras.layers.GlobalAveragePooling2D, tf.keras.layers.Dense)):
                    if "prediction" not in layer.name and "output" not in layer.name:
                        gap_layer = layer
                        break

            if gap_layer is not None:
                extractor = tf.keras.Model(inputs=full_model.inputs, outputs=gap_layer.output)
                _FEATURE_EXTRACTOR_CACHE[cache_key] = extractor
                return extractor
        except Exception as e:
            print(f"Warning: Could not build extractor from trained model: {e}")

    # Fallback / default: EfficientNetB0 backbone with GlobalAveragePooling2D
    base = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=config.INPUT_SHAPE
    )
    inputs = tf.keras.Input(shape=config.INPUT_SHAPE)
    x = base(inputs, training=False)
    outputs = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    extractor = tf.keras.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_FeatureExtractor")
    _FEATURE_EXTRACTOR_CACHE[cache_key] = extractor
    return extractor


def extract_single_image_features(image_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extracts high-dimensional deep feature embeddings for a single chest X-ray image.
    Returns the vector dimension, summary statistics, and first 25 numeric elements.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at {path}")

    # Preprocess image
    norm_img, meta = preprocess_image(path)
    batch_img = np.expand_dims(norm_img, axis=0)

    # Extract features
    extractor = get_feature_extractor(config.BEST_MODEL_PATH)
    features = extractor.predict(batch_img, verbose=0)[0]  # Shape: (1280,)

    dim = int(features.shape[0])
    feature_slice = [
        {"index": i + 1, "value": round(float(val), 5)}
        for i, val in enumerate(features[:25])
    ]

    return {
        "filename": path.name,
        "feature_extractor": extractor.name,
        "feature_dim": dim,
        "features_preview": feature_slice,
        "min_value": round(float(np.min(features)), 5),
        "max_value": round(float(np.max(features)), 5),
        "mean_value": round(float(np.mean(features)), 5),
        "std_value": round(float(np.std(features)), 5),
        "l2_norm": round(float(np.linalg.norm(features)), 5)
    }


def generate_pca_feature_space_plot(max_samples_per_class: int = 30) -> Dict[str, Any]:
    """
    Extracts feature representations across dataset samples, performs 2D Principal
    Component Analysis (PCA), and plots the sample distributions to demonstrate
    clustering and separation between Normal and TB cases in feature space.
    """
    stats = scan_dataset()
    if stats["total_images"] < 4:
        return {
            "success": False,
            "message": "Insufficient images in dataset (minimum 4 required for PCA feature projection).",
            "plot_url": None
        }

    # Collect sample paths
    normal_files = sorted(list(config.NORMAL_DIR.glob("*.jpg")) + list(config.NORMAL_DIR.glob("*.png")))[:max_samples_per_class]
    tb_files = sorted(list(config.TB_DIR.glob("*.jpg")) + list(config.TB_DIR.glob("*.png")))[:max_samples_per_class]

    if len(normal_files) < 2 or len(tb_files) < 2:
        return {
            "success": False,
            "message": f"Both Normal (found {len(normal_files)}) and TB (found {len(tb_files)}) must have at least 2 samples.",
            "plot_url": None
        }

    paths = normal_files + tb_files
    labels = ["Normal"] * len(normal_files) + ["TB"] * len(tb_files)

    # Preprocess all samples
    images = []
    for p in paths:
        norm, _ = preprocess_image(p)
        images.append(norm)

    X = np.array(images, dtype=np.float32)
    extractor = get_feature_extractor(config.BEST_MODEL_PATH)
    embeddings = extractor.predict(X, batch_size=8, verbose=0)

    # Perform PCA down to 2 dimensions
    pca = PCA(n_components=2, random_state=config.RANDOM_SEED)
    coords_2d = pca.fit_transform(embeddings)

    var_pc1 = float(pca.explained_variance_ratio_[0]) * 100
    var_pc2 = float(pca.explained_variance_ratio_[1]) * 100

    # Render modern dark-themed scatter plot
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=140)
    fig.patch.set_facecolor("#0e1726")
    ax.set_facecolor("#111c30")

    # Plot Normal samples (cyan)
    idx_normal = [i for i, l in enumerate(labels) if l == "Normal"]
    ax.scatter(
        coords_2d[idx_normal, 0], coords_2d[idx_normal, 1],
        color="#00e5a3", label=f"Normal (n={len(idx_normal)})",
        s=80, alpha=0.85, edgecolors="#ffffff", linewidths=1.2
    )

    # Plot TB samples (coral/red)
    idx_tb = [i for i, l in enumerate(labels) if l == "TB"]
    ax.scatter(
        coords_2d[idx_tb, 0], coords_2d[idx_tb, 1],
        color="#ff7675", label=f"TB (n={len(idx_tb)})",
        s=80, alpha=0.85, edgecolors="#ffffff", linewidths=1.2, marker="^"
    )

    ax.set_title("Deep Feature Space Projection (PCA 2D)", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(f"Principal Component 1 ({var_pc1:.1f}% Variance)", color="#94a3b8", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Principal Component 2 ({var_pc2:.1f}% Variance)", color="#94a3b8", fontsize=10, fontweight="bold")
    ax.tick_params(colors="#94a3b8")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", loc="best")
    ax.grid(True, linestyle="--", alpha=0.3, color="#334155")

    plt.tight_layout()
    plot_file = config.PLOTS_DIR / "pca_features.png"
    fig.savefig(str(plot_file), facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    return {
        "success": True,
        "plot_url": "/static/plots/pca_features.png",
        "plot_path": str(plot_file),
        "total_samples": len(paths),
        "normal_samples": len(idx_normal),
        "tb_samples": len(idx_tb),
        "feature_dim": embeddings.shape[1],
        "pc1_variance": round(var_pc1, 2),
        "pc2_variance": round(var_pc2, 2),
        "total_explained_variance": round(var_pc1 + var_pc2, 2)
    }

if __name__ == "__main__":
    print("Testing Deep Feature Extraction...")
    test_img = next(config.NORMAL_DIR.glob("*.jpg"), None)
    if test_img:
        feats = extract_single_image_features(test_img)
        print(f"Extracted features for {test_img.name}:")
        print(f"  Dimension: {feats['feature_dim']}")
        print(f"  Preview (first 5): {feats['features_preview'][:5]}")

    print("Generating PCA 2D Feature Space Plot...")
    pca_res = generate_pca_feature_space_plot()
    print("PCA Result:", pca_res)
