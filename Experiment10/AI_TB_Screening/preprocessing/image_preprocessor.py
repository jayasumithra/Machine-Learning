"""
=============================================================================
AI-Based Tuberculosis (TB) Screening from Chest X-Ray
Image Preprocessing Pipeline
=============================================================================
Handles image loading, format validation, color space conversion (RGB),
bilateral resizing (224x224), contrast enhancement (CLAHE), pixel normalization,
and multi-step educational preprocessing visualization.
"""

import sys
import os
from pathlib import Path
from typing import Tuple, Dict, Any, Union, Optional
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

def validate_image_file(file_path: Union[str, Path]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Validates that a file exists, has an allowed extension, does not exceed
    the size threshold, and can be successfully decoded as a valid image.
    """
    path = Path(file_path)
    if not path.exists():
        return False, f"File not found: {path.name}", None

    # Check file extension
    ext = path.suffix.lower().replace(".", "")
    if ext not in config.ALLOWED_EXTENSIONS:
        return False, f"Unsupported format (.{ext}). Allowed formats: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}", None

    # Check file size
    size_bytes = path.stat().st_size
    if size_bytes == 0:
        return False, "File is empty (0 bytes)", None
    if size_bytes > config.MAX_CONTENT_LENGTH:
        return False, f"File exceeds maximum allowed size ({config.MAX_CONTENT_LENGTH // (1024*1024)} MB)", None

    # Verify decodability with OpenCV / PIL
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as e:
        return False, f"Corrupted or invalid image file: {str(e)}", None

    # Read dimensions safely
    try:
        cv_img = cv2.imread(str(path))
        if cv_img is None:
            return False, "Could not decode image pixel matrix", None
        h, w, c = cv_img.shape
    except Exception as e:
        return False, f"Error reading image dimensions: {str(e)}", None

    info = {
        "filename": path.name,
        "width": w,
        "height": h,
        "channels": c,
        "size_bytes": size_bytes,
        "format": ext.upper()
    }
    return True, "Valid image", info

def apply_clahe(img_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to enhance lung parenchymal visibility in chest radiographs.
    """
    # Convert BGR to LAB color space
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)

    # Merge channels and convert back to BGR
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def preprocess_image(
    image_input: Union[str, Path, np.ndarray],
    target_size: Tuple[int, int] = config.IMAGE_SIZE,
    enhance_contrast: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Standard preprocessing workflow for chest X-ray deep learning input:
    1. Read Image
    2. Convert to RGB
    3. Resize to target dimension (default 224x224)
    4. Optional CLAHE contrast enhancement
    5. Normalize pixel values to [0.0, 1.0]

    Returns:
        - normalized_array: np.ndarray of shape (target_size[0], target_size[1], 3), float32 in [0, 1]
        - metadata: Dictionary containing actual computed image properties
    """
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        raw_bgr = cv2.imread(str(img_path))
        if raw_bgr is None:
            raise ValueError(f"Unable to read image at: {img_path}")
        original_h, original_w = raw_bgr.shape[:2]
        original_c = raw_bgr.shape[2] if len(raw_bgr.shape) > 2 else 1
    elif isinstance(image_input, np.ndarray):
        raw_bgr = image_input.copy()
        original_h, original_w = raw_bgr.shape[:2]
        original_c = raw_bgr.shape[2] if len(raw_bgr.shape) > 2 else 1
    else:
        raise TypeError("image_input must be a file path or numpy array")

    # Apply CLAHE if requested
    if enhance_contrast:
        processed_bgr = apply_clahe(raw_bgr)
    else:
        processed_bgr = raw_bgr

    # Convert to RGB color space
    if len(processed_bgr.shape) == 2 or original_c == 1:
        rgb_img = cv2.cvtColor(processed_bgr, cv2.COLOR_GRAY2RGB)
        color_conv = "Grayscale to RGB (3 Channels)"
    else:
        rgb_img = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
        color_conv = "BGR to RGB (3 Channels)"

    # Resize image with area/linear interpolation
    resized_rgb = cv2.resize(rgb_img, target_size, interpolation=cv2.INTER_AREA)

    # Compute actual pixel statistics before normalization
    min_val_raw = float(np.min(resized_rgb))
    max_val_raw = float(np.max(resized_rgb))
    mean_val_raw = float(np.mean(resized_rgb))
    std_val_raw = float(np.std(resized_rgb))

    # Normalize to [0.0, 1.0] float32
    normalized_array = resized_rgb.astype(np.float32) / 255.0

    metadata = {
        "original_size": f"{original_w} × {original_h}",
        "original_width": original_w,
        "original_height": original_h,
        "original_channels": original_c,
        "processed_size": f"{target_size[0]} × {target_size[1]} × 3",
        "processed_shape": list(normalized_array.shape),
        "raw_min_pixel": round(min_val_raw, 2),
        "raw_max_pixel": round(max_val_raw, 2),
        "raw_mean_pixel": round(mean_val_raw, 2),
        "raw_std_pixel": round(std_val_raw, 2),
        "normalized_min": round(float(np.min(normalized_array)), 4),
        "normalized_max": round(float(np.max(normalized_array)), 4),
        "normalization_range": "[0.0, 1.0]",
        "color_conversion": color_conv,
        "contrast_enhanced": enhance_contrast,
        "target_dtype": "float32"
    }

    return normalized_array, metadata

def generate_preprocessing_pipeline_visualization(
    image_path: Union[str, Path],
    output_filename: str = "preprocessing_comparison.png"
) -> Dict[str, Any]:
    """
    Renders an educational 5-stage side-by-side visualization of the
    actual preprocessing pipeline and saves it to static/results.
    Stages:
    1. Original Chest X-Ray
    2. RGB Conversion
    3. Resized (224x224)
    4. CLAHE Enhanced
    5. Data Augmentation (Simulated Training Variation)
    """
    path = Path(image_path)
    raw_bgr = cv2.imread(str(path))
    if raw_bgr is None:
        raise ValueError(f"Cannot load image: {path}")

    # 1. Original RGB
    orig_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

    # 2. Resized (224x224)
    resized_rgb = cv2.resize(orig_rgb, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    # 3. CLAHE Enhanced
    clahe_bgr = apply_clahe(raw_bgr)
    clahe_rgb = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB)
    clahe_resized = cv2.resize(clahe_rgb, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    # 4. Normalized float representation (scaled back to 0-255 for visualization)
    norm_arr, meta = preprocess_image(path, enhance_contrast=False)
    norm_vis = (norm_arr * 255.0).astype(np.uint8)

    # 5. Data Augmentation example (rotation + zoom + slight brightness)
    h, w = resized_rgb.shape[:2]
    m_rot = cv2.getRotationMatrix2D((w // 2, h // 2), angle=-8, scale=1.05)
    augmented = cv2.warpAffine(resized_rgb, m_rot, (w, h), borderMode=cv2.BORDER_REFLECT)
    augmented = cv2.convertScaleAbs(augmented, alpha=1.1, beta=10)

    # Plot figure with matplotlib
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), dpi=130)
    fig.patch.set_facecolor("#0e1726")

    stages = [
        ("1. Original X-Ray", orig_rgb, f"{orig_rgb.shape[1]}×{orig_rgb.shape[0]}"),
        ("2. Resized RGB", resized_rgb, "224×224×3"),
        ("3. CLAHE Enhanced", clahe_resized, "High Contrast"),
        ("4. Normalized [0, 1]", norm_vis, "Float32 Input"),
        ("5. Augmented (Train Only)", augmented, "Rot: -8°, Zoom: 1.05")
    ]

    for ax, (title, img_data, sub) in zip(axes, stages):
        ax.imshow(img_data)
        ax.set_title(title, color="#00e5a3", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel(sub, color="#94a3b8", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#1e293b")
            spine.set_linewidth(1.5)

    plt.tight_layout()
    output_path = config.RESULTS_DIR / output_filename
    fig.savefig(str(output_path), facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    meta["visualization_file"] = output_filename
    meta["visualization_url"] = f"/static/results/{output_filename}"
    return meta

if __name__ == "__main__":
    # Test with first normal image
    test_img = next(config.NORMAL_DIR.glob("*.jpg"), None)
    if test_img:
        print(f"Testing preprocessing pipeline on: {test_img.name}")
        norm, info = preprocess_image(test_img)
        print("Metadata:", info)
        res = generate_preprocessing_pipeline_visualization(test_img)
        print("Visualization generated:", res["visualization_url"])
    else:
        print("No images found in dataset. Run sample_data_generator.py first.")
