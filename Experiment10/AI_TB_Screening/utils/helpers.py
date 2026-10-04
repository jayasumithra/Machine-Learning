"""
=============================================================================
AI-Based Tuberculosis (TB) Screening from Chest X-Ray
Utility and Helper Functions
=============================================================================
Provides hardware detection, dataset discovery, file validation,
and metadata persistence routines.
"""

import os
import sys
import json
import platform
from pathlib import Path
from typing import Dict, Any, List, Optional
from werkzeug.utils import secure_filename

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

def get_hardware_info() -> Dict[str, Any]:
    """
    Detects system hardware, available accelerator devices (CPU/GPU),
    and runtime environment details.
    """
    device_type = "CPU"
    device_name = "CPU"
    gpu_available = False
    details = []

    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            gpu_available = True
            device_type = "GPU"
            device_name = f"GPU ({len(gpus)} device(s) detected)"
            details = [g.name for g in gpus]
        else:
            cpus = tf.config.list_physical_devices("CPU")
            device_type = "CPU"
            device_name = "CPU (x86_64 / Intel / AMD)"
            details = [c.name for c in cpus]
        tf_version = tf.__version__
    except Exception as e:
        tf_version = f"Error: {e}"

    return {
        "device_type": device_type,
        "device_name": device_name,
        "gpu_available": gpu_available,
        "details": details,
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "tensorflow_version": tf_version
    }

def allowed_file(filename: str) -> bool:
    """Verifies if the file has an allowed extension."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in config.ALLOWED_EXTENSIONS

def format_file_size(size_bytes: int) -> str:
    """Converts raw byte count into a human-readable string (KB, MB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"

def scan_dataset() -> Dict[str, Any]:
    """
    Scans the dataset directory, counts valid images per class,
    and returns sample paths and distribution metrics.
    """
    stats = {
        "exists": False,
        "is_empty": True,
        "total_images": 0,
        "tb_count": 0,
        "normal_count": 0,
        "tb_percentage": 0.0,
        "normal_percentage": 0.0,
        "tb_samples": [],
        "normal_samples": [],
        "valid_formats": sorted(list(config.ALLOWED_EXTENSIONS)),
        "dataset_path": str(config.DATASET_DIR)
    }

    if not config.DATASET_DIR.exists():
        return stats

    stats["exists"] = True

    tb_files = []
    normal_files = []

    if config.TB_DIR.exists():
        for file_path in config.TB_DIR.iterdir():
            if file_path.is_file() and allowed_file(file_path.name):
                tb_files.append(file_path.name)

    if config.NORMAL_DIR.exists():
        for file_path in config.NORMAL_DIR.iterdir():
            if file_path.is_file() and allowed_file(file_path.name):
                normal_files.append(file_path.name)

    total = len(tb_files) + len(normal_files)
    stats["total_images"] = total
    stats["tb_count"] = len(tb_files)
    stats["normal_count"] = len(normal_files)
    stats["is_empty"] = (total == 0)

    if total > 0:
        stats["tb_percentage"] = round((len(tb_files) / total) * 100, 1)
        stats["normal_percentage"] = round((len(normal_files) / total) * 100, 1)

    # Provide up to 6 sample filenames for gallery preview
    stats["tb_samples"] = sorted(tb_files)[:6]
    stats["normal_samples"] = sorted(normal_files)[:6]

    return stats

def get_model_status() -> Dict[str, Any]:
    """Checks for the presence and readiness of trained models and evaluation reports."""
    model_exists = config.BEST_MODEL_PATH.exists()
    status = {
        "model_available": model_exists,
        "best_model_path": str(config.BEST_MODEL_PATH) if model_exists else None,
        "model_file_size": format_file_size(config.BEST_MODEL_PATH.stat().st_size) if model_exists else "0 B",
        "metadata": None,
        "evaluation": None,
        "comparison": []
    }

    if config.MODEL_METADATA_PATH.exists():
        try:
            with open(config.MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
                status["metadata"] = json.load(f)
        except Exception:
            status["metadata"] = None

    if config.EVALUATION_RESULTS_PATH.exists():
        try:
            with open(config.EVALUATION_RESULTS_PATH, "r", encoding="utf-8") as f:
                status["evaluation"] = json.load(f)
        except Exception:
            status["evaluation"] = None

    if config.MODEL_COMPARISON_PATH.exists():
        try:
            with open(config.MODEL_COMPARISON_PATH, "r", encoding="utf-8") as f:
                status["comparison"] = json.load(f)
        except Exception:
            status["comparison"] = []

    return status

def save_json(data: Any, file_path: Path) -> None:
    """Safely writes a python object to a formatted JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def load_json(file_path: Path, default: Any = None) -> Any:
    """Safely loads a JSON file or returns the default value if absent."""
    if not file_path.exists():
        return default
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default
