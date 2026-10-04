"""
=============================================================================
AI-Based Tuberculosis (TB) Screening from Chest X-Ray Using Deep Learning
Configuration Module
=============================================================================
This file centralizes all paths, hyperparameters, model options, and
system constants. Paths are relative to the project root, making the
project portable across different Windows, Linux, and macOS environments.
"""

import os
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Dataset paths (configurable, not hard-coded to a specific machine)
DATASET_DIR = BASE_DIR / "dataset"
TB_DIR = DATASET_DIR / "TB"
NORMAL_DIR = DATASET_DIR / "Normal"

# Models and artifacts storage
MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"
MODEL_COMPARISON_PATH = MODELS_DIR / "model_comparison.json"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
EVALUATION_RESULTS_PATH = MODELS_DIR / "evaluation_results.json"
TRAINING_STATUS_PATH = MODELS_DIR / "training_status.json"

# Database path
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "tb_screening.db"

# Static directories for uploads, visual results, and plots
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
RESULTS_DIR = STATIC_DIR / "results"
PLOTS_DIR = STATIC_DIR / "plots"

# Ensure runtime directories exist
for directory in [DATASET_DIR, TB_DIR, NORMAL_DIR, MODELS_DIR, DATABASE_DIR,
                  STATIC_DIR, UPLOADS_DIR, RESULTS_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Image preprocessing specifications
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload limit

# Class labeling
# Label 0: Normal, Label 1: TB (Positive screening target)
CLASSES = ["Normal", "TB"]
CLASS_LABELS = {0: "Normal", 1: "TB"}

# Training Hyperparameters and Defaults
DEFAULT_MODEL = "EfficientNetB0"
SUPPORTED_MODELS = ["EfficientNetB0", "ResNet50", "CNN_Baseline"]
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_VAL_SPLIT = 0.15
DEFAULT_TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Early Stopping & Checkpoint settings
PATIENCE_EARLY_STOP = 4
PATIENCE_REDUCE_LR = 2

# Medical Safety Disclaimer
MEDICAL_DISCLAIMER = (
    "This application is an academic/research prototype for AI-assisted TB screening from chest X-ray images. "
    "It is not a certified medical diagnostic device and must not be used as a substitute for professional medical evaluation."
)

# Flask Server Config
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000
FLASK_DEBUG = True
SECRET_KEY = os.environ.get("SECRET_KEY", "academic_tb_screening_secret_key_2026")
