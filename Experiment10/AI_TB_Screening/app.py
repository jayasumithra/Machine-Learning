"""
=============================================================================
AI-Based Tuberculosis (TB) Screening from Chest X-Ray Using Deep Learning
Flask Web Application Server
=============================================================================
Academic/research prototype only.
Not a medical diagnostic device.

Important:
- Normal = 0
- TB     = 1
- EfficientNetB0 training input = approximately 0-255
- ResNet50 training input = approximately 0-255, followed by its internal
  ResNet preprocessing layer
- CNN Baseline training input = approximately 0-255
=============================================================================
"""

import sys
import os
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    flash,
    send_from_directory,
)

from werkzeug.utils import secure_filename

import numpy as np
import tensorflow as tf


# =============================================================================
# PROJECT PATH
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# PROJECT IMPORTS
# =============================================================================

import config

from utils.helpers import (
    get_hardware_info,
    scan_dataset,
    get_model_status,
    allowed_file,
    load_json,
)

from utils.sample_data_generator import generate_starter_dataset

from database.database import (
    init_db,
    save_prediction,
    get_history,
    delete_prediction,
    clear_history,
    get_history_stats,
)

from preprocessing.image_preprocessor import (
    validate_image_file,
    preprocess_image,
    generate_preprocessing_pipeline_visualization,
)

# IMPORTANT:
# training.py contains train_model(), not train_tb_model()
from training.train import train_model

from training.evaluate import evaluate_model

from training.feature_extraction import (
    extract_single_image_features,
    generate_pca_feature_space_plot,
)

from explainability.gradcam import generate_gradcam_artifacts


# =============================================================================
# FLASK APPLICATION
# =============================================================================

app = Flask(__name__)

app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = str(config.UPLOADS_DIR)


# =============================================================================
# DATABASE
# =============================================================================

init_db()


# =============================================================================
# MODEL CACHE
# =============================================================================

_MODEL_CACHE: Dict[str, tf.keras.Model] = {}

_MODEL_CACHE_LOCK = threading.Lock()


# =============================================================================
# MODEL PATH HELPER
# =============================================================================

def get_model_path(model_name: Optional[str] = None) -> tuple:
    """
    Returns the exact model path and cache key.

    IMPORTANT:
    We do NOT silently replace a requested model with best_model.keras.
    """

    selected_name = model_name or config.DEFAULT_MODEL

    if selected_name == "default":
        selected_name = config.DEFAULT_MODEL

    if selected_name not in config.SUPPORTED_MODELS:
        selected_name = config.DEFAULT_MODEL

    model_path = (
        config.MODELS_DIR
        / f"{selected_name}_model.keras"
    )

    cache_key = selected_name

    return model_path, cache_key


# =============================================================================
# MODEL LOADING
# =============================================================================

def get_loaded_model(
    model_name: Optional[str] = None
) -> Optional[tf.keras.Model]:
    """
    Loads and caches the exact selected model.

    Example:
        EfficientNetB0
        -> models/EfficientNetB0_model.keras

        ResNet50
        -> models/ResNet50_model.keras

        CNN_Baseline
        -> models/CNN_Baseline_model.keras
    """

    model_path, cache_key = get_model_path(model_name)

    with _MODEL_CACHE_LOCK:

        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        if not model_path.exists():

            print()
            print("=" * 70)
            print("MODEL FILE NOT FOUND")
            print("=" * 70)
            print("Requested model :", cache_key)
            print("Expected file   :", model_path)
            print("=" * 70)
            print()

            return None

        try:

            print()
            print("=" * 70)
            print("LOADING MODEL")
            print("=" * 70)
            print("Model name :", cache_key)
            print("Model path :", model_path)
            print("=" * 70)

            loaded_model = tf.keras.models.load_model(
                str(model_path)
            )

            print("Model loaded successfully.")
            print("Input shape :", loaded_model.input_shape)
            print("Output shape:", loaded_model.output_shape)
            print()

            _MODEL_CACHE[cache_key] = loaded_model

            return loaded_model

        except Exception as error:

            print()
            print("=" * 70)
            print("MODEL LOADING ERROR")
            print("=" * 70)
            print(error)
            print("=" * 70)
            print()

            return None


# =============================================================================
# MODEL INPUT PREPARATION
# =============================================================================

def prepare_model_input(
    image_path: Path,
    model_name: str
):
    """
    Applies the SAME base preprocessing used by training.

    preprocess_image() returns:
        float32 values in [0, 1]

    training/train.py converts these values to:
        float32 values in [0, 255]

    Therefore prediction must also use:
        [0, 1] -> [0, 255]

    The ResNet50 model contains its own ResNet preprocessing layer,
    so we still provide the same 0-255 range here.
    """

    norm_img, preproc_meta = preprocess_image(
        image_path,
        enhance_contrast=False
    )

    # Match training/train.py exactly.
    model_input = (
        norm_img.astype(np.float32) * 255.0
    )

    batch = np.expand_dims(
        model_input,
        axis=0
    )

    return batch, preproc_meta


# =============================================================================
# GLOBAL TEMPLATE VARIABLES
# =============================================================================

@app.context_processor
def inject_global_template_vars():

    hw = get_hardware_info()

    model_st = get_model_status()

    db_st = get_history_stats()

    ds_st = scan_dataset()

    return {
        "hardware_info": hw,
        "model_status": model_st,
        "db_stats": db_st,
        "dataset_stats": ds_st,
        "disclaimer": config.MEDICAL_DISCLAIMER,
        "supported_models": config.SUPPORTED_MODELS,
        "app_title": "AI TB Screening",
    }


# =============================================================================
# DATASET IMAGE SERVER
# =============================================================================

@app.route("/dataset-images/<path:filename>")
def serve_dataset_image(filename):

    return send_from_directory(
        str(config.DATASET_DIR),
        filename
    )


# =============================================================================
# DASHBOARD
# =============================================================================

@app.route("/")
def index():

    model_st = get_model_status()

    ds_st = scan_dataset()

    db_st = get_history_stats()

    recent_history = get_history(
        limit=5
    )

    return render_template(
        "index.html",
        dataset=ds_st,
        model_status=model_st,
        history=recent_history,
        db_stats=db_st,
    )


# =============================================================================
# DATASET PAGE
# =============================================================================

@app.route("/dataset")
def dataset_page():

    ds_st = scan_dataset()

    return render_template(
        "dataset.html",
        dataset=ds_st
    )


# =============================================================================
# PREPROCESSING PAGE
# =============================================================================

@app.route(
    "/preprocessing",
    methods=["GET", "POST"]
)
def preprocessing_page():

    comparison_result = None

    selected_image = None

    ds_st = scan_dataset()

    if request.method == "POST":

        if (
            "file" in request.files
            and request.files["file"].filename != ""
        ):

            file = request.files["file"]

            if allowed_file(file.filename):

                fname = (
                    f"preproc_{int(time.time())}_"
                    f"{secure_filename(file.filename)}"
                )

                save_path = (
                    config.UPLOADS_DIR / fname
                )

                file.save(
                    str(save_path)
                )

                selected_image = str(
                    save_path
                )

            else:

                flash(
                    "Unsupported format. "
                    f"Allowed formats: "
                    f"{', '.join(config.ALLOWED_EXTENSIONS)}",
                    "danger",
                )

        elif "sample_image" in request.form:

            sample_rel = request.form[
                "sample_image"
            ]

            potential_path = (
                config.DATASET_DIR
                / sample_rel
            )

            if potential_path.exists():

                selected_image = str(
                    potential_path
                )

    if not selected_image:

        all_samples = (
            list(config.TB_DIR.glob("*.jpg"))
            + list(config.NORMAL_DIR.glob("*.jpg"))
        )

        if all_samples:

            selected_image = str(
                all_samples[0]
            )

    if (
        selected_image
        and Path(selected_image).exists()
    ):

        try:

            comparison_result = (
                generate_preprocessing_pipeline_visualization(
                    selected_image,
                    output_filename="preprocessing_comparison.png",
                )
            )

        except Exception as error:

            flash(
                f"Error during preprocessing visualization: {error}",
                "danger",
            )

    return render_template(
        "preprocessing.html",
        comparison=comparison_result,
        selected_image=(
            Path(selected_image).name
            if selected_image
            else None
        ),
        dataset=ds_st,
    )


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

@app.route("/features")
def features_page():

    sample_img = (
        next(
            config.TB_DIR.glob("*.jpg"),
            None
        )
        or
        next(
            config.NORMAL_DIR.glob("*.jpg"),
            None
        )
    )

    features_info = None

    if sample_img:

        try:

            features_info = (
                extract_single_image_features(
                    sample_img
                )
            )

        except Exception as error:

            print(
                "Feature extraction preview error:",
                error
            )

    pca_plot_path = (
        config.PLOTS_DIR
        / "pca_features.png"
    )

    pca_available = pca_plot_path.exists()

    return render_template(
        "features.html",
        features=features_info,
        pca_available=pca_available,
        pca_url=(
            "/static/plots/pca_features.png"
            if pca_available
            else None
        ),
    )


# =============================================================================
# TRAINING PAGE
# =============================================================================

@app.route("/training")
def training_page():

    ds_st = scan_dataset()

    model_st = get_model_status()

    training_status = load_json(
        config.TRAINING_STATUS_PATH,
        {
            "is_training": False,
            "status": "idle",
        },
    )

    return render_template(
        "training.html",
        dataset=ds_st,
        model_status=model_st,
        training_status=training_status,
    )


# =============================================================================
# EVALUATION PAGE
# =============================================================================

@app.route("/evaluation")
def evaluation_page():

    model_st = get_model_status()

    eval_results = load_json(
        config.EVALUATION_RESULTS_PATH,
        None
    )

    return render_template(
        "evaluation.html",
        model_status=model_st,
        evaluation=eval_results,
    )


# =============================================================================
# MODEL COMPARISON PAGE
# =============================================================================

@app.route("/comparison")
def comparison_page():

    comparison_data = load_json(
        config.MODEL_COMPARISON_PATH,
        []
    )

    return render_template(
        "comparison.html",
        comparison=comparison_data
    )


# =============================================================================
# PREDICTION PAGE
# =============================================================================

@app.route(
    "/prediction",
    methods=["GET", "POST"]
)
def prediction_page():

    prediction_result = None

    if request.method == "POST":

        # ---------------------------------------------------------
        # CHECK FILE
        # ---------------------------------------------------------

        if (
            "file" not in request.files
            or request.files["file"].filename == ""
        ):

            flash(
                "Please select a chest X-ray image file to upload.",
                "warning",
            )

            return redirect(
                request.url
            )

        file = request.files["file"]

        # ---------------------------------------------------------
        # CHECK EXTENSION
        # ---------------------------------------------------------

        if not allowed_file(
            file.filename
        ):

            flash(
                "Invalid file extension. "
                f"Please upload a "
                f"{', '.join(config.ALLOWED_EXTENSIONS).upper()} file.",
                "danger",
            )

            return redirect(
                request.url
            )

        # ---------------------------------------------------------
        # SAVE UPLOADED IMAGE
        # ---------------------------------------------------------

        unique_name = (
            f"xray_{int(time.time())}_"
            f"{secure_filename(file.filename)}"
        )

        save_path = (
            config.UPLOADS_DIR
            / unique_name
        )

        file.save(
            str(save_path)
        )

        # ---------------------------------------------------------
        # VALIDATE IMAGE
        # ---------------------------------------------------------

        is_valid, msg, img_info = (
            validate_image_file(
                save_path
            )
        )

        if not is_valid:

            save_path.unlink(
                missing_ok=True
            )

            flash(
                f"Image validation failed: {msg}",
                "danger",
            )

            return redirect(
                request.url
            )

        # ---------------------------------------------------------
        # SELECT MODEL
        # ---------------------------------------------------------

        selected_model_name = request.form.get(
            "model_name",
            config.DEFAULT_MODEL
        )

        if selected_model_name == "default":

            selected_model_name = (
                config.DEFAULT_MODEL
            )

        if selected_model_name not in config.SUPPORTED_MODELS:

            flash(
                f"Unsupported model: {selected_model_name}",
                "danger",
            )

            return redirect(
                request.url
            )

        # ---------------------------------------------------------
        # LOAD MODEL
        # ---------------------------------------------------------

        model = get_loaded_model(
            selected_model_name
        )

        if model is None:

            flash(
                f"No trained {selected_model_name} model found. "
                f"Please train this model first.",
                "warning",
            )

            return redirect(
                url_for("training_page")
            )

        # ---------------------------------------------------------
        # PREDICTION
        # ---------------------------------------------------------

        start_t = time.time()

        try:

            # IMPORTANT:
            # This now matches training:
            #
            # preprocess -> [0,1]
            # multiply by 255 -> [0,255]
            # model.predict()

            batch, preproc_meta = (
                prepare_model_input(
                    save_path,
                    selected_model_name
                )
            )

            print()
            print("=" * 70)
            print("TB PREDICTION")
            print("=" * 70)
            print("Selected model:", selected_model_name)
            print("Input shape   :", batch.shape)
            print(
                "Input minimum :",
                float(batch.min())
            )
            print(
                "Input maximum :",
                float(batch.max())
            )

            # -----------------------------------------------------
            # MODEL PREDICTION
            # -----------------------------------------------------

            prediction_output = model.predict(
                batch,
                verbose=0
            )

            prob_raw = float(
                prediction_output[0][0]
            )

            # Keep probability safely in [0,1]
            prob_raw = float(
                np.clip(
                    prob_raw,
                    0.0,
                    1.0
                )
            )

            elapsed_ms = round(
                (time.time() - start_t) * 1000,
                1
            )

            # -----------------------------------------------------
            # PROBABILITIES
            # -----------------------------------------------------

            tb_prob = round(
                prob_raw * 100,
                2
            )

            normal_prob = round(
                (1.0 - prob_raw) * 100,
                2
            )

            # -----------------------------------------------------
            # CLASSIFICATION
            #
            # Normal = 0
            # TB     = 1
            # -----------------------------------------------------

            predicted_class = (
                "TB"
                if prob_raw >= 0.5
                else "Normal"
            )

            confidence = (
                tb_prob
                if predicted_class == "TB"
                else normal_prob
            )

            print(
                "TB probability     :",
                f"{tb_prob:.2f}%"
            )

            print(
                "Normal probability :",
                f"{normal_prob:.2f}%"
            )

            print(
                "Prediction         :",
                predicted_class
            )

            print(
                "Inference time     :",
                f"{elapsed_ms} ms"
            )

            print("=" * 70)
            print()

            # -----------------------------------------------------
            # GRAD-CAM
            # -----------------------------------------------------

            gradcam_result = None

            overlay_file = ""

            try:

                gradcam_result = (
                    generate_gradcam_artifacts(
                        save_path,
                        model=model
                    )
                )

                if gradcam_result:

                    overlay_file = (
                        gradcam_result.get(
                            "overlay_filename",
                            ""
                        )
                    )

            except Exception as error:

                print(
                    "Grad-CAM generation warning:",
                    error
                )

            # -----------------------------------------------------
            # SAVE HISTORY
            # -----------------------------------------------------

            record_id = save_prediction(

                image_filename=unique_name,

                model_name=selected_model_name,

                prediction=predicted_class,

                tb_probability=tb_prob,

                normal_probability=normal_prob,

                gradcam_image=overlay_file,

                execution_time_ms=elapsed_ms,
            )

            # -----------------------------------------------------
            # RESULT
            # -----------------------------------------------------

            prediction_result = {

                "id": record_id,

                "filename": unique_name,

                "image_url":
                    f"/static/uploads/{unique_name}",

                "model_name":
                    selected_model_name,

                "prediction":
                    predicted_class,

                "confidence":
                    confidence,

                "tb_probability":
                    tb_prob,

                "normal_probability":
                    normal_prob,

                "execution_time_ms":
                    elapsed_ms,

                "image_info":
                    img_info,

                "preprocessing_info":
                    preproc_meta,

                "gradcam":
                    gradcam_result,
            }

        except Exception as error:

            print()
            print(
                "PREDICTION ERROR:",
                error
            )
            print()

            flash(
                f"Prediction failed: {error}",
                "danger"
            )

    return render_template(
        "prediction.html",
        result=prediction_result
    )


# =============================================================================
# EXPLAINABILITY
# =============================================================================

@app.route("/explainability")
def explainability_page():

    results = sorted(
        list(
            config.RESULTS_DIR.glob(
                "gradcam_overlay_*.jpg"
            )
        ),
        key=os.path.getmtime,
        reverse=True,
    )

    gradcam_info = None

    if results:

        over_file = results[0].name

        stem_match = (
            over_file
            .replace(
                "gradcam_overlay_",
                ""
            )
            .replace(
                ".jpg",
                ""
            )
        )

        orig_file = (
            f"gradcam_orig_{stem_match}.jpg"
        )

        heat_file = (
            f"gradcam_heat_{stem_match}.jpg"
        )

        gradcam_info = {

            "overlay_image_url":
                f"/static/results/{over_file}",

            "original_image_url":
                (
                    f"/static/results/{orig_file}"
                    if (
                        config.RESULTS_DIR
                        / orig_file
                    ).exists()
                    else
                    f"/static/results/{over_file}"
                ),

            "heatmap_image_url":
                (
                    f"/static/results/{heat_file}"
                    if (
                        config.RESULTS_DIR
                        / heat_file
                    ).exists()
                    else
                    f"/static/results/{over_file}"
                ),

            "target_conv_layer":
                "Model-dependent convolutional layer",

            "explanation":
                (
                    "Grad-CAM highlights image regions "
                    "that contributed to the neural network "
                    "prediction. This visualization is "
                    "for research/educational interpretation "
                    "and is not a clinical diagnostic explanation."
                ),
        }

    else:

        test_img = next(
            config.TB_DIR.glob("*.jpg"),
            None
        )

        if (
            test_img
            and config.BEST_MODEL_PATH.exists()
        ):

            try:

                gradcam_info = (
                    generate_gradcam_artifacts(
                        test_img
                    )
                )

            except Exception as error:

                print(
                    "Error generating Grad-CAM preview:",
                    error
                )

    return render_template(
        "explainability.html",
        gradcam=gradcam_info
    )


# =============================================================================
# HISTORY
# =============================================================================

@app.route("/history")
def history_page():

    history_records = get_history(
        limit=100
    )

    stats = get_history_stats()

    return render_template(
        "history.html",
        history=history_records,
        stats=stats
    )


# =============================================================================
# ABOUT
# =============================================================================

@app.route("/about")
def about_page():

    return render_template(
        "about.html"
    )


# =============================================================================
# API - START TRAINING
# =============================================================================

@app.route(
    "/api/start-training",
    methods=["POST"]
)
def api_start_training():

    current_status = load_json(
        config.TRAINING_STATUS_PATH,
        {
            "is_training": False
        }
    )

    if current_status.get(
        "is_training",
        False
    ):

        return jsonify(
            {
                "success": False,
                "message":
                    "Training is already running in background.",
            }
        ), 400

    data = request.get_json() or {}

    model_name = data.get(
        "model_name",
        config.DEFAULT_MODEL
    )

    epochs = int(
        data.get(
            "epochs",
            config.DEFAULT_EPOCHS
        )
    )

    batch_size = int(
        data.get(
            "batch_size",
            config.DEFAULT_BATCH_SIZE
        )
    )

    learning_rate = float(
        data.get(
            "learning_rate",
            config.DEFAULT_LEARNING_RATE
        )
    )

    if model_name == "default":

        model_name = (
            config.DEFAULT_MODEL
        )

    if model_name not in config.SUPPORTED_MODELS:

        return jsonify(
            {
                "success": False,
                "message":
                    f"Unsupported model: {model_name}",
            }
        ), 400

    # ---------------------------------------------------------
    # IMPORTANT:
    # train_model() accepts:
    #
    # model_name
    # epochs
    # batch_size
    # learning_rate
    #
    # It does NOT accept val_split.
    # ---------------------------------------------------------

    with _MODEL_CACHE_LOCK:

        _MODEL_CACHE.clear()

    def run_training_worker():

        try:

            print()
            print("=" * 70)
            print("BACKGROUND TRAINING STARTED")
            print("=" * 70)
            print("Model       :", model_name)
            print("Epochs      :", epochs)
            print("Batch size  :", batch_size)
            print("Learning rate:", learning_rate)
            print("=" * 70)
            print()

            train_model(

                model_name=model_name,

                epochs=epochs,

                batch_size=batch_size,

                learning_rate=learning_rate,
            )

            # -----------------------------------------------------
            # Clear cache after training.
            # -----------------------------------------------------

            with _MODEL_CACHE_LOCK:

                _MODEL_CACHE.clear()

            # -----------------------------------------------------
            # Evaluation
            #
            # train.py creates:
            # models/<model_name>_model.keras
            #
            # Do NOT automatically evaluate best_model.keras
            # here because the selected model is the one just trained.
            # -----------------------------------------------------

            trained_model_path = (
                config.MODELS_DIR
                / f"{model_name}_model.keras"
            )

            if trained_model_path.exists():

                try:

                    evaluate_model(
                        trained_model_path
                    )

                except Exception as error:

                    print(
                        "Automatic evaluation warning:",
                        error
                    )

            # -----------------------------------------------------
            # PCA
            # -----------------------------------------------------

            try:

                generate_pca_feature_space_plot()

            except Exception as error:

                print(
                    "PCA generation warning:",
                    error
                )

            print()
            print(
                "BACKGROUND TRAINING COMPLETED"
            )
            print()

        except Exception as error:

            print()
            print("=" * 70)
            print("BACKGROUND TRAINING FAILED")
            print("=" * 70)
            print(error)
            print("=" * 70)
            print()

    thread = threading.Thread(
        target=run_training_worker,
        daemon=True
    )

    thread.start()

    return jsonify(
        {
            "success": True,
            "message":
                (
                    f"Training started for "
                    f"{model_name} "
                    f"({epochs} epochs, "
                    f"batch size {batch_size})."
                ),
        }
    )


# =============================================================================
# API - TRAINING STATUS
# =============================================================================

@app.route(
    "/api/training-status",
    methods=["GET"]
)
def api_training_status():

    status = load_json(
        config.TRAINING_STATUS_PATH,
        {
            "is_training": False,
            "status": "idle",
            "current_epoch": 0,
            "total_epochs": 0,
            "progress_percent": 0.0,
            "message":
                "No active training job.",
        },
    )

    return jsonify(status)


# =============================================================================
# API - RUN EVALUATION
# =============================================================================

@app.route(
    "/api/run-evaluation",
    methods=["POST"]
)
def api_run_evaluation():

    try:

        # Use the configured best model only when
        # explicitly requested by the evaluation page.
        results = evaluate_model(
            config.BEST_MODEL_PATH
        )

        return jsonify(
            {
                "success": True,
                "evaluation": results,
            }
        )

    except Exception as error:

        return jsonify(
            {
                "success": False,
                "error": str(error),
            }
        ), 500


# =============================================================================
# API - GENERATE STARTER DATA
# =============================================================================

@app.route(
    "/api/generate-starter-data",
    methods=["POST"]
)
def api_generate_starter_data():

    try:

        data = request.get_json() or {}

        count = int(
            data.get(
                "count",
                25
            )
        )

        force = bool(
            data.get(
                "force",
                False
            )
        )

        result = generate_starter_dataset(
            num_per_class=count,
            overwrite=force
        )

        return jsonify(
            {
                "success": True,
                "result": result,
            }
        )

    except Exception as error:

        return jsonify(
            {
                "success": False,
                "error": str(error),
            }
        ), 500


# =============================================================================
# API - DELETE HISTORY
# =============================================================================

@app.route(
    "/api/delete-history/<int:record_id>",
    methods=["POST"]
)
def api_delete_history(
    record_id: int
):

    deleted = delete_prediction(
        record_id
    )

    if deleted:

        flash(
            f"Prediction record #{record_id} "
            "deleted successfully.",
            "success",
        )

    else:

        flash(
            f"Record #{record_id} not found.",
            "warning",
        )

    return redirect(
        url_for("history_page")
    )


# =============================================================================
# API - CLEAR HISTORY
# =============================================================================

@app.route(
    "/api/clear-history",
    methods=["POST"]
)
def api_clear_history():

    count = clear_history()

    flash(
        f"Cleared {count} historical screening records.",
        "success",
    )

    return redirect(
        url_for("history_page")
    )


# =============================================================================
# APPLICATION START
# =============================================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print(
        "AI-Based Tuberculosis (TB) Screening Web Server"
    )
    print("=" * 70)

    print(
        f"Server URL: "
        f"http://{config.FLASK_HOST}:{config.FLASK_PORT}"
    )

    try:

        hardware = get_hardware_info()

        print(
            f"Active Device: "
            f"{hardware.get('device_name', 'Unknown')}"
        )

    except Exception:

        print(
            "Active Device: Unknown"
        )

    print(
        f"Default Model: "
        f"{config.DEFAULT_MODEL}"
    )

    print(
        f"Supported Models: "
        f"{', '.join(config.SUPPORTED_MODELS)}"
    )

    print("=" * 70)
    print()

    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG
    )