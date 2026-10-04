"""
=============================================================================
AI-Based Tuberculosis (TB) Screening from Chest X-Ray
Explainable AI Pipeline Using Grad-CAM
=============================================================================
Computes Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize
the spatial regions in the chest radiograph that most strongly influence
the deep neural network's screening inference.
"""

import sys
import os
import uuid
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

import numpy as np
import tensorflow as tf
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from preprocessing.image_preprocessor import preprocess_image

def find_target_conv_layer(model: tf.keras.Model) -> Tuple[Optional[tf.keras.layers.Layer], Optional[tf.keras.Model]]:
    """
    Finds the deepest 4D convolutional feature layer in the model or nested sub-models.
    Prioritizes:
    - 'top_conv' for EfficientNet architectures
    - 'conv5_block3_out' for ResNet50
    - Last Conv2D layer in CNN baseline
    """
    # Check if there is a nested submodel (e.g. efficientnetb0 or resnet50)
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            sub_model = layer
            # Look for top_conv or conv5_block3_out
            for sub_layer in reversed(sub_model.layers):
                if sub_layer.name in ["top_conv", "conv5_block3_out"]:
                    return sub_layer, sub_model
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer, sub_model

    # Look in the root model
    for layer in reversed(model.layers):
        if layer.name in ["top_conv", "conv5_block3_out"]:
            return layer, None
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer, None

    return None, None


def compute_gradcam_heatmap(
    model: tf.keras.Model,
    img_array: np.ndarray,
    target_class_idx: int = 1
) -> Tuple[np.ndarray, str]:
    """
    Calculates the 2D Grad-CAM attention heatmap using TensorFlow GradientTape.
    img_array: float32 array of shape (1, 224, 224, 3)
    target_class_idx: 1 for TB screening focus
    """
    conv_layer, sub_model = find_target_conv_layer(model)
    if conv_layer is None:
        raise ValueError("Could not find a convolutional layer in the specified model for Grad-CAM.")

    target_layer_name = conv_layer.name

    if sub_model is not None:
        # Model has a nested transfer learning backbone
        feature_model = tf.keras.Model(sub_model.inputs, [conv_layer.output, sub_model.output])
        # Find index of sub_model in top model
        sub_model_idx = model.layers.index(sub_model)
        classifier_layers = model.layers[sub_model_idx + 1:]

        with tf.GradientTape() as tape:
            conv_outputs, backbone_features = feature_model(img_array)
            tape.watch(conv_outputs)

            x = backbone_features
            for layer in classifier_layers:
                x = layer(x)

            # Score for class (sigmoid output: single unit)
            if x.shape[-1] == 1:
                class_score = x[:, 0]
            else:
                class_score = x[:, target_class_idx]

        grads = tape.gradient(class_score, conv_outputs)
    else:
        # Standard unrolled or custom CNN model
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            tape.watch(conv_outputs)
            if predictions.shape[-1] == 1:
                class_score = predictions[:, 0]
            else:
                class_score = predictions[:, target_class_idx]

        grads = tape.gradient(class_score, conv_outputs)

    if grads is None:
        raise ValueError("Gradient calculation returned None. Ensure the model is differentiable.")

    # Global average pooling of gradients: weights across feature maps
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Linear combination of convolutional feature maps
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)

    # Apply ReLU: only positive contributions towards target class
    cam = tf.maximum(cam, 0)

    # Normalize heatmap between 0 and 1
    max_val = tf.reduce_max(cam)
    if max_val > 0:
        cam = cam / max_val

    return cam.numpy(), target_layer_name


def generate_gradcam_artifacts(
    image_path: Union[str, Path],
    model: Optional[tf.keras.Model] = None,
    model_path: Optional[Path] = None,
    alpha: float = 0.40
) -> Dict[str, Any]:
    """
    End-to-end Grad-CAM pipeline:
    1. Loads and preprocesses the chest X-ray.
    2. Runs model inference.
    3. Computes Grad-CAM gradients from the target convolutional layer.
    4. Generates:
       - High-resolution heatmap (JET colormap)
       - Superimposed blended overlay on the original radiograph
    5. Saves visual artifacts to static/results/
    6. Identifies highest activation focus region (lung zone).
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at {path}")

    # Load model if not provided
    if model is None:
        target_path = model_path or config.BEST_MODEL_PATH
        if not target_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {target_path}. "
                "Please train the model before requesting Grad-CAM explanations."
            )
        model = tf.keras.models.load_model(str(target_path))

    # Read original image for background overlay
    orig_bgr = cv2.imread(str(path))
    if orig_bgr is None:
        raise ValueError(f"Cannot decode image: {path}")
    orig_h, orig_w = orig_bgr.shape[:2]
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    # Preprocess image for model input
    norm_img, meta = preprocess_image(path, enhance_contrast=False)
    batch_img = np.expand_dims(norm_img, axis=0)

    # Predict screening score
    pred_prob = float(model.predict(batch_img, verbose=0)[0][0])
    tb_percent = round(pred_prob * 100, 2)
    normal_percent = round((1.0 - pred_prob) * 100, 2)
    predicted_class = "TB" if pred_prob >= 0.5 else "Normal"

    # Compute raw Grad-CAM heatmap
    cam_2d, layer_name = compute_gradcam_heatmap(model, batch_img, target_class_idx=1)

    # Resize CAM to match original image dimensions
    cam_resized = cv2.resize(cam_2d, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.clip(cam_resized, 0, 1)

    # Convert to 8-bit heatmap
    heatmap_uint8 = np.uint8(255 * cam_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend original image and heatmap
    overlay_rgb = cv2.addWeighted(orig_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)

    # Determine region of highest activation (anatomical zone focus)
    max_idx = np.unravel_index(np.argmax(cam_resized), cam_resized.shape)
    peak_y, peak_x = max_idx[0] / orig_h, max_idx[1] / orig_w

    vert_zone = "Apical / Upper Lung Zone" if peak_y < 0.38 else (
        "Mid Lung Zone" if peak_y < 0.68 else "Basilar / Lower Lung Zone"
    )
    horiz_zone = "Right Lung (Viewer Left)" if peak_x < 0.50 else "Left Lung (Viewer Right)"
    anatomical_focus = f"{horiz_zone} - {vert_zone}"

    # Generate unique filenames and save artifacts
    run_id = uuid.uuid4().hex[:8]
    stem = path.stem

    orig_filename = f"gradcam_orig_{stem}_{run_id}.jpg"
    heat_filename = f"gradcam_heat_{stem}_{run_id}.jpg"
    over_filename = f"gradcam_overlay_{stem}_{run_id}.jpg"

    cv2.imwrite(str(config.RESULTS_DIR / orig_filename), cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(config.RESULTS_DIR / heat_filename), cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(config.RESULTS_DIR / over_filename), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

    return {
        "success": True,
        "input_filename": path.name,
        "predicted_class": predicted_class,
        "tb_probability": tb_percent,
        "normal_probability": normal_percent,
        "target_conv_layer": layer_name,
        "anatomical_focus": anatomical_focus,
        "peak_activation_score": round(float(np.max(cam_resized)), 4),
        "mean_activation_score": round(float(np.mean(cam_resized)), 4),
        "original_image_url": f"/static/results/{orig_filename}",
        "heatmap_image_url": f"/static/results/{heat_filename}",
        "overlay_image_url": f"/static/results/{over_filename}",
        "overlay_filename": over_filename,
        "explanation": (
            "The Grad-CAM visualization highlights the specific feature maps and anatomical regions "
            "that contributed most intensely to the neural network's screening decision. "
            "Warm colors (red/orange) represent maximum positive model activation, while cool colors "
            "(blue/cyan) represent background or unweighted regions. "
            "This visualization illustrates model attention and does NOT represent a medically confirmed clinical lesion."
        ),
        "medical_disclaimer": config.MEDICAL_DISCLAIMER
    }

if __name__ == "__main__":
    print("Testing Grad-CAM generation...")
    test_img = next(config.TB_DIR.glob("*.jpg"), None)
    if test_img and config.BEST_MODEL_PATH.exists():
        res = generate_gradcam_artifacts(test_img)
        print("Grad-CAM generation successful:")
        print("  Predicted Class   :", res["predicted_class"])
        print("  TB Probability    :", res["tb_probability"])
        print("  Conv Layer        :", res["target_conv_layer"])
        print("  Anatomical Focus  :", res["anatomical_focus"])
        print("  Overlay Artifact  :", res["overlay_image_url"])
    else:
        print("Ensure dataset and trained model exist before testing Grad-CAM standalone.")
