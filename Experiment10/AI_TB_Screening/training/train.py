"""
AI-Based TB Screening from Chest X-Ray
Training Pipeline

Labels:
    Normal = 0
    TB     = 1

Academic/research prototype only.
Not a medical diagnostic device.
"""

import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

import config

from preprocessing.image_preprocessor import preprocess_image
from utils.helpers import save_json


# ============================================================
# RANDOM SEED
# ============================================================

np.random.seed(config.RANDOM_SEED)
tf.random.set_seed(config.RANDOM_SEED)


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path, label):
    """
    Loads an image using the existing project preprocessor.

    preprocess_image() returns values approximately in [0, 1].
    We convert them to [0, 255] because EfficientNetB0
    contains its own input rescaling.
    """

    path = path.numpy().decode("utf-8")

    image, _ = preprocess_image(
        path,
        enhance_contrast=False
    )

    image = image.astype(np.float32) * 255.0

    label = np.float32(label)

    return image, label


# ============================================================
# DATASET CREATION
# ============================================================

def create_dataset(
    paths,
    labels,
    batch_size=4,
    training=False
):

    def loader(path, label):

        image, label = tf.py_function(
            func=load_image,
            inp=[path, label],
            Tout=[tf.float32, tf.float32]
        )

        image.set_shape(config.INPUT_SHAPE)
        label.set_shape([])

        return image, label

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:

        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=config.RANDOM_SEED
        )

    dataset = dataset.map(
        loader,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # --------------------------------------------------------
    # Data augmentation
    # --------------------------------------------------------

    if training:

        augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip(
                    mode="horizontal"
                ),
                tf.keras.layers.RandomRotation(
                    factor=0.03
                ),
                tf.keras.layers.RandomZoom(
                    height_factor=0.05,
                    width_factor=0.05
                ),
                tf.keras.layers.RandomContrast(
                    factor=0.05
                )
            ],
            name="augmentation"
        )

        dataset = dataset.map(
            lambda x, y: (
                augmentation(x, training=True),
                y
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# BUILD EFFICIENTNETB0
# ============================================================

def build_efficientnet():

    print()
    print("Loading EfficientNetB0...")

    try:

        base_model = tf.keras.applications.EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=config.INPUT_SHAPE
        )

        print("ImageNet weights loaded successfully.")

    except Exception as error:

        print("Could not load ImageNet weights.")
        print("Using weights=None.")
        print("Reason:", error)

        base_model = tf.keras.applications.EfficientNetB0(
            weights=None,
            include_top=False,
            input_shape=config.INPUT_SHAPE
        )

    # Freeze backbone because dataset is very small.
    base_model.trainable = False

    inputs = tf.keras.Input(
        shape=config.INPUT_SHAPE,
        name="input_image"
    )

    # EfficientNetB0 has internal rescaling.
    x = base_model(
        inputs,
        training=False
    )

    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name="batch_normalization"
    )(x)

    x = tf.keras.layers.Dropout(
        0.40,
        name="dropout_1"
    )(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        name="dense_features"
    )(x)

    x = tf.keras.layers.Dropout(
        0.30,
        name="dropout_2"
    )(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="tb_prediction"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="EfficientNetB0_TB_Screening"
    )

    return model


# ============================================================
# BUILD RESNET50
# ============================================================

def build_resnet50():

    print()
    print("Loading ResNet50...")

    try:

        base_model = tf.keras.applications.ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=config.INPUT_SHAPE
        )

        print("ImageNet weights loaded successfully.")

    except Exception as error:

        print("Could not load ImageNet weights.")
        print("Using weights=None.")
        print("Reason:", error)

        base_model = tf.keras.applications.ResNet50(
            weights=None,
            include_top=False,
            input_shape=config.INPUT_SHAPE
        )

    base_model.trainable = False

    inputs = tf.keras.Input(
        shape=config.INPUT_SHAPE,
        name="input_image"
    )

    x = tf.keras.layers.Lambda(
        tf.keras.applications.resnet50.preprocess_input,
        name="resnet_preprocess"
    )(inputs)

    x = base_model(
        x,
        training=False
    )

    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name="batch_normalization"
    )(x)

    x = tf.keras.layers.Dropout(
        0.40,
        name="dropout_1"
    )(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        name="dense_features"
    )(x)

    x = tf.keras.layers.Dropout(
        0.30,
        name="dropout_2"
    )(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="tb_prediction"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="ResNet50_TB_Screening"
    )

    return model


# ============================================================
# BUILD BASIC CNN
# ============================================================

def build_cnn():

    inputs = tf.keras.Input(
        shape=config.INPUT_SHAPE,
        name="input_image"
    )

    x = tf.keras.layers.Rescaling(
        1.0 / 255.0,
        name="rescaling"
    )(inputs)

    x = tf.keras.layers.Conv2D(
        32,
        3,
        padding="same",
        activation="relu"
    )(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(
        64,
        3,
        padding="same",
        activation="relu"
    )(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(
        128,
        3,
        padding="same",
        activation="relu"
    )(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(
        0.40
    )(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="tb_prediction"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="CNN_Baseline_TB_Screening"
    )

    return model


# ============================================================
# MODEL SELECTOR
# ============================================================

def build_model(model_name):

    if model_name == "EfficientNetB0":

        return build_efficientnet()

    if model_name == "ResNet50":

        return build_resnet50()

    if model_name == "CNN_Baseline":

        return build_cnn()

    raise ValueError(
        "Invalid model name: "
        + str(model_name)
    )


# ============================================================
# GET DATASET
# ============================================================

def get_all_images():

    normal_paths = []
    tb_paths = []

    allowed = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    }

    if config.NORMAL_DIR.exists():

        for file in config.NORMAL_DIR.iterdir():

            if (
                file.is_file()
                and file.suffix.lower() in allowed
            ):
                normal_paths.append(
                    str(file)
                )

    if config.TB_DIR.exists():

        for file in config.TB_DIR.iterdir():

            if (
                file.is_file()
                and file.suffix.lower() in allowed
            ):
                tb_paths.append(
                    str(file)
                )

    paths = normal_paths + tb_paths

    labels = (
        [0] * len(normal_paths)
        +
        [1] * len(tb_paths)
    )

    print()
    print("DATASET")
    print("----------------------------------------")
    print(
        "Normal images:",
        len(normal_paths)
    )
    print(
        "TB images    :",
        len(tb_paths)
    )
    print(
        "Total images :",
        len(paths)
    )

    if len(paths) < 6:

        raise ValueError(
            "Not enough images in the dataset."
        )

    return paths, labels


# ============================================================
# SPLIT DATASET
# ============================================================

def split_dataset():

    paths, labels = get_all_images()

    train_paths, test_paths, train_labels, test_labels = (
        train_test_split(
            paths,
            labels,
            test_size=config.DEFAULT_TEST_SPLIT,
            stratify=labels,
            random_state=config.RANDOM_SEED
        )
    )

    train_paths, val_paths, train_labels, val_labels = (
        train_test_split(
            train_paths,
            train_labels,
            test_size=config.DEFAULT_VAL_SPLIT,
            stratify=train_labels,
            random_state=config.RANDOM_SEED
        )
    )

    print()
    print("DATA SPLIT")
    print("----------------------------------------")

    print(
        "Training   :",
        len(train_paths)
    )

    print(
        "Validation :",
        len(val_paths)
    )

    print(
        "Testing    :",
        len(test_paths)
    )

    # --------------------------------------------------------
    # Save test split
    # --------------------------------------------------------

    test_information = {

        "test_paths": test_paths,

        "test_labels": test_labels,

        "total_test": len(test_paths),

        "test_tb_count": int(
            sum(test_labels)
        ),

        "test_normal_count": int(
            len(test_labels)
            -
            sum(test_labels)
        )
    }

    save_json(
        test_information,
        config.MODELS_DIR / "test_split.json"
    )

    return (
        train_paths,
        train_labels,
        val_paths,
        val_labels,
        test_paths,
        test_labels
    )


# ============================================================
# SAVE TRAINING GRAPH
# ============================================================

def save_training_graph(history):

    epochs = range(
        1,
        len(history.history["loss"]) + 1
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history.history["accuracy"],
        marker="o",
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        history.history["val_accuracy"],
        marker="s",
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title(
        "TB Screening Model Accuracy"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        config.PLOTS_DIR
        /
        "training_history.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    return "/static/plots/training_history.png"


# ============================================================
# UPDATE TRAINING STATUS
# ============================================================

def update_status(
    model_name,
    status,
    message,
    epoch=0,
    total_epochs=0
):

    if total_epochs > 0:

        progress = (
            epoch
            /
            total_epochs
            *
            100
        )

    else:

        progress = 0

    data = {

        "is_training": (
            status == "training"
        ),

        "status": status,

        "model_name": model_name,

        "current_epoch": epoch,

        "total_epochs": total_epochs,

        "progress_percent": round(
            progress,
            1
        ),

        "message": message,

        "updated_at": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }

    save_json(
        data,
        config.TRAINING_STATUS_PATH
    )


# ============================================================
# TRAINING CALLBACK
# ============================================================

class TrainingStatusCallback(
    tf.keras.callbacks.Callback
):

    def __init__(
        self,
        model_name,
        total_epochs
    ):

        super().__init__()

        self.model_name = model_name
        self.total_epochs = total_epochs

    def on_epoch_end(
        self,
        epoch,
        logs=None
    ):

        logs = logs or {}

        epoch_number = epoch + 1

        accuracy = logs.get(
            "accuracy",
            0
        )

        val_accuracy = logs.get(
            "val_accuracy",
            0
        )

        message = (
            f"Epoch {epoch_number}/"
            f"{self.total_epochs} completed. "
            f"Training accuracy: "
            f"{accuracy:.2%}, "
            f"Validation accuracy: "
            f"{val_accuracy:.2%}"
        )

        update_status(
            self.model_name,
            "training",
            message,
            epoch_number,
            self.total_epochs
        )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    model_name="EfficientNetB0",
    epochs=15,
    batch_size=4,
    learning_rate=0.0001
):

    start_time = time.time()

    print()
    print("=" * 60)
    print("TB SCREENING MODEL TRAINING")
    print("=" * 60)

    print(
        "Model:",
        model_name
    )

    print(
        "Epochs:",
        epochs
    )

    print(
        "Batch size:",
        batch_size
    )

    print(
        "Learning rate:",
        learning_rate
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    update_status(
        model_name,
        "starting",
        "Preparing dataset...",
        0,
        epochs
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    (
        train_paths,
        train_labels,
        val_paths,
        val_labels,
        test_paths,
        test_labels
    ) = split_dataset()

    # --------------------------------------------------------
    # TensorFlow datasets
    # --------------------------------------------------------

    train_dataset = create_dataset(
        train_paths,
        train_labels,
        batch_size=batch_size,
        training=True
    )

    val_dataset = create_dataset(
        val_paths,
        val_labels,
        batch_size=batch_size,
        training=False
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    update_status(
        model_name,
        "starting",
        "Building model...",
        0,
        epochs
    )

    model = build_model(
        model_name
    )

    print()
    print("MODEL CREATED")
    print("----------------------------------------")

    print(
        "Total parameters:",
        f"{model.count_params():,}"
    )

    trainable_parameters = int(
        sum(
            np.prod(
                variable.shape
            )
            for variable in model.trainable_weights
        )
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}"
    )

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            ),
            tf.keras.metrics.AUC(
                name="auc"
            )
        ]
    )

    # --------------------------------------------------------
    # Model save path
    # --------------------------------------------------------

    model_path = (
        config.MODELS_DIR
        /
        f"{model_name}_model.keras"
    )

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    callbacks = [

        TrainingStatusCallback(
            model_name,
            epochs
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=0.000001,
            verbose=1
        )
    ]

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    update_status(
        model_name,
        "training",
        "Training started...",
        0,
        epochs
    )

    try:

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

    except Exception as error:

        update_status(
            model_name,
            "failed",
            str(error),
            0,
            epochs
        )

        raise

    # ----------------------------------------------------