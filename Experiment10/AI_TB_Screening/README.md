# AI-Based Tuberculosis (TB) Screening from Chest X-Ray Using Machine Learning and Deep Learning

An end-to-end, academic-grade Computer-Aided Detection (CAD) screening system that analyzes Posteroanterior (PA) chest radiographs to identify potential manifestations of pulmonary tuberculosis. Built with Python, TensorFlow/Keras 3, OpenCV, Scikit-Learn, SQLite, and Flask.

---

> [!WARNING]
> ### MEDICAL SAFETY DISCLAIMER
> **This application is an academic/research prototype for AI-assisted TB screening from chest X-ray images. It is not a certified medical diagnostic device and must not be used as a substitute for professional medical evaluation, sputum microbiological testing, or clinical diagnosis by a licensed healthcare professional.**

---

## Table of Contents
1. [Project Abstract](#1-project-abstract)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Existing System vs Proposed System](#4-existing-system-vs-proposed-system)
5. [End-to-End System Architecture](#5-end-to-end-system-architecture)
6. [Pipeline Workflow](#6-pipeline-workflow)
7. [Dataset Configuration & Instructions](#7-dataset-configuration--instructions)
8. [Image Preprocessing & Augmentation](#8-image-preprocessing--augmentation)
9. [Deep Feature Extraction & PCA Projection](#9-deep-feature-extraction--pca-projection)
10. [Model Architectures & Transfer Learning](#10-model-architectures--transfer-learning)
11. [Model Training & Safeguards](#11-model-training--safeguards)
12. [Evaluation Metrics & Clinical Significance](#12-evaluation-metrics--clinical-significance)
13. [Explainable AI (Grad-CAM)](#13-explainable-ai-grad-cam)
14. [SQLite Database Persistence](#14-sqlite-database-persistence)
15. [Project File Structure](#15-project-file-structure)
16. [Beginner-Friendly Windows Setup Guide](#16-beginner-friendly-windows-setup-guide)
17. [Step-by-Step Execution Guide](#17-step-by-step-execution-guide)
18. [College Viva Voce Q&A](#18-college-viva-voce-qa)
19. [Limitations & Future Enhancements](#19-limitations--future-enhancements)

---

## 1. Project Abstract
Tuberculosis (TB), an airborne infectious disease caused by *Mycobacterium tuberculosis*, is a leading cause of mortality worldwide. Early detection via chest radiography is endorsed by the World Health Organization (WHO) as an essential triaging tool. However, resource-constrained clinics often lack trained radiologists to interpret radiographs in a timely manner.

This project delivers a complete, reproducible Computer-Aided Detection screening system. The platform integrates:
- Automated discovery and stratified splitting of chest X-rays without data leakage.
- Standardized image preprocessing (RGB standardization, bicubic resizing to $224 \times 224 \times 3$, CLAHE contrast enhancement, and $[0, 1]$ float normalization).
- Deep feature extraction generating $1280$-dimensional latent vector embeddings and 2D Principal Component Analysis (PCA) feature space projections.
- Transfer learning with **EfficientNetB0** (and comparative benchmarks against **ResNet50** and a **Custom CNN Baseline**).
- Quantitative test set evaluation measuring Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score, and ROC-AUC with Seaborn heatmaps.
- Gradient-weighted Class Activation Mapping (**Grad-CAM**) providing visual interpretability of model activations across anatomical lung zones.
- An embedded **SQLite** historical database and a responsive, modern medical dark-themed **Flask** web dashboard.

---

## 2. Problem Statement
Manual screening of chest radiographs is labor-intensive, subjective, and prone to inter-observer variability. In high-incidence regions across Asia and Africa, the patient-to-radiologist ratio can exceed $100,000 : 1$. Traditional machine learning approaches rely on handcrafted textural descriptors (e.g. SIFT, GLCM, LBP) that fail to capture subtle apical infiltrates, cavitary lesions, or fibrocalcific opacities. There is a critical clinical need for an automated, reproducible deep learning pipeline capable of rapid, explainable triaging.

---

## 3. Objectives
1. Build a functional, end-to-end deep learning screening pipeline for chest X-ray binary classification (`Normal` vs `TB`).
2. Implement **Transfer Learning** using pre-trained convolutional backbones (primarily `EfficientNetB0`).
3. Guarantee zero data leakage through stratified test set segregation and training-only data augmentation.
4. Provide mathematical interpretability using **Grad-CAM** to highlight regions contributing to model predictions.
5. Create a unified, single-frame dashboard with clickable stages:
   $$\text{Input} \longrightarrow \text{Preprocess} \longrightarrow \text{Features} \longrightarrow \text{Training} \longrightarrow \text{Evaluation} \longrightarrow \text{Prediction}$$
6. Persist screening audits and telemetry in an embedded SQLite database.
7. Support both CPU and GPU execution on standard Windows machines.

---

## 4. Existing System vs Proposed System

| Dimension | Existing / Traditional System | Proposed CAD System |
| :--- | :--- | :--- |
| **Feature Extraction** | Handcrafted features (SIFT, HOG, LBP) | Hierarchical deep features via ImageNet transfer learning ($1280\text{D}$) |
| **Model Generalization** | Overfits easily on small medical sets | Transfer learning with inverted residual blocks (EfficientNetB0) |
| **Data Leakage** | Augmentation applied before dataset splitting | Strict stratified test segregation; augmentation on training split only |
| **Explainability** | Black-box output; clinical distrust | Authentic Grad-CAM heatmaps & anatomical lung zone localization |
| **Architecture Comparison** | Single model without baseline comparison | Tri-model comparison (CNN Baseline, ResNet50, EfficientNetB0) |
| **User Interface** | Terminal scripts or disconnected tools | Unified Flask dashboard with real-time training telemetry |
| **Data Persistence** | Volatile in-memory logs | Embedded SQLite database logging probabilities, latency, and artifacts |

---

## 5. End-to-End System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   AI TB SCREENING PIPELINE ARCHITECTURE                 │
└────────────────────────────────────────────────────────────────────────┘

  📁 INPUT RADIOGRAPH (PA / AP View)
         ↓
  ⚙ PREPROCESSING PIPELINE
     ├── File Integrity & Format Validation (.jpg, .jpeg, .png)
     ├── Color Space Conversion (Grayscale / BGR → RGB 3 Channels)
     ├── Spatial Resizing (Bicubic / Area interpolation to 224 × 224)
     ├── CLAHE (Contrast Limited Adaptive Histogram Equalization)
     └── Float32 Normalization: pixel_val / 255.0  → [0.0, 1.0]
         ↓
  🔍 FEATURE EXTRACTION & EMBEDDINGS
     ├── Backbone: EfficientNetB0 (ImageNet Pre-trained)
     ├── Pooling: GlobalAveragePooling2D
     ├── Latent Embedding: 1280-Dimensional Vector
     └── 2D Dimensionality Reduction: Principal Component Analysis (PCA)
         ↓
  🧠 MODEL TRAINING & TRANSFER LEARNING
     ├── Stratified Partitioning (70% Train, 15% Validation, 15% Test)
     ├── Augmentation (RandomFlip, RandomRotation, RandomZoom — Train Only)
     ├── Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
     └── Serialized Weights: models/best_model.keras
         ↓
  📊 EMPIRICAL EVALUATION (Held-out Test Split)
     ├── Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score
     ├── ROC Curve & AUC Score (Receiver Operating Characteristic)
     └── Confusion Matrix Heatmap (TN, FP, FN, TP)
         ↓
  🩺 INFERENCE & EXPLAINABLE AI (Grad-CAM)
     ├── Screening Probabilities: TB % vs Normal %
     ├── GradientTape partial derivatives on convolutional layer `top_conv`
     ├── Spatial importance weights & ReLU positive filtering
     └── Translucent Jet Colormap Blended Overlay on Original Radiograph
         ↓
  🗄 PERSISTENCE & TELEMETRY
     └── SQLite Database (Timestamp, Latency, Confidence, Artifact Path)
```

---

## 6. Pipeline Workflow

The dashboard features an interactive **One-Frame Workflow** where each stage is directly inspectable:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  📁 Input  →  ⚙ Preprocess  →  🔍 Features  →  🧠 Model  →  📊 Evaluate │
│                                                          →  🩺 Predict  │
└─────────────────────────────────────────────────────────────────────────┘
```

1. **Input (`/dataset`):** Scans the configured dataset directory, validates image formats, tallies class distributions, and renders sample galleries.
2. **Preprocess (`/preprocessing`):** Applies image transformations and provides an educational 5-stage side-by-side comparison with real pixel statistics ($\mu, \sigma, \min, \max$).
3. **Features (`/features`):** Extracts $1280$-dimensional latent feature vectors and projects sample distributions onto a 2D PCA plane.
4. **Model Training (`/training`):** User configures architecture, epochs, batch size, learning rate, and starts training with live progress telemetry.
5. **Evaluation (`/evaluation`):** Evaluates the trained checkpoint on unseen test samples, generating confusion matrices and ROC curves.
6. **Comparison (`/comparison`):** Compares performance metrics across architectures in an empirical ledger.
7. **Prediction (`/prediction`):** Accepts new chest X-rays via drag-and-drop, validates integrity, runs inference, outputs probabilities, and triggers Grad-CAM.
8. **Explainable AI (`/explainability`):** Displays original radiograph, pure activation heatmap, and blended overlay with anatomical zone attribution.
9. **History (`/history`):** Inspects and manages historical screening records stored in SQLite.

---

## 7. Dataset Configuration & Instructions

The application dynamically discovers images from the configured paths in [`config.py`](file:///d:/AI_TB_Screening/config.py):

```text
dataset/
├── TB/
│   ├── TB_0001.jpg
│   ├── TB_0002.jpg
│   └── ...
└── Normal/
    ├── Normal_0001.jpg
    ├── Normal_0002.jpg
    └── ...
```

### Supported Image Formats
- JPEG (`.jpg`, `.jpeg`)
- Portable Network Graphics (`.png`)

### Where to Download Real-World Chest X-Ray Datasets
You can download publicly available TB chest radiograph datasets from:
1. **Kaggle Tuberculosis (TB) Chest X-ray Database:**  
   [https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)  
   Contains 700 TB and 3,500 Normal chest radiographs.
2. **Montgomery County and Shenzhen Chest X-ray Sets (NIH / NLM):**  
   [https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-chest-x-ray-image-data-sets](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-chest-x-ray-image-data-sets)

Place the images into `dataset/TB/` and `dataset/Normal/`. The system will automatically discover, count, and stratify them upon page refresh.

### Built-in Starter Dataset Generator
To enable immediate demonstration, testing, and college viva presentation without downloading gigabytes of data, run:
```powershell
python utils/sample_data_generator.py --count 25
```
This generates anatomically inspired radiographs featuring realistic rib cages, clavicles, lung fields, cardiac silhouettes, and simulated TB apical infiltrates.

---

## 8. Image Preprocessing & Augmentation

Raw chest radiographs from different imaging centers differ in spatial dimensions and exposure. Our preprocessing pipeline in [`preprocessing/image_preprocessor.py`](file:///d:/AI_TB_Screening/preprocessing/image_preprocessor.py) enforces standard inputs:

1. **RGB Color Space Standardization:** Ensures both single-channel grayscale and multi-channel radiographs have a consistent 3-channel shape $(H, W, 3)$.
2. **Spatial Resizing:** Uses bicubic and area interpolation (`cv2.INTER_AREA`) to resize images to $224 \times 224$ pixels.
3. **Contrast Limited Adaptive Histogram Equalization (CLAHE):** Prevents over-amplification of noise while enhancing subtle pulmonary opacities.
4. **Pixel Normalization:** Divides uint8 intensities $[0, 255]$ by $255.0$ to produce float32 tensors bounded in $[0.0, 1.0]$.
5. **Training-Only Data Augmentation:**
   - Random Horizontal Flip
   - Slight Rotation ($\pm 4\%$)
   - Slight Zoom ($\pm 4\%$)
   - Contrast Adjustment ($\pm 8\%$)  
   *Augmentations are applied strictly to the training split. Validation and test sets are never augmented, guaranteeing zero data leakage.*

---

## 9. Deep Feature Extraction & PCA Projection

In [`training/feature_extraction.py`](file:///d:/AI_TB_Screening/training/feature_extraction.py):
- The classification head is removed from the pre-trained EfficientNetB0 backbone.
- A `GlobalAveragePooling2D` layer pools spatial feature maps into a **1280-dimensional feature embedding**.
- The web interface presents an educational numeric table displaying components (Feature 1, Feature 2, etc.), their activation signs, and vector summary metrics ($\mu, \sigma, \|\mathbf{x}\|_2$).
- **Principal Component Analysis (PCA):** Projects high-dimensional embeddings down to 2 principal components to visualize cluster separation between healthy and TB radiographs in latent space.

---

## 10. Model Architectures & Transfer Learning

The system implements and compares three deep neural network architectures:

### 1. EfficientNetB0 (Primary Model - Recommended)
- Utilizes compound scaling to balance network depth, channel width, and input resolution.
- Employs Mobile Inverted Bottleneck Convolution (**MBConv**) blocks with squeeze-and-excitation optimization.
- Fine-tuned with a customized top head:
  `GlobalAveragePooling2D → BatchNormalization → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.2) → Dense(1, Sigmoid)`
- Total Parameters: ~4,200,000 (~16 MB weight size), optimal for fast CPU/GPU inference.

### 2. ResNet50 (Comparison Deep Architecture)
- 50-layer deep residual network utilizing identity skip connections to circumvent vanishing gradients.
- Excellent capacity for large-scale datasets, with ~23,800,000 parameters.

### 3. CNN Baseline (Custom ConvNet)
- A 4-block custom convolutional network (`Conv2D` + `BatchNorm` + `MaxPooling2D` + `Dense(64)` + `Dense(1)`).
- Serves as an empirical baseline to demonstrate the performance gain achieved by ImageNet transfer learning.

---

## 11. Model Training & Safeguards

In [`training/train.py`](file:///d:/AI_TB_Screening/training/train.py):
- **Stratified Split:** Partitions the dataset into 70% Training, 15% Validation, and 15% Held-out Testing.
- **Loss Function:** Binary Cross-Entropy:
  $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
- **Optimizer:** Adam optimizer with configurable learning rate (default: $1 \times 10^{-4}$).
- **EarlyStopping:** Monitors `val_loss` with patience of 4 epochs; restores best model weights automatically.
- **ModelCheckpoint:** Serializes the optimal model checkpoint to `models/best_model.keras`.
- **ReduceLROnPlateau:** Automatically halves the learning rate if validation loss plateaus for 2 epochs.
- **Telemetry Callback:** Writes epoch metrics to `models/training_status.json` enabling the Flask web dashboard to display live progress bars and training loss/accuracy curves.

---

## 12. Evaluation Metrics & Clinical Significance

In [`training/evaluate.py`](file:///d:/AI_TB_Screening/training/evaluate.py), performance is evaluated exclusively on the held-out test split:

- **Accuracy:** Overall proportion of correct classifications:
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
- **Recall / Sensitivity:** Proportion of actual TB cases correctly identified:
  $$\text{Sensitivity} = \frac{TP}{TP + FN}$$
  *Clinically vital: missing an active TB case can lead to disease transmission and patient mortality.*
- **Specificity:** Proportion of healthy controls correctly identified:
  $$\text{Specificity} = \frac{TN}{TN + FP}$$
  *Minimizes unnecessary downstream diagnostic burdens (e.g. CT scans, GeneXpert).*
- **Precision (Positive Predictive Value):**
  $$\text{Precision} = \frac{TP}{TP + FP}$$
- **F1-Score:** Harmonic mean of Precision and Recall:
  $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **ROC-AUC:** Area under the Receiver Operating Characteristic curve plotting True Positive Rate vs False Positive Rate across all diagnostic thresholds.
- **Confusion Matrix:** Rendered graphically via Seaborn heatmap showing counts of $TN, FP, FN, TP$.

---

## 13. Explainable AI (Grad-CAM)

In [`explainability/gradcam.py`](file:///d:/AI_TB_Screening/explainability/gradcam.py):
To foster clinical interpretability, the system computes **Gradient-Weighted Class Activation Mapping**:
1. Uses `tf.GradientTape` to compute gradients of the class score $y^c$ with respect to feature activation maps $A^k$ of the final convolutional layer (`top_conv`):
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
2. Computes the weighted combination of forward activation maps and applies a ReLU function:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
3. Generates 3 visual outputs:
   - Original standardized radiograph
   - Pure Jet activation heatmap
   - Translucent blended overlay ($\alpha = 0.40$)
4. Analyzes the spatial centroid of activation to describe the anatomical focus (e.g. *Right Lung - Apical / Upper Lung Zone*).

---

## 14. SQLite Database Persistence

All screening predictions are permanently recorded in [`database/tb_screening.db`](file:///d:/AI_TB_Screening/database/tb_screening.db):

### Table Schema: `predictions`
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | `INTEGER PRIMARY KEY AUTOINCREMENT` | Unique screening identifier |
| `date` | `TEXT` | Screening date (`YYYY-MM-DD`) |
| `time` | `TEXT` | Screening timestamp (`HH:MM:SS`) |
| `image_filename` | `TEXT` | Uploaded image reference |
| `model_name` | `TEXT` | Architecture used (e.g. `EfficientNetB0`) |
| `prediction` | `TEXT` | Screening result (`TB` or `Normal`) |
| `tb_probability` | `REAL` | Calculated TB probability percentage |
| `normal_probability` | `REAL` | Calculated Normal probability percentage |
| `gradcam_image` | `TEXT` | Generated Grad-CAM overlay filename |
| `execution_time_ms` | `REAL` | End-to-end inference latency in ms |
| `created_at` | `TIMESTAMP` | System creation timestamp |

The History interface supports viewing full Grad-CAM overlays, deleting individual records, and clearing historical logs.

---

## 15. Project File Structure

```text
AI_TB_Screening/
├── app.py                         # Flask web application server & REST API
├── config.py                      # Global configuration, hyperparameters & paths
├── requirements.txt               # Pinned Python package dependencies
├── test_pipeline.py               # Comprehensive automated test suite
├── README.md                      # Complete academic project documentation
│
├── dataset/                       # Dataset directory (configurable)
│   ├── TB/                        # Tuberculosis chest radiographs
│   └── Normal/                    # Healthy control chest radiographs
│
├── models/                        # Serialized weights and metadata
│   ├── best_model.keras           # Optimal trained checkpoint
│   ├── EfficientNetB0_model.keras # Architecture checkpoint
│   ├── CNN_Baseline_model.keras   # Baseline checkpoint
│   ├── test_split.json            # Held-out test split (prevents leakage)
│   ├── model_metadata.json        # Hyperparameters and training history
│   ├── evaluation_results.json    # Test metrics and confusion values
│   ├── model_comparison.json      # Comparative multi-architecture ledger
│   └── training_status.json       # Live training telemetry for web UI
│
├── training/                      # Training, evaluation & feature extraction
│   ├── train.py                   # Model training script with callbacks
│   ├── evaluate.py                # Test set evaluation & plot generator
│   └── feature_extraction.py      # Latent embeddings & 2D PCA projection
│
├── preprocessing/                 # Image preprocessing pipeline
│   └── image_preprocessor.py      # RGB conversion, CLAHE, resizing, normalization
│
├── explainability/                # Explainable AI
│   └── gradcam.py                 # Grad-CAM heatmap & overlay generator
│
├── database/                      # Persistence layer
│   ├── database.py                # SQLite database interface & CRUD queries
│   └── tb_screening.db            # SQLite database file
│
├── utils/                         # Helper utilities
│   ├── helpers.py                 # Hardware detection, dataset scanning, JSON I/O
│   └── sample_data_generator.py   # Anatomical starter dataset generator
│
├── static/                        # Web assets & generated artifacts
│   ├── css/
│   │   └── style.css              # Custom medical dark-themed stylesheet
│   ├── js/
│   │   └── script.js              # Drag-and-drop & live training polling
│   ├── uploads/                   # Uploaded chest X-rays
│   ├── results/                   # Generated Grad-CAM overlays & preproc plots
│   └── plots/                     # Confusion matrix, ROC curve, training curves
│
└── templates/                     # Jinja2 HTML templates
    ├── base.html                  # Master layout with sidebar & disclaimer
    ├── index.html                 # Dashboard with One-Frame clickable workflow
    ├── dataset.html               # Dataset explorer & class distribution
    ├── preprocessing.html         # 5-stage preprocessing visualization
    ├── features.html              # 1280D vector preview & 2D PCA plot
    ├── training.html              # Hyperparameter controls & live training UI
    ├── evaluation.html            # Confusion matrix & ROC curve dashboard
    ├── comparison.html            # Multi-architecture comparison table
    ├── prediction.html            # Drag-and-drop inference interface
    ├── explainability.html        # 3-panel Grad-CAM visualization
    ├── history.html               # SQLite historical log table
    └── about.html                 # Project report & viva voce Q&A
```

---

## 16. Beginner-Friendly Windows Setup Guide

Follow these exact steps to run the project on any Windows 10/11 system:

### Step 1: Install Python
Ensure Python **3.10** or **3.11** is installed from [python.org](https://www.python.org/downloads/).  
*During installation, make sure to check the box: **"Add Python to PATH"**.*

### Step 2: Open Windows PowerShell
Press `Win + X` and select **Terminal** or **Windows PowerShell**.

### Step 3: Navigate to Project Directory
```powershell
cd D:\AI_TB_Screening
```

### Step 4: Create a Virtual Environment
```powershell
python -m venv venv
```

### Step 5: Activate the Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```
> [!TIP]
> If you encounter a script execution error in PowerShell (`PSSecurityException`), run this safe command once to allow local script activation:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\venv\Scripts\Activate.ps1
> ```

### Step 6: Install Required Packages
```powershell
pip install -r requirements.txt
```

---

## 17. Step-by-Step Execution Guide

### Option A: Run the Complete Web Application
Start the Flask web dashboard:
```powershell
python app.py
```
Open Google Chrome or any modern browser and navigate to:
```text
http://127.0.0.1:5000
```

### Option B: Run Individual Modules via CLI

#### 1. Generate Starter Dataset (Optional)
If you do not have Kaggle chest radiographs downloaded yet:
```powershell
python utils/sample_data_generator.py --count 25
```

#### 2. Train the Model from Terminal
```powershell
python training/train.py --model EfficientNetB0 --epochs 5 --batch_size 8 --lr 0.0001
```

#### 3. Evaluate the Model on the Held-Out Test Set
```powershell
python training/evaluate.py
```

#### 4. Run the Deep Feature Extractor & PCA
```powershell
python training/feature_extraction.py
```

#### 5. Generate Grad-CAM for a Sample Radiograph
```powershell
python explainability/gradcam.py
```

#### 6. Run the Automated End-to-End Test Suite
```powershell
python test_pipeline.py
```

---

## 18. College Viva Voce Q&A

### Q1: What is Tuberculosis and how does chest radiography detect it?
**Answer:** Tuberculosis is an infection caused by *Mycobacterium tuberculosis* primarily affecting the lungs. Radiographs reveal hallmark structural alterations: patchy airspace consolidation, apical infiltrates, cavitary lesions with thick walls, and pleural thickening.

### Q2: Why is EfficientNetB0 preferred over ResNet50 or VGG16?
**Answer:** EfficientNetB0 uses compound scaling to uniformly scale network depth, width, and resolution. With only ~4.2 million parameters (compared to ~25 million in ResNet50 and ~138 million in VGG16), it achieves high accuracy on medical imaging tasks while executing fast inferences on standard CPUs.

### Q3: What is the difference between Grad-CAM and standard CAM?
**Answer:** Class Activation Mapping (CAM) requires replacing fully connected layers with Global Average Pooling immediately preceding the softmax layer, necessitating architectural redesign and retraining. Grad-CAM generalizes CAM by using gradients of the target class with respect to any convolutional layer via backpropagation, requiring no architectural modification.

### Q4: How does the system ensure there is no data leakage?
**Answer:** The dataset is split into training, validation, and test partitions *prior* to data transformation. Augmentations (rotation, zoom, flips) are applied strictly during runtime training on the training batches. The test set is stored in `models/test_split.json` and remains unaugmented.

### Q5: What is the significance of the ROC-AUC score?
**Answer:** The Receiver Operating Characteristic (ROC) curve evaluates sensitivity versus specificity across all diagnostic discrimination thresholds. An AUC of $1.0$ indicates perfect separability, while $0.5$ represents random guessing. In medical screening, AUC provides threshold-independent assessment of diagnostic power.

---

## 19. Limitations & Future Enhancements

### Current Limitations
1. **Binary Target:** Distinguishes only between normal and TB; does not differentiate other thoracic pathologies such as viral pneumonia, lung cancer, or COVID-19.
2. **2D Projection Only:** Does not account for lateral views, prior clinical history, or sputum laboratory assays.
3. **Hardware Constraint:** CPU execution takes approximately 1–2 seconds per inference, whereas GPU execution achieves real-time inference (<100 ms).

### Planned Future Enhancements
1. **Multi-label Thoracic Classification:** Expand classification across the NIH ChestX-ray14 taxonomy (Pneumothorax, Atelectasis, Effusion, Infiltration, etc.).
2. **DICOM Integration:** Implement DICOM parser with PACS (Picture Archiving and Communication System) network integration.
3. **Vision Transformers (ViT):** Benchmark self-attention architectures (Swin Transformer, DeiT) against convolutional backbones.
4. **Federated Learning:** Enable decentralized training across hospital networks preserving patient data privacy.
