/* ==========================================================================
   AI-Based TB Screening - Client-Side Interactivity & Polling
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
    initDragAndDrop();
    initTrainingPolling();
});

/**
 * Configures drag-and-drop file upload zones and image previews.
 */
function initDragAndDrop() {
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("xray-file-input");
    const previewContainer = document.getElementById("preview-container");
    const previewImage = document.getElementById("preview-image");
    const fileNameDisplay = document.getElementById("file-name-display");
    const fileSizeDisplay = document.getElementById("file-size-display");

    if (!dropzone || !fileInput) return;

    // Trigger file selection on click
    dropzone.addEventListener("click", () => fileInput.click());

    // Highlight on drag
    ["dragenter", "dragover"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dragover");
        }, false);
    });

    // Handle dropped files
    dropzone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            fileInput.files = files;
            displayPreview(files[0]);
        }
    });

    // Handle manual file selection
    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            displayPreview(fileInput.files[0]);
        }
    });

    function displayPreview(file) {
        // Validate extension
        const allowedExts = ["jpg", "jpeg", "png"];
        const ext = file.name.split(".").pop().toLowerCase();
        if (!allowedExts.includes(ext)) {
            alert(`Unsupported format (.${ext}). Please select a JPG, JPEG, or PNG image.`);
            fileInput.value = "";
            return;
        }

        // Validate max size (16MB)
        if (file.size > 16 * 1024 * 1024) {
            alert("File is too large. Maximum supported size is 16 MB.");
            fileInput.value = "";
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            if (previewImage) previewImage.src = e.target.result;
            if (previewContainer) previewContainer.style.display = "block";
            if (fileNameDisplay) fileNameDisplay.textContent = file.name;
            if (fileSizeDisplay) fileSizeDisplay.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
        };
        reader.readAsDataURL(file);
    }
}

/**
 * Handles training API execution and periodic polling for real-time progress.
 */
let trainingPollInterval = null;

function initTrainingPolling() {
    const trainingContainer = document.getElementById("training-progress-section");
    if (!trainingContainer) return;

    // Check if training is currently running on initial load
    fetch("/api/training-status")
        .then(res => res.json())
        .then(data => {
            if (data.is_training) {
                showTrainingUI(data);
                startStatusPolling();
            }
        })
        .catch(err => console.error("Status check failed:", err));
}

function triggerTraining() {
    const modelSelect = document.getElementById("train-model-name");
    const epochsInput = document.getElementById("train-epochs");
    const batchInput = document.getElementById("train-batch-size");
    const lrInput = document.getElementById("train-lr");
    const valSplitInput = document.getElementById("train-val-split");
    const btn = document.getElementById("btn-start-training");

    const payload = {
        model_name: modelSelect ? modelSelect.value : "EfficientNetB0",
        epochs: epochsInput ? parseInt(epochsInput.value) : 10,
        batch_size: batchInput ? parseInt(batchInput.value) : 16,
        learning_rate: lrInput ? parseFloat(lrInput.value) : 0.0001,
        val_split: valSplitInput ? parseFloat(valSplitInput.value) : 0.15
    };

    if (btn) btn.disabled = true;

    fetch("/api/start-training", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            startStatusPolling();
        } else {
            alert(data.message || "Failed to start training.");
            if (btn) btn.disabled = false;
        }
    })
    .catch(err => {
        alert("Training request error: " + err);
        if (btn) btn.disabled = false;
    });
}

function startStatusPolling() {
    if (trainingPollInterval) clearInterval(trainingPollInterval);

    const progressSection = document.getElementById("training-progress-section");
    if (progressSection) progressSection.style.display = "block";

    trainingPollInterval = setInterval(() => {
        fetch("/api/training-status")
            .then(res => res.json())
            .then(status => {
                updateTrainingUI(status);
                if (!status.is_training) {
                    clearInterval(trainingPollInterval);
                    trainingPollInterval = null;
                    const btn = document.getElementById("btn-start-training");
                    if (btn) btn.disabled = false;

                    if (status.status === "completed") {
                        setTimeout(() => {
                            window.location.reload();
                        }, 1200);
                    }
                }
            })
            .catch(err => console.error("Polling error:", err));
    }, 1500);
}

function updateTrainingUI(status) {
    const bar = document.getElementById("training-progress-bar");
    const percentText = document.getElementById("training-percent-text");
    const statusMsg = document.getElementById("training-status-message");
    const currentEpoch = document.getElementById("stat-curr-epoch");
    const trainAcc = document.getElementById("stat-train-acc");
    const valAcc = document.getElementById("stat-val-acc");
    const trainLoss = document.getElementById("stat-train-loss");
    const valLoss = document.getElementById("stat-val-loss");

    if (bar) bar.style.width = `${status.progress_percent || 0}%`;
    if (percentText) percentText.textContent = `${status.progress_percent || 0}%`;
    if (statusMsg) statusMsg.textContent = status.message || "Training in progress...";

    if (currentEpoch) currentEpoch.textContent = `${status.current_epoch || 0} / ${status.total_epochs || 0}`;
    if (trainAcc && status.latest_accuracy !== undefined) trainAcc.textContent = `${(status.latest_accuracy * 100).toFixed(1)}%`;
    if (valAcc && status.latest_val_accuracy !== undefined) valAcc.textContent = `${(status.latest_val_accuracy * 100).toFixed(1)}%`;
    if (trainLoss && status.latest_loss !== undefined) trainLoss.textContent = status.latest_loss;
    if (valLoss && status.latest_val_loss !== undefined) valLoss.textContent = status.latest_val_loss;
}

function showTrainingUI(status) {
    const progressSection = document.getElementById("training-progress-section");
    if (progressSection) progressSection.style.display = "block";
    updateTrainingUI(status);
}

/**
 * Triggers starter dataset generation.
 */
function triggerGenerateStarterDataset(count = 25) {
    if (!confirm(`Generate ${count} Normal and ${count} TB sample chest X-rays?`)) return;

    const btn = document.getElementById("btn-generate-dataset");
    if (btn) {
        btn.disabled = true;
        btn.textContent = "Generating...";
    }

    fetch("/api/generate-starter-data", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ count: count, force: true })
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            alert(data.result.message);
            window.location.reload();
        } else {
            alert("Error: " + data.error);
            if (btn) {
                btn.disabled = false;
                btn.textContent = "Generate Starter Dataset";
            }
        }
    })
    .catch(err => {
        alert("Request error: " + err);
        if (btn) {
            btn.disabled = false;
            btn.textContent = "Generate Starter Dataset";
        }
    });
}
