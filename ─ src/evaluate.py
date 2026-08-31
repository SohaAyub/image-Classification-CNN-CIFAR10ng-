# ==========================================
# Image Classification using CNN
# CIFAR-10 Dataset
# Evaluation Script
# ==========================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)


# ==========================================
# Configuration
# ==========================================

VOCAB_SIZE = 10

MODEL_PATH = "models/cifar10_cnn_model.keras"

IMAGE_DIR = "images"

os.makedirs(IMAGE_DIR, exist_ok=True)


# ==========================================
# Class Names
# ==========================================

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# ==========================================
# Load Dataset
# ==========================================

print("Loading CIFAR-10 test dataset...")

(_, _), (X_test, y_test) = cifar10.load_data()

y_test = y_test.flatten()

X_test = X_test.astype(
    "float32"
) / 255.0


# ==========================================
# Load Model
# ==========================================

print("Loading trained CNN model...")

model = load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ==========================================
# Evaluate Model
# ==========================================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\n==========================================")
print("MODEL EVALUATION")
print("==========================================")

print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ==========================================
# Generate Predictions
# ==========================================

print("\nGenerating predictions...")

predictions = model.predict(
    X_test,
    verbose=1
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)


# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_test,
    predicted_labels
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title(
    "CNN Confusion Matrix - CIFAR-10"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)

plt.tight_layout()

confusion_path = os.path.join(
    IMAGE_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "Confusion matrix saved to:",
    confusion_path
)


# ==========================================
# Classification Report
# ==========================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

report = classification_report(
    y_test,
    predicted_labels,
    target_names=class_names
)

print(report)


# ==========================================
# Correct / Wrong Predictions
# ==========================================

correct_predictions = np.sum(
    predicted_labels == y_test
)

wrong_predictions = np.sum(
    predicted_labels != y_test
)

print("\n==========================================")
print("PREDICTION SUMMARY")
print("==========================================")

print(
    "Total Images:",
    len(y_test)
)

print(
    "Correct Predictions:",
    correct_predictions
)

print(
    "Wrong Predictions:",
    wrong_predictions
)
