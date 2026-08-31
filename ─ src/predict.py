# ==========================================
# Image Classification using CNN
# CIFAR-10 Dataset
# Prediction Script
# ==========================================

import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model


# ==========================================
# Configuration
# ==========================================

MODEL_PATH = "models/cifar10_cnn_model.keras"

IMAGE_DIR = "images"

os.makedirs(
    IMAGE_DIR,
    exist_ok=True
)


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
# Load Test Dataset
# ==========================================

print("Loading CIFAR-10 dataset...")

(_, _), (X_test, y_test) = cifar10.load_data()

y_test = y_test.flatten()

X_test = X_test.astype(
    "float32"
) / 255.0


# ==========================================
# Load Model
# ==========================================

print("Loading trained model...")

model = load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ==========================================
# Generate Predictions
# ==========================================

predictions = model.predict(
    X_test,
    verbose=1
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)


# ==========================================
# Display Predictions
# ==========================================

plt.figure(figsize=(12, 10))

for i in range(16):

    plt.subplot(4, 4, i + 1)

    plt.imshow(X_test[i])

    actual = class_names[
        y_test[i]
    ]

    predicted = class_names[
        predicted_labels[i]
    ]

    confidence = (
        predictions[i][predicted_labels[i]]
        * 100
    )

    plt.title(
        f"Actual: {actual}\n"
        f"Predicted: {predicted}\n"
        f"Confidence: {confidence:.1f}%"
    )

    plt.axis("off")


plt.tight_layout()


# ==========================================
# Save Prediction Visualization
# ==========================================

prediction_path = os.path.join(
    IMAGE_DIR,
    "predictions.png"
)

plt.savefig(
    prediction_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "Prediction visualization saved to:",
    prediction_path
)


# ==========================================
# Print Individual Predictions
# ==========================================

print("\n==========================================")
print("SAMPLE PREDICTIONS")
print("==========================================")

for i in range(10):

    actual = class_names[
        y_test[i]
    ]

    predicted = class_names[
        predicted_labels[i]
    ]

    confidence = (
        predictions[i][predicted_labels[i]]
        * 100
    )

    print(
        f"{i + 1}. "
        f"Actual: {actual} | "
        f"Predicted: {predicted} | "
        f"Confidence: {confidence:.2f}%"
    )
