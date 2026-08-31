# ==========================================
# Image Classification using CNN
# CIFAR-10 Dataset
# Training Script
# ==========================================

import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, models


# ==========================================
# Configuration
# ==========================================

EPOCHS = 10
BATCH_SIZE = 64

MODEL_DIR = "models"
IMAGE_DIR = "images"

os.makedirs(MODEL_DIR, exist_ok=True)
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

print("Loading CIFAR-10 dataset...")

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print("Dataset loaded successfully.")

print("Training images:", X_train.shape)
print("Testing images:", X_test.shape)


# ==========================================
# Prepare Labels
# ==========================================

y_train = y_train.flatten()
y_test = y_test.flatten()


# ==========================================
# Normalize Images
# ==========================================

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

print("Image normalization completed.")


# ==========================================
# Display Sample Images
# ==========================================

plt.figure(figsize=(10, 8))

for i in range(16):

    plt.subplot(4, 4, i + 1)

    plt.imshow(X_train[i])

    plt.title(class_names[y_train[i]])

    plt.axis("off")

plt.tight_layout()

sample_path = os.path.join(
    IMAGE_DIR,
    "sample_images.png"
)

plt.savefig(
    sample_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Sample images saved to:", sample_path)


# ==========================================
# Build CNN Model
# ==========================================

print("Building CNN model...")

model = models.Sequential([

    layers.Input(shape=(32, 32, 3)),

    # Convolutional Block 1
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Convolutional Block 2
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Convolutional Block 3
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    # Flatten
    layers.Flatten(),

    # Fully Connected Layer
    layers.Dense(
        128,
        activation="relu"
    ),

    # Output Layer
    layers.Dense(
        10,
        activation="softmax"
    )
])


# ==========================================
# Compile Model
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Model compiled successfully.")

model.summary()


# ==========================================
# Train Model
# ==========================================

print("Starting model training...")

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2
)


# ==========================================
# Evaluate Model
# ==========================================

print("Evaluating model...")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\nTest Loss:", test_loss)

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)


# ==========================================
# Save Model
# ==========================================

model_path = os.path.join(
    MODEL_DIR,
    "cifar10_cnn_model.keras"
)

model.save(model_path)

print("Model saved to:", model_path)


# ==========================================
# Accuracy Plot
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "CNN Training and Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

accuracy_path = os.path.join(
    IMAGE_DIR,
    "accuracy_plot.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "Accuracy graph saved to:",
    accuracy_path
)


# ==========================================
# Loss Plot
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "CNN Training and Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

loss_path = os.path.join(
    IMAGE_DIR,
    "loss_plot.png"
)

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "Loss graph saved to:",
    loss_path
)


print("\n==========================================")
print("Training Completed Successfully!")
print("==========================================")
