# Image Classification Using CNN on CIFAR-10

## Overview

This project implements an image classification system using a Convolutional Neural Network (CNN) with TensorFlow and Keras.

The model is trained on the CIFAR-10 dataset to classify images into 10 different categories. The project covers the complete Deep Learning workflow, including dataset loading, image preprocessing, CNN model development, model training, evaluation, prediction, visualization, and error analysis.

The objective of this project is to gain practical experience in Computer Vision and Deep Learning by developing an end-to-end image classification model.

---

## Project Objectives

The main objectives of this project are:

- Understand the fundamentals of image classification.
- Work with the CIFAR-10 image dataset.
- Perform image preprocessing and normalization.
- Build a Convolutional Neural Network.
- Train a CNN model using TensorFlow and Keras.
- Evaluate model performance on unseen test data.
- Generate accuracy and loss visualizations.
- Create a confusion matrix.
- Generate classification reports.
- Analyze correct and incorrect predictions.
- Save the trained model for future use.

---

## Dataset

This project uses the CIFAR-10 dataset.

CIFAR-10 contains 60,000 color images divided into 10 different classes.

| Property | Value |
|---|---|
| Dataset | CIFAR-10 |
| Total Images | 60,000 |
| Training Images | 50,000 |
| Testing Images | 10,000 |
| Image Size | 32 × 32 pixels |
| Channels | 3 (RGB) |
| Number of Classes | 10 |

### Classes

The dataset contains the following classes:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

The dataset is automatically downloaded using TensorFlow Keras.

```python
from tensorflow.keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
