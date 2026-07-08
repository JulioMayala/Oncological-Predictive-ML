# Oncological Predictive ML Model (Deep Learning) 🔬

## Overview
This repository contains an end-to-end Deep Learning pipeline designed to predict cancer presence, specific diagnoses, and cancer subtypes based on clinical and symptomatic datasets. This project bridges my clinical laboratory background with advanced software engineering and neural network architectures.

## Technical Stack
* **Languages & Frameworks:** Python, TensorFlow / Keras
* **Data Processing:** Pandas, NumPy, Scikit-Learn (StandardScaler)
* **Architecture:** Multi-Layer Perceptrons (MLP) / Sequential Neural Networks

## Pipeline & Methodology
1. **Dynamic Feature Engineering:** 
   * Developed custom Python functions to dynamically parse and one-hot encode variable-length symptom strings.
   * Applied targeted transformations for blood types (ABO/Rh factor) and medical treatments, while dropping non-clinical noise (e.g., IDs, zip codes).
2. **Deep Learning Architecture:** 
   * Designed three distinct `Sequential` neural networks to handle different classification layers:
     - Binary classification for cancer presence (`sigmoid` activation, `binary_crossentropy`).
     - Multi-class classification for specific diagnosis and subtypes (`softmax` activation, `sparse_categorical_crossentropy`).
   * Architecture features hidden Dense layers (64 -> 32 -> 16 nodes) utilizing `ReLU` activation functions and the `Adam` optimizer.
3. **Training & Inference:** 
   * Models trained with a 80/20 train-test split.
   * Deployed `joblib` and `.h5` model serialization to export the trained weights and scalers for seamless new-data inference.

## Results
* The binary classification model achieved an **~80-84% accuracy** during validation.
* Successfully outputted robust CSV predictions chaining presence, diagnosis, and subtype.

*Note: The dataset utilized in this repository is purely synthetic/mock data generated for demonstration purposes and contains no real patient information.*
