# Audio Classification with CNN - UrbanSound8K

## Overview
This project demonstrates an end-to-end workflow for classifying urban sounds using a Convolutional Neural Network (CNN) on the UrbanSound8K dataset. The notebook guides you through data exploration, preprocessing, feature extraction, model building, training, evaluation, and Inferance.

## Dataset
- **UrbanSound8K**: [https://www.kaggle.com/datasets/chrisfilo/urbansound8k](https://www.kaggle.com/datasets/chrisfilo/urbansound8k)

## Workflow

### 1. Introduction
- Presents the problem: classifying 10 types of urban sounds (e.g., dog bark, siren, drilling) from short audio clips.
- States the goal: build a robust neural network to recognize and categorize environmental sounds.

### 2. Importing Libraries
- Loads essential libraries for data handling (`pandas`, `numpy`), audio processing (`librosa`), visualization (`matplotlib`, `seaborn`), and deep learning (`torch`, `torchvision`).
- Sets up the device for computation (GPU/CPU).

### 3. Data Loading and Exploration
- Loads the UrbanSound8K metadata CSV and inspects its structure.
- Explores the dataset, including the number of samples, class distribution, and available sound classes.
- Visualizes the number of files per class using bar plots for a quick overview of class balance.

### 4. Data Preprocessing & Feature Extraction
- Defines functions to extract Mel spectrogram features from audio files, which are well-suited for CNN input.
- Handles normalization and padding/truncation to ensure uniform input size.
- Visualizes sample waveforms and spectrograms to understand the data characteristics.

### 5. Dataset Preparation
- Iterates through the dataset, extracting features and labels for each audio file.
- Splits the data into training and testing sets, ensuring stratified sampling for balanced class representation.
- Converts data into PyTorch tensors and wraps them in DataLoader objects for efficient batching.

### 6. Model Architecture
- Implements a deep CNN with multiple convolutional blocks, each consisting of Conv2D, ReLU, BatchNorm, and MaxPooling layers.
- Uses global average pooling and fully connected layers with dropout for regularization.
- The final layer outputs class probabilities for the 10 sound categories.

### 7. Training & Evaluation
- Defines training and evaluation loops, tracking loss and accuracy per epoch.
- Trains the model on the training set and evaluates performance on the test set.
- Visualizes training history (loss and accuracy curves) and confusion matrix to assess model performance and class-wise accuracy.

### 8. Inference
- Provides a function to predict the class of new audio files using the trained model.
- Demonstrates real-time inference capability.

### 9. Results & Visualization
- Presents final accuracy and loss metrics.
- Displays confusion matrix for detailed error analysis.

### 10. Conclusion & Next Steps
- Summarizes the workflow and results.
- Suggests improvements such as advanced architectures, transfer learning, and hyperparameter tuning.
