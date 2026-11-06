Automotive Voice Recognition System (TinyML)

Technologies: Python, PyTorch, ESP32-S3, MFCC, Deep Neural Networks (DNN)

Project Overview

This project focuses on developing a lightweight voice recognition system for automotive applications using TinyML concepts. The system uses a Deep Neural Network (DNN) to recognize 11 predefined voice commands designed to control automotive functions.

The main goal is to develop an efficient speech recognition model that can run on resource-constrained embedded devices, enabling voice-based interaction with automotive systems.

Key Features

Deep Learning Model: Development of a Deep Neural Network (DNN) trained to recognize 11 automotive-related voice commands.
Feature Extraction: Implementation of Mel-Frequency Cepstral Coefficients (MFCC) to extract relevant speech features.
Model Optimization: Hyperparameter tuning and feature optimization, achieving the best performance with 20 MFCC coefficients.
Performance: Achieved 91.35% accuracy in local testing.
Embedded Deployment: Model currently in the deployment and optimization phase for the ESP32-S3 microcontroller.
Technologies Used
Python – Model development and experimentation.
PyTorch – Neural network implementation and training.
MFCC – Audio feature extraction.
ESP32-S3 – Target microcontroller for embedded deployment.
TinyML – Machine learning implementation for resource-constrained devices.
Results

The model achieved the following results during local testing:

Metric	Result

Recognized Commands	11
MFCC Coefficients	20
Local Test Accuracy	91.35%
Target Hardware	ESP32-S3
Deployment Status	In Progress
Current Development Status

The neural network has been developed and evaluated through local testing. Current efforts are focused on optimizing the model for embedded deployment, considering the computational and memory constraints of the ESP32-S3.

Project Goals

Develop reliable voice recognition for automotive applications.
Optimize neural network performance for embedded hardware.
Reduce computational and memory requirements.
Enable real-time voice command recognition using TinyML.
GitHub: Project Repository

This project is part of an ongoing development effort in embedded artificial intelligence, TinyML, and voice-controlled automotive systems
