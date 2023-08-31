# Phishing URL Classification Project

This project focuses on classifying URLs as either phishing (malicious) or legitimate using machine learning techniques. The project involves creating and training various models to achieve accurate classification.

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Building and training the models](#building-and-training-the-models)
- [Hyperparameters](#hyperparameters)
- [Testing Models](#testing-models)
- [Combine Models](#combine-models)

## Introduction

Phishing attacks remain a significant cybersecurity concern. This project aims to address this issue by developing machine learning models to automatically identify phishing URLs. Various techniques, including LSTM and CNN architectures, are employed to achieve accurate classification. This project is developed during my internship as Deep Learning Engineer at DNSSense.

## Project Overview

- The project involves training machine learning models to classify URLs as either phishing or legitimate.
- Different architectural approaches, including LSTM and CNN, are implemented to understand their impact on model performance.
- Model evaluation metrics such as accuracy, precision, recall, and F1-score are used to assess model effectiveness.

## Data

- The project uses two different datasets. The first dataset is a collection of over 36,000 legitimate URLs and over 37,000 phishing URLs. The first dataset is used to train and evaluate the models. The second dataset is a collection of over 250,000 legitimate URLs and 270,000 phishing URLs. The second dataset is used to test the models performance on unseen data.
- The first dataset consists of two json files, one for legitimate URLs and one for phishing URLs. The second dataset is a txt file that has both legitimate and phishing URLs.
- The first dataset is preprocessed and split into training, validation, and testing subsets.
- Features are extracted from URLs, which may include textual, sequential, and tabular data.

## Preprocessing

- After cloning the repository, install the required packages. The project uses Python 3.11.3.
- The data folder contains both datasets and a python file which reads and preporcesses the first dataset.
- The preprocessing file reads the json files and converts them into tuples with binary labels attached to each kind of url. The data is then randomized and split into training, validation, and testing subsets. 

## Building and training the models

- Inside the src folder, model.py contains the code for building and training the models.
- Four different models are implemented: a LSTM model, a CNN model, a LSTM dominant hybrid model, and a CNN dominant hybrid model.
- The models are trained using 27 different hyperparameter combinations and the parameters, as well as the metrics are saved in MLflow.
- The models are trained using the training and validation subsets of the first dataset.
- The models are evaluated using the test set of the first dataset.
- Finally the results of the evaluation are saved in a MLflow.
- To vuild and train your model you can run the script.

## Hyperparameters

- Inside the input folder, there is a python script called hyperparameters.py. This script creates all the possible hyperparameter combinations for the models. The hyperparameters are:
    
- Embedding output dimension
- LSTM units
- Batch size

## Testing Models

- The test_best_model inside the src folder contains the code for getting the best model for a specific experiment in MLflow and testign on the second dataset.
- The script evaluates the performance of the best model on an unseen dataset and prints the accuracy, precision, recall, and F1-score.
- After that the code produces a confusion matrix for the user to see the results.
- To run this script you must have the second dataset in the data folder and the best model saved in your desired experiment in MLflow. After that just enter your experiment name and run the script. 

## Combine Models

- Finally the combine_models.py script inside the src folder contains the code for combining the four models from the experiments.
- The script finds the best model for each experiment and combines them into one model by taking the average of the predictions of the four models.
- The script then evaluates the performance of the combined model on an unseen dataset and prints the accuracy, precision, recall, and F1-score.
- To run this script you must have the second dataset in the data folder and the four best models saved in your desired experiments in MLflow. After that just enter your experiment names and run the script.