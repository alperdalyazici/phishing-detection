import json
import pickle
import seaborn as sns
from matplotlib import pyplot as plt
import mlflow
import mlflow.keras
import pandas as pd
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import numpy as np
from tensorflow import keras

#Get the model with the highest recall and accuracy from each experiment
mlflow.set_experiment("Testing")  
with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name("Testing").experiment_id):
    runs = mlflow.search_runs(experiment_ids= mlflow.get_experiment_by_name("CNN").experiment_id)
    best_run = runs.sort_values(by=['metrics.recall', 'metrics.test_accuracy'], ascending=[False, False]).iloc[0]
    best_run_id = best_run.run_id
    best_model_cnn = mlflow.keras.load_model(f"runs:/{best_run_id}/model")

    runs = mlflow.search_runs(experiment_ids= mlflow.get_experiment_by_name("LSTM-CNN").experiment_id)
    best_run = runs.sort_values(by=['metrics.recall', 'metrics.test_accuracy'], ascending=[False, False]).iloc[0]
    best_run_id = best_run.run_id
    best_model_lstm_cnn = mlflow.keras.load_model(f"runs:/{best_run_id}/model")

    runs = mlflow.search_runs(experiment_ids= mlflow.get_experiment_by_name("RNN-hybrid").experiment_id)
    best_run = runs.sort_values(by=['metrics.recall', 'metrics.test_accuracy'], ascending=[False, False]).iloc[0]
    best_run_id = best_run.run_id
    best_model_hybrid = mlflow.keras.load_model(f"runs:/{best_run_id}/model")

    runs = mlflow.search_runs(experiment_ids= mlflow.get_experiment_by_name("RNN-2LSTM").experiment_id)
    best_run = runs.sort_values(by=['metrics.recall', 'metrics.test_accuracy'], ascending=[False, False]).iloc[0]
    best_run_id = best_run.run_id
    best_model_lstm = mlflow.keras.load_model(f"runs:/{best_run_id}/model")

    #Store the models and the tokenizers in a list
    models = [best_model_cnn, best_model_lstm_cnn, best_model_hybrid, best_model_lstm]
    tokenizer_paths = ["CNN.pkl", "LSTM-CNN.pkl", "RNN-hybrid.pkl", "2LSTM.pkl"]
    loaded_tokenizers = []

    for path in tokenizer_paths:
        with open(path, "rb") as tokenizer_file:
            loaded_tokenizers.append(pickle.load(tokenizer_file))

    # Read the new dataset and preprocess it
    new_data = pd.read_csv("data/combined_output.txt", sep="\t", header=None, names=["label", "url"])
    new_data["label"] = new_data["label"].apply(lambda x: 1 if x == "phishing" else 0)

    predictions1 = []
    predictions2 = []
    predictions3 = []
    predictions4 = []

    for i in range(len(models)):
        new_sequences = loaded_tokenizers[i].texts_to_sequences(new_data["url"])
        new_padded_sequences = keras.preprocessing.sequence.pad_sequences(new_sequences, maxlen=models[i].input_shape[1])

        predictions = models[i].predict(new_padded_sequences)
        if i == 0:
            predictions1.extend(predictions)
        elif i == 1:
            predictions2.extend(predictions)
        elif i == 2:
            predictions3.extend(predictions)
        elif i == 3:
            predictions4.extend(predictions)

    # Combine the predictions from all models and calculate the average 
    final_predictions = []
    for p1, p2, p3, p4 in zip(predictions1, predictions2, predictions3, predictions4):
        avg_prediction = (p1 + p2 + p3 + p4) / 4
        final_predictions.append(int(avg_prediction > 0.5))

    # Evaluate the predictions and log the corresponding metrics into MLflow
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

    accuracy = accuracy_score(new_data["label"], final_predictions)
    precision = precision_score(new_data["label"], final_predictions)
    recall = recall_score(new_data["label"], final_predictions)
    f1 = f1_score(new_data["label"], final_predictions)

    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1-score", f1)

    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}')