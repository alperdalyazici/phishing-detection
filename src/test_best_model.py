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

#Get the model with the highest recall and accuracy from a specific experiment
mlflow.set_experiment("Testing") 
with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name("Testing").experiment_id):
        
    runs = mlflow.search_runs(experiment_ids= mlflow.get_experiment_by_name("RNN-hybrid").experiment_id)
    best_run = runs.sort_values(by=['metrics.f1_score', 'metrics.test_accuracy'], ascending=[False, False]).iloc[0]
    best_run_id = best_run.run_id
    best_model = mlflow.keras.load_model(f"runs:/{best_run_id}/model")

    with open("RNN-hybrid.pkl", "rb") as tokenizer_file:
        loaded_tokenizer = pickle.load(tokenizer_file)

    # Read the new dataset and preprocess it
    new_data = pd.read_csv("data/combined_output.txt", sep="\t", header=None, names=["label", "url"])
    new_data["label"] = new_data["label"].apply(lambda x: 1 if x == "phishing" else 0)

    # Convert URLs to sequences of integers
    new_sequences = loaded_tokenizer.texts_to_sequences(new_data["url"])
    new_padded_sequences = keras.preprocessing.sequence.pad_sequences(new_sequences, maxlen = 64)

    # Make predictions on the new dataset
    predictions = best_model.predict(new_padded_sequences)
    predictions = (predictions > 0.5).astype(int)

    # Evaluate the predictions using accuracy, precision, recall, and F1-score and log them into MLflow
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(new_data["label"], predictions)
    precision = precision_score(new_data["label"], predictions)
    recall = recall_score(new_data["label"], predictions)
    f1 = f1_score(new_data["label"], predictions)

    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1-score", f1)

    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")

    # Calculate and visualize the confusion matrix
    confusion = confusion_matrix(new_data["label"], predictions)

    # Create a dictionary with the confusion matrix values
    confusion_dict = {
        "true_positive": confusion[1, 1],
        "false_positive": confusion[0, 1],
        "true_negative": confusion[0, 0],
        "false_negative": confusion[1, 0]
    }

    # Create a labels list for the confusion matrix
    confusion_labels = ["True Negative", "False Positive", "False Negative", "True Positive"]

    # Create a matrix for the heatmap
    confusion_matrix_visual = np.array([[confusion_dict["true_negative"], confusion_dict["false_positive"]],
                                        [confusion_dict["false_negative"], confusion_dict["true_positive"]]])

    # Set up the figure and axes for the heatmap
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(confusion_matrix_visual, annot=True, fmt="d", cmap="Blues", cbar=False)

    # Set the tick labels
    ax.set_xticks([0, 1])  # Only two positions: 0 and 1
    ax.set_xticklabels(["Predicted Negative", "Predicted Positive"], rotation=0)
    ax.set_yticks([0, 1])  # Only two positions: 0 and 1
    ax.set_yticklabels(["Actual Negative", "Actual Positive"], rotation=0) 

    for i in range(2):
        for j in range(2):
            color = "white" if i == j else "black"  # Set white color for diagonal numbers, black for others
            ax.text(j + 0.5, i + 0.5, str(confusion_matrix_visual[i, j]), ha="center", va="center", color=color)

    # Add labels to the axes
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Set the title
    plt.title("Confusion Matrix")

    # Show the heatmap
    plt.show()

    confusion_dict = {
        "true_positive": int(confusion[1, 1]),
        "false_positive": int(confusion[0, 1]),
        "true_negative": int(confusion[0, 0]),
        "false_negative": int(confusion[1, 0]),
        "labels": confusion_labels
    }

    # Save the dictionary as a JSON file
    with open("confusion_matrix.json", "w") as json_file:
        json.dump(confusion_dict, json_file, indent=4)

