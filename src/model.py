import pickle
import mlflow
import mlflow.keras
import tensorflow as tf
import numpy as np
from tensorflow import keras
import sys
import os
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataprocess import DataProcess
from input.hyperparameters import hyperparameter_configs

class URLClassifier:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.model = None
        self.tokenizer = None
        self.max_sequence_length = None
    

    #Build 4 different models to train and evaluate the performance
    def build_cnn_model(self, tokenizer, embedding_output_dim, lstm_units):
        model = keras.Sequential()
        model.add(keras.layers.Embedding(input_dim=len(self.tokenizer.word_index) + 1, output_dim=embedding_output_dim, input_length=self.max_sequence_length))
        model.add(keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu'))
        model.add(keras.layers.MaxPooling1D(pool_size=2))
        model.add(keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu'))
        model.add(keras.layers.GlobalMaxPooling1D())
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model
    
    def build__cnn_dominant_hybrid_model(self, tokenizer, embedding_output_dim, lstm_units):
        model = keras.Sequential()
        model.add(keras.layers.Embedding(input_dim=len(self.tokenizer.word_index) + 1, output_dim=embedding_output_dim, input_length=self.max_sequence_length))
        model.add(keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu'))
        model.add(keras.layers.MaxPooling1D(pool_size=4))
        model.add(keras.layers.LSTM(lstm_units, return_sequences=True))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.LSTM(lstm_units))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(1, activation='sigmoid')) 
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def build_lstm_dominant_hybrid_model(self, tokenizer, embedding_output_dim, lstm_units):
        model = keras.Sequential()
        model.add(keras.layers.Embedding(input_dim=len(self.tokenizer.word_index) + 1, output_dim=embedding_output_dim, input_length=self.max_sequence_length))
        model.add(keras.layers.LSTM(lstm_units, return_sequences=True)) 
        model.add(keras.layers.LSTM(lstm_units)) 
        model.add(keras.layers.Conv1D(filters = 64, kernel_size = 5, activation='relu')) 
        model.add(keras.layers.MaxPooling1D(pool_size=4)) 
        model.add(keras.layers.GlobalMaxPooling1D())  
        model.add(keras.layers.Dense(1, activation='sigmoid'))  
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def build_lstm_model(self, tokenizer, embedding_output_dim, lstm_units):
        model = keras.Sequential()
        model.add(keras.layers.Embedding(input_dim=len(self.tokenizer.word_index)+1, output_dim=embedding_output_dim, input_length=self.max_sequence_length))
        model.add(keras.layers.LSTM(lstm_units, return_sequences=True))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.LSTM(lstm_units))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(1, activation='sigmoid')) 
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, embedding_output_dim, lstm_units, epochs, batch_size):
        mlflow.set_experiment("CNN")  # Set the experiment name to be used in MLflow
        with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name("CNN").experiment_id):
            
            data_processor = self.data_processor
            
            # Log hyperparameters into MLflow
            mlflow.log_param("embedding_output_dim", embedding_output_dim)
            mlflow.log_param("lstm_units", lstm_units)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)

            # Combine all data for training
            training_data = data_processor.train_data
            validation_data = data_processor.val_data
            
            # Split the features and labels
            X_train, y_train = data_processor.get_features_labels(training_data)
            X_val, y_val = data_processor.get_features_labels(validation_data)  # Separate validation data

            # Convert URLs to integer sequences using a tokenizer
            self.tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
            self.tokenizer.fit_on_texts(X_train)
            
            # Convert URLs to sequences of integers
            X_train_sequences = self.tokenizer.texts_to_sequences(X_train)
            X_val_sequences = self.tokenizer.texts_to_sequences(X_val)

            max_sequence_length = 64
        
            # Pad the sequences to ensure they have the same length
            X_train_padded_sequences = keras.preprocessing.sequence.pad_sequences(X_train_sequences, maxlen=64)
            X_val_padded_sequences = keras.preprocessing.sequence.pad_sequences(X_val_sequences, maxlen=64)
            
            # Save the tokenizer for later use in evaluation
            with open("tokenizer.pkl", "wb") as tokenizer_file:
                pickle.dump(self.tokenizer, tokenizer_file)

            # Create and compile the model
            self.model = self.build_cnn_model(self.tokenizer, embedding_output_dim, lstm_units)
            
            # Log the model summary into MLflow
            model_summary = []
            self.model.summary(print_fn=lambda x: model_summary.append(x))
            model_summary = '\n'.join(model_summary)
            mlflow.log_text(model_summary, "model_summary.txt")

            train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

            # Train the model
            for epoch in range(epochs):
                self.model.fit(X_train_padded_sequences, y_train, epochs=1, batch_size=batch_size, validation_data=(X_val_padded_sequences, y_val))

                train_loss = self.model.history.history["loss"][0]
                train_accuracy = self.model.history.history["accuracy"][0]
                val_loss = self.model.history.history["val_loss"][0]
                val_accuracy = self.model.history.history["val_accuracy"][0]

                
                train_losses.append(train_loss)
                train_accuracies.append(train_accuracy)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)

                # Log metrics for each epoch
                mlflow.log_metric("train_loss_epoch_" + str(epoch), train_loss)
                mlflow.log_metric("train_accuracy_epoch_" + str(epoch), train_accuracy)
                mlflow.log_metric("val_loss_epoch_" + str(epoch), val_loss)
                mlflow.log_metric("val_accuracy_epoch_" + str(epoch), val_accuracy)
            

            # Save the model for later use in evaluation
            mlflow.keras.log_model(self.model, "model")
            mlflow.log_artifact("tokenizer.pkl")
            self.max_sequence_length = max_sequence_length

            # Seperate the features and labels for test data and convert URLs to sequences of integers
            X_test, y_test = data_processor.get_features_labels(data_processor.test_data)
            X_test_sequences = self.tokenizer.texts_to_sequences(X_test)

            # Pad the sequences for test data
            X_test_padded_sequences = keras.preprocessing.sequence.pad_sequences(X_test_sequences, maxlen=64)

            # Evaluate the model
            test_loss, test_accuracy = self.model.evaluate(X_test_padded_sequences, y_test)
            print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
            
            # Make predictions on test data and convert the probabilities into binary predictions
            y_pred = self.model.predict(X_test_padded_sequences)
            y_pred = (y_pred > 0.5).astype(int)
            
            # Calculate recall, precision, and F1-score for the predictions 
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            print(f'Precision: {precision}, Recall: {recall}, F1-score: {f1}')

            mlflow.log_metric("test_loss", test_loss)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

data_processor = DataProcess()
data_processor.read_legit_data()
data_processor.read_phish_data()
data_processor.shuffle_data()
data_processor.split_data()
url_classifier = URLClassifier(data_processor)

for config in hyperparameter_configs:
    url_classifier.train_model(**config)