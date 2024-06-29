import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, log_loss, classification_report,
    balanced_accuracy_score
)
import shap
import os
import sys
import logging
import tensorflow as tf
import json
import time
import joblib
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.layers import Input, Dropout # type: ignore
from sklearn.preprocessing import OneHotEncoder

# Add the SummerResearch2024 directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class CalculateTradeoffs:
    def __init__(self, config, df, RNN_type):
        # Set up logging
        logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])

        # Get RNN model type
        self.RNN_type = RNN_type

        # Data
        self.data = df
        self.private_columns = config["calculating_tradeoffs"]["privacy_cols"]
        self.other_columns = config["privacy"]["X_list"]
        self.encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.impute_values()

    def impute_values(self):
        # Impute missing values and ensure the datatype stays a pandas dataframe
        self.data = pd.DataFrame(self.imputer.fit_transform(self.data), columns=self.data.columns, index=self.data.index)

    def log_data_info(self, data, name):
        logging.debug(f"{name} shape: {data.shape}")
        logging.debug(f"{name} mean: {data.mean().to_dict()}")
        logging.debug(f"{name} std: {data.std().to_dict()}")

    def create_model(self, input_shape, output_shape):
        model = Sequential()
        model.add(Input(shape=(input_shape,)))  # Adjust the shape as needed
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))  # Add dropout with 20% rate
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))  # Add dropout with 20% rate
        if output_shape == 1:
            model.add(Dense(output_shape, activation='sigmoid'))  # Use sigmoid for binary classification
            model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])  # Use binary_crossentropy for binary classification
        else:
            model.add(Dense(output_shape, activation='softmax'))
            model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def compute_shap_values(self, model, X, target_name):
        # Explainer for SHAP values using DeepExplainer
        explainer = shap.DeepExplainer(model, X)
        shap_values = explainer.shap_values(X)
        
        shap_values_reshaped = shap_values.values
        columns = X.columns

        # Note how there is a '_' instead of a space
        model_name = 'Neural_Network'


        shap_df = pd.DataFrame(shap_values_reshaped, columns=columns)
        shap_df["target"] = [target_name] * len(shap_df)
        shap_df["model"] = [model_name] * len(shap_df)
        shap_filename = f"shap_values_NN_basic/{self.RNN_type}/{model_name}/shap_values_{model_name}_{target_name.replace(' ', '_')}.csv"
        shap_df.to_csv(shap_filename, index=False)

    def train_and_evaluate(self):
        results = {}

        # Helper function to evaluate a single target
        def evaluate_target(target_columns, results_title=None):

            X = self.data[self.other_columns]
            y = self.data[target_columns]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            input_shape = X_train.shape[1]
        
            # One-hot encode the target labels
            encoder = OneHotEncoder(sparse_output=False)
            y_train = encoder.fit_transform(y_train)
            y_test = encoder.transform(y_test)
            
            output_shape = y_train.shape[1]

            # Create and train model
            model = self.create_model(input_shape, output_shape)
            start_time = time.time()
            model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
            end_time = time.time()
            training_time = end_time - start_time

            y_pred = model.predict(X_test)
            y_test_classes = y_test.argmax(axis=1)
            y_pred_classes = y_pred.argmax(axis=1)
            
            # Calculate Accuracy Values
            accuracy = accuracy_score(y_test_classes, y_pred_classes)
            precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
            recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
            f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
            balanced_accuracy = balanced_accuracy_score(y_test_classes, y_pred_classes)
            report = classification_report(y_test_classes, y_pred_classes, output_dict=True)
            conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
            log_loss_value = log_loss(y_test, y_pred, labels=list(range(output_shape)))

            shap_values = self.compute_shap_values(model, X_test)

            # Make results_title equal to target_columns if no title was given
            if results_title == None:
                results_title = target_columns
            results_title = results_title.title()
            name = 'Neural Network'
            
            key = f"{name}_{results_title}"
            results[key] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'Balanced Accuracy': balanced_accuracy,
                'Log Loss': log_loss_value,
                'Training Time': training_time,
                'Classification Report': report,
                'Confusion Matrix': conf_matrix.tolist(),
                'SHAP Values': shap_values
            }

            # Save the trained model
            self.save_model(model, name)
        
        # Individual columns as targets
        for column in self.private_columns:
            evaluate_target([column], column)

        # All private_columns as targets
        evaluate_target(self.private_columns, "Private Columns")

        if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(f"Results Dictionary: {results}")

        return results
    
    def save_model(self, model, model_name, target_name):
        model_dir = 'saved_models'
        os.makedirs(model_dir, exist_ok=True)
        file_path = os.path.join(model_dir, f"{model_name}_{target_name}.joblib")
        joblib.dump(model, file_path)
        logging.info(f"Saved model {model_name} for target {target_name} at {file_path}")

    def save_results_to_files(self, results):
        base_path = f"tradeoff_NN_results_basic/{self.RNN_type}"
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        
        summary_rows = []
        
        for model_target, scores in results.items():
            if not isinstance(scores, dict):
                logging.error(f"Expected dict for scores, got {type(scores)} for {model_target}")
                continue

            summary_rows.append({
                'Model_Target': model_target,
                'Accuracy': scores['Accuracy'],
                'Precision': scores['Precision'],
                'Recall': scores['Recall'],
                'F1 Score': scores['F1 Score'],
                'Balanced Accuracy': scores['Balanced Accuracy'],
                'Log Loss': scores['Log Loss'],
                'Training Time': scores['Training Time']
            })

            report_file_path = os.path.join(base_path, f"{model_target}_classification_report.json")
            with open(report_file_path, 'w') as f:
                json.dump(scores['Classification Report'], f, indent=4)

            conf_matrix_file_path = os.path.join(base_path, f"{model_target}_confusion_matrix.csv")
            pd.DataFrame(scores['Confusion Matrix']).to_csv(conf_matrix_file_path, index=False)

        summary_df = pd.DataFrame(summary_rows)
        summary_file_path = os.path.join(base_path, "model_evaluation_summary.csv")
        summary_df.to_csv(summary_file_path, index=False)

if __name__ == "__main__":
    # Import necessary dependencies
    from config import load_config

    # Load configuration
    config = load_config()

    # List of RNN models to run
    RNN_model_list = ['GRU1', 'LSTM1', 'Simple1']

    for RNN_model in RNN_model_list:
        logging.info(f"Starting {RNN_model}")
        # Get Data Paths
        path_name = f"completely preprocessed {RNN_model} data path"
        preprocessed_dataset_path = config["running_model"][path_name]

        # Load combined dataset
        preprocessed_dataset_df = pd.read_csv(preprocessed_dataset_path)

        # Instantiate the class and run the training and evaluation
        predictor = CalculateTradeoffs(config, preprocessed_dataset_df, RNN_model)
        results = predictor.train_and_evaluate()

        # Save the results to a CSV file
        predictor.save_results_to_files(results)
