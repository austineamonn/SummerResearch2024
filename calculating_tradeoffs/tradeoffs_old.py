import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import shap
import time
import joblib
import numpy as np
import os
import sys
import logging

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
        self.utility_columns = config["privacy"]["Xu_list"]
        self.other_columns = config["privacy"]["X_list"]
        self.models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(n_estimators=10)
        }
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

    def compute_shap_values(self, model, X_train, X_test, model_name, target_name, num_background_samples=1000, num_test_samples=200, batch_size=100):
        # Log the data info before passing to SHAP explainer
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            self.log_data_info(X_test, "Data passed to SHAP")

        # Select a subset of background samples
        if len(X_train) > num_background_samples:
            background_samples = X_train.sample(num_background_samples, random_state=0)
        else:
            background_samples = X_train

        # Select a subset of test samples
        if len(X_test) > num_test_samples:
            test_samples = X_test.sample(num_test_samples, random_state=0)
        else:
            test_samples = X_test

        # Ensure the predicted classes are computed on the same test samples
        predicted_classes = model.predict(test_samples)

        # Choose the appropriate SHAP explainer
        if model_name in ['Decision Tree', 'Random Forest']:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, background_samples)

        # Batch processing for SHAP values
        shap_values_list = []
        for i in range(0, len(test_samples), batch_size):
            batch_samples = test_samples.iloc[i:i+batch_size]
            shap_values = explainer(batch_samples)
            shap_values_list.append(shap_values.values)

        # Combine all the SHAP values
        shap_values_combined = np.concatenate(shap_values_list, axis=0)

        if shap_values_combined.ndim == 2:  # For 2D SHAP values
            shap_df = pd.DataFrame(shap_values_combined, columns=[f'shap_feature_{i}' for i in range(shap_values_combined.shape[1])])
        elif shap_values_combined.ndim == 3:  # For 3D SHAP values (multi-class classification)
            shap_df = pd.DataFrame(shap_values_combined.reshape(shap_values_combined.shape[0], -1), 
                                   columns=[f'shap_class_{j}_feature_{i}' for j in range(shap_values_combined.shape[1]) for i in range(shap_values_combined.shape[2])])

        shap_df = pd.concat([test_samples.reset_index(drop=True), shap_df], axis=1)

        # Add predicted classes
        shap_df['predicted_class'] = predicted_classes

        # Remove space in model name to not create problems with file saving
        model_name_for_files = model_name.replace(' ', '_')

        # Add target and model name for context
        shap_df["target"] = [target_name] * len(shap_df)
        shap_df["model"] = [model_name] * len(shap_df)

        shap_filename = f"shap_values_basic/{self.RNN_type}/{model_name_for_files}/shap_values_{model_name_for_files}_{target_name.replace(' ', '_')}.csv"
        shap_df.to_csv(shap_filename, index=False)

    def train_and_evaluate(self):
        results = {}

        # Helper function to evaluate a single target
        def evaluate_target(target_columns, results_title=None):
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug("The target(s): %s", target_columns)
            X = self.data[self.other_columns]
            X = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)  # Impute missing values in features
            if len(target_columns) == 1:
                y = self.data[target_columns].values.ravel()  # Flatten the target variable but only if there is just one column being targeted
            else:
                y = self.data[target_columns]

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                # Log the training data info
                self.log_data_info(X, "Training data")
            
            # Test-Train split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Run Each model
            for name, model in self.models.items():
                logging.info(f"Running {name} for {results_title}")
                start_time = time.time()
                model.fit(X_train, y_train)
                end_time = time.time()
                training_time = end_time - start_time
                y_pred = model.predict(X_test)

                # Calculate Errors
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse) # Root Mean Squared Error
                epsilon = 1e-10  # A small number to avoid division by zero
                mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test == 0, epsilon, y_test))) * 100 # Mean Absolute Percentage Error
                medae = np.median(np.abs(y_test - y_pred)) # Median Absolute Error
                explained_variance = explained_variance_score(y_test, y_pred)
                mbd = np.mean(y_test - y_pred)  # Mean Bias Deviation

                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(f"Results for {name} on {target_columns}: MSE={mse}, MAE={mae}, R2={r2}")

                # Make results_title equal to target_columns if no title was given
                if results_title == None:
                    results_title = target_columns
                results_title = results_title.title()
                
                # Ensure the key is a string and results are stored correctly
                key = f"{name}_{results_title}"
                results[key] = {
                    'MSE': mse,
                    'MAE': mae,
                    'R2': r2,
                    'RMSE': rmse,
                    'MAPE': mape,
                    'MedAE': medae,
                    'Explained Variance': explained_variance,
                    'MBD': mbd,
                    'Training Time': training_time
                    }

                # Compute and save SHAP values
                self.compute_shap_values(model, X_train, X_test, name, results_title)

                # Save the trained model
                self.save_model(model, name, results_title)

        # Individual columns as targets
        for column in self.private_columns + self.utility_columns:
            evaluate_target([column], column)

        # All private_columns as targets
        evaluate_target(self.private_columns, "Private Columns")

        # All utility_columns as targets
        evaluate_target(self.utility_columns, "Utility Columns")

        # All columns as targets
        evaluate_target(self.private_columns + self.utility_columns, "All Columns")

        if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(f"Results Dictionary: {results}")

        return results
    
    def save_model(self, model, model_name, target_name):
        model_dir = 'models_basic'
        os.makedirs(model_dir, exist_ok=True)
        file_path = os.path.join(model_dir, f"{model_name}_{target_name}.joblib")
        joblib.dump(model, file_path)
        logging.info(f"Saved model {model_name} for target {target_name} at {file_path}")

    def save_results_to_csv(self, results):
        rows = []
        for model_target, scores in results.items():
            if not isinstance(scores, dict):
                logging.error(f"Expected dict for scores, got {type(scores)} for {model_target}")
                continue
            required_keys = ['MSE', 'MAE', 'R2', 'RMSE', 'MAPE', 'MedAE', 'Explained Variance', 'MBD', 'Training Time']
            missing_keys = [key for key in required_keys if key not in scores]
            if missing_keys:
                logging.error(f"Missing expected keys in scores for {model_target}: {missing_keys}")
                continue
            rows.append({
                'Model_Target': model_target,
                'MSE': scores['MSE'],
                'MAE': scores['MAE'],
                'R2': scores['R2'],
                'RMSE': scores['RMSE'],
                'MAPE': scores['MAPE'],
                'MedAE': scores['MedAE'],
                'Explained Variance': scores['Explained Variance'],
                'MBD': scores['MBD'],
                'Training Time': scores['Training Time']
            })
        df = pd.DataFrame(rows)
        file_path = f"tradeoff_results_basic/model_evaluation_results_{self.RNN_type}.csv"
        df.to_csv(file_path, index=False)

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
        predictor.save_results_to_csv(results)
