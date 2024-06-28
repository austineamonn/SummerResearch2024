import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
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
            'LinearRegression': LinearRegression(),
            'DecisionTree': DecisionTreeRegressor(),
            'RandomForest': RandomForestRegressor(n_estimators=10)
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

    def compute_shap_values(self, model, X, model_name, target_name):
        # Log the data info before passing to SHAP explainer
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            self.log_data_info(X, "Data passed to SHAP")

        if model_name in ['DecisionTree', 'RandomForest']:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, X)
            
        shap_values = explainer(X)

        # Handle different shapes of SHAP values
        if len(shap_values.values.shape) > 2:  # Handle multi-dimensional SHAP values
            reshaped_dim = shap_values.values.shape[-1]
            shap_values_reshaped = shap_values.values.reshape(shap_values.values.shape[0], -1)
            columns = [f"{col}_{i}" for i in range(reshaped_dim) for col in X.columns]
        else:
            shap_values_reshaped = shap_values.values
            columns = X.columns

        shap_df = pd.DataFrame(shap_values_reshaped, columns=columns)
        shap_df["target"] = [target_name] * len(shap_df)
        shap_df["model"] = [model_name] * len(shap_df)
        shap_filename = f"shap_values_basic/{self.RNN_type}/shap_values_{model_name}_{target_name.replace(' ', '_')}.csv"
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
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                # Calculate Errors
                mse = mean_squared_error(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)

                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(f"Results for {name} on {target_columns}: MSE={mse}, MAE={mae}, R2={r2}")

                # Make results_title equal to target_columns if no title was given
                if results_title == None:
                    results_title = target_columns
                
                # Ensure the key is a string and results are stored correctly
                key = f"{name}_{results_title}"
                results[key] = {"MSE": mse, "MAE": mae, "R2": r2}

                # Compute and save SHAP values
                self.compute_shap_values(model, X_test, name, results_title)

        # Individual columns as targets
        for column in self.private_columns + self.utility_columns:
            evaluate_target([column], column)

        # All private_columns as targets
        evaluate_target(self.private_columns, "private_columns")

        # All utility_columns as targets
        evaluate_target(self.utility_columns, "utility_columns")

        # All columns as targets
        evaluate_target(self.private_columns + self.utility_columns, "all_columns")

        if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(f"Results Dictionary: {results}")

        return results

    def save_results_to_csv(self, results):
        rows = []
        for model_target, scores in results.items():
            if not isinstance(scores, dict):
                logging.error(f"Expected dict for scores, got {type(scores)} for {model_target}")
                continue
            if 'MSE' not in scores or 'MAE' not in scores or 'R2' not in scores:
                logging.error(f"Missing expected keys in scores for {model_target}: {scores}")
                continue
            rows.append({
                'Model_Target': model_target,
                'MSE': scores['MSE'],
                'MAE': scores['MAE'],
                'R2': scores['R2']
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
    RNN_model_list = ['GRU1'] # ['LSTM1', 'Simple1']

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
