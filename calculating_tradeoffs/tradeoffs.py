import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import sys
import logging

# Add the SummerResearch2024 directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class CalculateTradeoffs:
    def __init__(self, config, df):
        # Set up logging
        logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])

        # Data
        self.data = df
        self.private_columns = config["calculating_tradeoffs"]["privacy_cols"]
        self.utility_columns = config["privacy"]["Xu_list"]
        self.other_columns = config["privacy"]["X_list"]
        self.models = {
            'LinearRegression': LinearRegression(),
            'DecisionTree': DecisionTreeRegressor(),
            'RandomForest': RandomForestRegressor(n_estimators=10)  # Reduced number of estimators for quicker run
        }
        self.encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.impute_values()

    def impute_values(self):
        # Impute missing values and ensure the datatype stays a pandas dataframe
        self.data = pd.DataFrame(self.imputer.fit_transform(self.data), columns=self.data.columns, index=self.data.index)

    def train_and_evaluate(self):
        results = {}

        # Helper function to evaluate a single target
        def evaluate_target(target_columns):
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug("The target(s): %s", target_columns)
            X = self.data[self.other_columns]
            X = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)  # Impute missing values in features
            print(len(target_columns))
            if len(target_columns) == 1:
                y = self.data[target_columns].values.ravel()  # Flatten the target variable but only if there is just one column being targeted
            else:
                y = self.data[target_columns]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            target_results = {}
            for model_name, model in self.models.items():
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                mse = mean_squared_error(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                target_results[model_name] = {'MSE': mse, 'MAE': mae, 'R^2': r2}
            return target_results

        # Individual columns as targets
        for column in self.private_columns + self.utility_columns:
            results[column] = evaluate_target([column])

        # All private_columns as targets
        results["all_private_columns"] = evaluate_target(self.private_columns)

        # All utility_columns as targets
        results["all_utility_columns"] = evaluate_target(self.utility_columns)

        # All columns as targets
        results["all_columns"] = evaluate_target(self.private_columns + self.utility_columns)

        return results

    def save_results_to_csv(self, results, filename):
        # Flatten the results for better CSV formatting
        flattened_results = []
        for target, metrics in results.items():
            for model, scores in metrics.items():
                flattened_results.append({
                    'Target': target,
                    'Model': model,
                    'MSE': scores['MSE'],
                    'MAE': scores['MAE'],
                    'R^2': scores['R^2']
                })
        results_df = pd.DataFrame(flattened_results)
        results_df.to_csv(filename, index=False)

if __name__ == "__main__":
    # Import necessary dependencies
    from config import load_config

    # Load configuration
    config = load_config()

    # Get Data Paths
    preprocessed_dataset_path = config["running_model"]["completely preprocessed data path"]
    model_comparison_path = config["running_model"]["model comparison path"]

    # Load combined dataset
    preprocessed_dataset_df = pd.read_csv(preprocessed_dataset_path)

    # Define private and utility columns
    private_columns = config["calculating_tradeoffs"]["privacy_cols"]
    utility_columns =config["privacy"]["Xu_list"]

    # Instantiate the class and run the training and evaluation
    predictor = CalculateTradeoffs(config, preprocessed_dataset_df)
    results = predictor.train_and_evaluate()

    # Save the results to a CSV file
    results_csv_path = 'model_evaluation_results.csv'
    predictor.save_results_to_csv(results, results_csv_path)

