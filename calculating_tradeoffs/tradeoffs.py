import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os
import sys

# Add the SummerResearch2024 directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class CalculateTradeoffs:
    def __init__(self, config, df):
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
        # Impute missing values
        self.data[self.other_columns] = self.imputer.fit_transform(self.data[self.other_columns])

    def train_and_evaluate(self):
        results = {}
        for target_column in self.utility_columns:
            results[target_column] = {}
            X = self.data[self.other_columns]
            y = self.data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            for model_name, model in self.models.items():
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                mse = mean_squared_error(y_test, predictions)
                results[target_column][model_name] = mse
        return results
    
    def save_results_to_csv(self, results, filename):
        results_df = pd.DataFrame(results)
        results_df.to_csv(filename, index=False)

if __name__ == "__main__":
    # Import necessary dependencies
    from config import load_config

    # Load configuration
    config = load_config()

    # Get Data Paths
    private_col_path = config["running_model"]["private columns path"]
    preprocessed_dataset_path = config["running_model"]["preprocessed with RNN data path"]
    model_comparison_path = config["running_model"]["model comparison path"]

    # Load the datasets and combine them
    private_columns_df = pd.read_csv(private_col_path)
    preprocessed_dataset_df = pd.read_csv(preprocessed_dataset_path)
    concatenated_df = pd.concat([private_columns_df, preprocessed_dataset_df], axis=1)

    # Define private and utility columns
    private_columns = config["calculating_tradeoffs"]["privacy_cols"]
    utility_columns =config["privacy"]["Xu_list"]

    # Instantiate the class and run the training and evaluation
    predictor = CalculateTradeoffs(config, concatenated_df)
    results = predictor.train_and_evaluate()

    # Save the results to a CSV file
    results_csv_path = 'model_evaluation_results.csv'
    predictor.save_results_to_csv(results, results_csv_path)

