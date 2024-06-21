import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import os
import sys
import logging
import matplotlib.pyplot as plt
import numpy as np

# Add the SummerResearch2024 directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class FeatureImportanceAnalyzer:
    def __init__(self, config, data, name):
        # Set up logging
        logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])

        self.dir_path = '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_preprocessing/feature_importance_' + name
        self.features = data.drop(columns=['career aspirations', 'future topics'])
        self.target_career = data['career aspirations']
        self.target_future = data['future topics']
        self.feature_names = config["privacy"]["X_list"]
        self.imputer = SimpleImputer(strategy='mean')
        self.model_career = RandomForestRegressor(random_state=42)
        self.model_future = RandomForestRegressor(random_state=42)
        self.feature_importances_career = None
        self.feature_importances_future = None

    def impute_data(self):
        self.features_imputed = self.imputer.fit_transform(self.features)

    def train_models(self):
        X_train_career, X_test_career, y_train_career, y_test_career = train_test_split(
            self.features_imputed, self.target_career, test_size=0.2, random_state=42)
        X_train_future, X_test_future, y_train_future, y_test_future = train_test_split(
            self.features_imputed, self.target_future, test_size=0.2, random_state=42)
        
        self.model_career.fit(X_train_career, y_train_career)
        self.model_future.fit(X_train_future, y_train_future)

        self.feature_importances_career = self.model_career.feature_importances_
        self.feature_importances_future = self.model_future.feature_importances_
        
    def save_feature_importance_plot(self, importances, feature_names, target, filename):
        plt.figure(figsize=(10, 8))
        indices = np.argsort(importances)[::-1]
        plt.title(f'Feature Importances for {target}')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def save_all_feature_importance_plots(self):
        career_path = self.dir_path + '/feature_importance_career_aspirations.png'
        self.save_feature_importance_plot(self.feature_importances_career, self.feature_names, "career aspirations", career_path)
        future_topics_path = self.dir_path + '/feature_importance_future_topics.png'
        self.save_feature_importance_plot(self.feature_importances_future, self.feature_names, "future topics", future_topics_path)
        
    def calculate_feature_importance(self):
        self.impute_data()
        self.train_models()
        self.save_all_feature_importance_plots()

if __name__ == "__main__":
    # Import necessary dependencies
    from config import load_config

    # Load configuration
    config = load_config()

    # File paths for the RNN model data
    file_paths = [
        '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_preprocessing/RNN_models/GRU1.csv',
        '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_preprocessing/RNN_models/LSTM1.csv',
        '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_preprocessing/RNN_models/LSTM2.csv',
        '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_preprocessing/RNN_models/Simple1.csv',
        '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_preprocessing/RNN_models/Simple2.csv'
    ]

    career_importances = []
    future_importances = []

    for path in file_paths:
        data = pd.read_csv(path)
        analyzer = FeatureImportanceAnalyzer(config, data, os.path.basename(path).split('.')[0])
        analyzer.calculate_feature_importance()
        career_importances.append(analyzer.feature_importances_career)
        future_importances.append(analyzer.feature_importances_future)

    # Compute average feature importances
    avg_career_importances = np.mean(career_importances, axis=0)
    avg_future_importances = np.mean(future_importances, axis=0)

    # Create DataFrame for average importances
    feature_names = config["privacy"]["X_list"]
    avg_importance_data = {
        'Feature': feature_names,
        'Career Aspirations Importance': avg_career_importances,
        'Future Topics Importance': avg_future_importances
    }
    avg_importance_df = pd.DataFrame(avg_importance_data)

    # Plot average importances
    fig, ax = plt.subplots(figsize=(12, 8))

    avg_importance_df.plot(kind='bar', x='Feature', ax=ax)

    plt.title('Average Feature Importances for Career Aspirations and Future Topics')
    plt.ylabel('Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save plot to file
    avg_plot_file_path = "/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_preprocessing/average_feature_importance_comparison.png"
    plt.savefig(avg_plot_file_path)

    # Calculate the MSE for each model's feature importances compared to the average
    career_mse = {}
    future_mse = {}

    for path, career_importance, future_importance in zip(file_paths, career_importances, future_importances):
        model_name = os.path.basename(path).split('.')[0]
        career_mse[model_name] = mean_squared_error(avg_career_importances, career_importance)
        future_mse[model_name] = mean_squared_error(avg_future_importances, future_importance)

    # Sort models by MSE
    sorted_career_mse = sorted(career_mse.items(), key=lambda x: x[1])
    sorted_future_mse = sorted(future_mse.items(), key=lambda x: x[1])

    # Convert to DataFrame
    career_df = pd.DataFrame(sorted_career_mse, columns=['Model', 'Career MSE'])
    future_df = pd.DataFrame(sorted_future_mse, columns=['Model', 'Future MSE'])

    # Save to CSV
    career_df.to_csv('/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_preprocessing/career_mse_ranking.csv', index=False)
    future_df.to_csv('/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_preprocessing/future_mse_ranking.csv', index=False)
