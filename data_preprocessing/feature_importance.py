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
import shap

# Add the SummerResearch2024 directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class FeatureImportanceAnalyzer:
    def __init__(self, config, data, name):
        # Set up logging
        logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])

        self.dir_path = f'/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_preprocessing/feature_importance_{name}'
        self.features = data.drop(columns=['career aspirations', 'future topics', 'ethnoracial group', 'gender', 'international status'])
        self.targets = {
            'career aspirations': data['career aspirations'],
            'future topics': data['future topics'],
            'ethnoracial group': data['ethnoracial group'],
            'gender': data['gender'],
            'international status': data['international status']
        }
        self.feature_names = config["privacy"]["X_list"]
        self.imputer = SimpleImputer(strategy='mean')
        self.models = {target: RandomForestRegressor(random_state=42) for target in self.targets}
        self.feature_importances = {target: None for target in self.targets}
        self.shap_values = {target: None for target in self.targets}

    def impute_data(self):
        self.features_imputed = self.imputer.fit_transform(self.features)
        self.targets_imputed = {target: self.imputer.fit_transform(values.values.reshape(-1, 1)).ravel() for target, values in self.targets.items()}


    def train_models(self):
        self.splits = {}
        for target, values in self.targets_imputed.items():
            X_train, X_test, y_train, y_test = train_test_split(
                self.features_imputed, values, test_size=0.2, random_state=42)
            self.models[target].fit(X_train, y_train)
            self.splits[target] = (X_train, X_test, y_train, y_test)
    
    def calculate_feature_importance(self, method='built-in'):
        for target in self.targets:
            if method == 'built-in':
                self.feature_importances[target] = self.models[target].feature_importances_
            elif method == 'shap':
                explainer = shap.TreeExplainer(self.models[target])
                self.shap_values[target] = explainer.shap_values(self.features_imputed)
                self.feature_importances[target] = np.abs(self.shap_values[target]).mean(axis=0)

    def average_feature_importance(self, files, method='built-in'):
        total_importance = {target: np.zeros(len(self.feature_names)) for target in self.targets}
        
        for file in files:
            data = pd.read_csv(file)
            self.features = data.drop(columns=['career aspirations', 'future topics', 'ethnoracial group', 'gender', 'international status'])
            for target in self.targets:
                self.targets[target] = data[target]
            self.impute_data()
            self.train_models()
            self.calculate_feature_importance(method)
            for target in self.targets:
                total_importance[target] += self.feature_importances[target]
        
        average_importance = {target: total_importance[target] / len(files) for target in self.targets}
        
        return average_importance

    def plot_feature_importance(self, importance, title):
        plt.figure(figsize=(10, 6))  # Adjusted figure size for small display
        indices = np.argsort(importance)
        bars = plt.barh(range(len(indices)), importance[indices], align='center', color='skyblue')  # Changed bar color
        plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices], fontsize=14, weight='bold')  # Smaller and bold font for y-ticks
        plt.xticks(fontsize=14, weight='bold')  # Smaller and bold font for x-ticks
        plt.title(title, fontsize=18, fontweight='bold')  # Smaller and bolded title font size
        plt.xlabel('Feature Importance', fontsize=16, weight='bold')  # Smaller and bold font size for x-label

        # Adding value labels inside the bars
        for bar, value in zip(bars, importance[indices]):
            plt.text(bar.get_width() - 0.05, bar.get_y() + bar.get_height() / 2, f'{value:.2f}', 
                     ha='center', va='center', fontsize=14, weight='bold', color='black')  # Value labels inside bars

        plt.tight_layout(pad=2.0)  # Adjusted layout with padding
        plt.show()

    def save_mse_differences(self, files):
        mse_differences = []
        
        for file in files:
            data = pd.read_csv(file)
            self.features = data.drop(columns=['career aspirations', 'future topics', 'ethnoracial group', 'gender', 'international status'])
            for target in self.targets:
                self.targets[target] = data[target]
            self.impute_data()
            self.train_models()
            
            for target in self.targets:
                X_train, X_test, y_train, y_test = self.splits[target]
                preds = self.models[target].predict(X_test)
                mse = mean_squared_error(y_test, preds)
                mse_differences.append({'file': file, 'target': target, 'mse': mse})
        
        mse_df = pd.DataFrame(mse_differences)
        mse_df.to_csv(os.path.join(self.dir_path, 'mse_differences.csv'), index=False)
    
    def process_files(self, files, method='built-in'):
        average_importance = self.average_feature_importance(files, method)
        
        for target in average_importance:
            self.plot_feature_importance(average_importance[target], f'Average Feature Importance for {target}')
        
        self.save_mse_differences(files)
        

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

    # Path to the private columns file
    private_columns_path = '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/calculating_tradeoffs/Private_Columns.csv'

    # Path to save feature importance results
    feature_importance_dir = '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_preprocessing/feature_importance/'

    # Ensure the feature importance directory exists
    os.makedirs(feature_importance_dir, exist_ok=True)

    # Load the private columns data
    private_columns_data = pd.read_csv(private_columns_path)

    # Process each file individually
    combined_file_paths = []

    for file_path in file_paths:
        # Load RNN model data
        rnn_data = pd.read_csv(file_path)
        
        # Combine with private columns data
        combined_data = pd.concat([rnn_data, private_columns_data], axis=1)
        final_data = combined_data.sample(frac=0.1)
        
        # Save the combined data to a new file
        combined_file_path = file_path.replace('.csv', '_combined.csv')
        final_data.to_csv(combined_file_path, index=False)
        combined_file_paths.append(combined_file_path)

        # Set feature names in config
        config["privacy"]["X_list"] = list(final_data.drop(columns=['career aspirations', 'future topics', 'ethnoracial group', 'gender', 'international status']).columns)

        # Create instances of FeatureImportanceAnalyzer for built-in and SHAP methods
        name = os.path.basename(file_path).replace('.csv', '')
        
        analyzer_built_in = FeatureImportanceAnalyzer(config, final_data, f'{name}_built_in')
        analyzer_shap = FeatureImportanceAnalyzer(config, final_data, f'{name}_shap')
        
        # Impute data and train models before calculating feature importance
        analyzer_built_in.impute_data()
        analyzer_built_in.train_models()
        analyzer_built_in.calculate_feature_importance(method='built-in')

        analyzer_shap.impute_data()
        analyzer_shap.train_models()
        analyzer_shap.calculate_feature_importance(method='shap')

        # Save results to the feature importance directory
        analyzer_built_in.dir_path = os.path.join(feature_importance_dir, f'{name}_built_in')
        analyzer_shap.dir_path = os.path.join(feature_importance_dir, f'{name}_shap')

        os.makedirs(analyzer_built_in.dir_path, exist_ok=True)
        os.makedirs(analyzer_shap.dir_path, exist_ok=True)

    # Initialize analyzers for averaging and MSE calculations using the first combined dataset as an example
    example_combined_data = pd.read_csv(combined_file_paths[0])
    analyzer_built_in = FeatureImportanceAnalyzer(config, example_combined_data, 'average_built_in')
    analyzer_shap = FeatureImportanceAnalyzer(config, example_combined_data, 'average_shap')

    # Set the directory path for saving average results
    analyzer_built_in.dir_path = os.path.join(feature_importance_dir, 'average_built_in')
    analyzer_shap.dir_path = os.path.join(feature_importance_dir, 'average_shap')

    os.makedirs(analyzer_built_in.dir_path, exist_ok=True)
    os.makedirs(analyzer_shap.dir_path, exist_ok=True)

    # Process all combined files to calculate and save average feature importance and MSE differences for both methods
    analyzer_built_in.process_files(combined_file_paths, method='built-in')
    analyzer_shap.process_files(combined_file_paths, method='shap')

    print("Feature importance and MSE differences have been calculated and saved using both methods.")