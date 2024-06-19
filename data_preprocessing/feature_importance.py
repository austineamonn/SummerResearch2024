import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import shap
import os
import sys
import logging
import matplotlib.pyplot as plt

# Add the SummerResearch2024 directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class FeatureImportanceAnalyzer:
    def __init__(self, config, data):
        # Set up logging
        logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])

        self.config = self.load_config(config_path)
        self.data = None
        self.features = None
        self.target_career = None
        self.target_future = None
        self.feature_names = None
        self.imputer = SimpleImputer(strategy='mean')
        self.model_career = RandomForestRegressor(random_state=42)
        self.model_future = RandomForestRegressor(random_state=42)
        self.shap_values_career = None
        self.shap_values_future = None

    def load_config(self, config_path):
        # Mocked load_config function, replace with actual implementation
        return {
            "running_model": {
                "preprocessed data path": config_path
            }
        }

    def load_data(self):
        file_path = self.config["running_model"]["preprocessed data path"]
        self.data = pd.read_csv(file_path)
        self.features = self.data.drop(columns=['career aspirations', 'future topics'])
        self.target_career = self.data['career aspirations']
        self.target_future = self.data['future topics']
        self.feature_names = self.features.columns

    def impute_data(self):
        self.features_imputed = self.imputer.fit_transform(self.features)

    def train_models(self):
        X_train_career, X_test_career, y_train_career, y_test_career = train_test_split(
            self.features_imputed, self.target_career, test_size=0.2, random_state=42)
        X_train_future, X_test_future, y_train_future, y_test_future = train_test_split(
            self.features_imputed, self.target_future, test_size=0.2, random_state=42)
        
        self.model_career.fit(X_train_career, y_train_career)
        self.model_future.fit(X_train_future, y_train_future)
        
        explainer_career = shap.Explainer(self.model_career, X_train_career)
        self.shap_values_career = explainer_career(X_test_career)

        explainer_future = shap.Explainer(self.model_future, X_train_future)
        self.shap_values_future = explainer_future(X_test_future)

    def save_shap_summary_plot(self, shap_values, features, feature_names, target, filename):
        plt.figure(figsize=(10, 8))  # Increase the figure size
        shap.summary_plot(shap_values, features, feature_names=feature_names, plot_type="bar", show=False)
        plt.title(f'SHAP Summary Plot for {target}')
        plt.subplots_adjust(left=0.3, right=0.9, top=0.9)  # Adjust the layout to prevent cutting off feature names and title
        plt.savefig(filename)
        plt.close()

    def save_all_shap_plots(self):
        X_train_career, X_test_career, y_train_career, y_test_career = train_test_split(
            self.features_imputed, self.target_career, test_size=0.2, random_state=42)
        X_train_future, X_test_future, y_train_future, y_test_future = train_test_split(
            self.features_imputed, self.target_future, test_size=0.2, random_state=42)

        self.save_shap_summary_plot(self.shap_values_career, X_test_career, self.feature_names,
                                    'Career Aspirations', 'shap_summary_career_aspirations.png')
        self.save_shap_summary_plot(self.shap_values_future, X_test_future, self.feature_names,
                                    'Future Topics', 'shap_summary_future_topics.png')

if __name__ == "__main__":
    config_path = 'path_to_config_file'
    analyzer = FeatureImportanceAnalyzer(config_path)
    analyzer.load_data()
    analyzer.impute_data()
    analyzer.train_models()
    analyzer.save_all_shap_plots()
