import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
#import shap
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
        self.shap_values_career = None
        self.shap_values_future = None

    def impute_data(self):
        self.features_imputed = self.imputer.fit_transform(self.features)

    def train_models(self):
        X_train_career, X_test_career, y_train_career, y_test_career = train_test_split(
            self.features_imputed, self.target_career, test_size=0.2, random_state=42)
        X_train_future, X_test_future, y_train_future, y_test_future = train_test_split(
            self.features_imputed, self.target_future, test_size=0.2, random_state=42)
        
        self.model_career.fit(X_train_career, y_train_career)
        self.model_future.fit(X_train_future, y_train_future)
        
        """explainer_career = shap.Explainer(self.model_career, X_train_career)
        self.shap_values_career = explainer_career(X_test_career)

        explainer_future = shap.Explainer(self.model_future, X_train_future)
        self.shap_values_future = explainer_future(X_test_future)"""

    """def save_shap_summary_plot(self, shap_values, features, feature_names, target, filename):
        plt.figure(figsize=(10, 8))  # Increase the figure size
        shap.summary_plot(shap_values, features, feature_names=feature_names, plot_type="bar", show=False)
        plt.title(f'SHAP Summary Plot for {target}')
        plt.subplots_adjust(left=0.3, right=0.9, top=0.9)  # Adjust the layout to prevent cutting off feature names and title
        plt.savefig(filename)
        plt.close()"""

    """def save_all_shap_plots(self):
        X_train_career, X_test_career, y_train_career, y_test_career = train_test_split(
            self.features_imputed, self.target_career, test_size=0.2, random_state=42)
        X_train_future, X_test_future, y_train_future, y_test_future = train_test_split(
            self.features_imputed, self.target_future, test_size=0.2, random_state=42)

        career_path = self.dir_path + '/shap_summary_career_aspirations.png'
        self.save_shap_summary_plot(self.shap_values_career, X_test_career, self.feature_names,
                                    'Career Aspirations', career_path)
        future_topics_path = self.dir_path + '/shap_summary_future_topics.png'
        self.save_shap_summary_plot(self.shap_values_future, X_test_future, self.feature_names,
                                    'Future Topics', future_topics_path)"""
        
    def save_feature_importance_plot(self, model, feature_names, target, filename):
        plt.figure(figsize=(10, 8))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.title(f'Feature Importances for {target}')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def save_all_feature_importance_plots(self):
        career_path = self.dir_path + '/feature_importance_career_aspirations.png'
        self.save_feature_importance_plot(self.model_career, self.feature_names, "career aspirations", career_path)
        future_topics_path = self.dir_path + '/feature_importance_future_topics.png'
        self.save_feature_importance_plot(self.model_future, self.feature_names, "future topics", future_topics_path)
        
    def calculate_feature_importance(self):
        self.impute_data()
        self.train_models()
        #self.save_all_shap_plots()
        self.save_all_feature_importance_plots()

if __name__ == "__main__":
    # Import necessary dependencies
    from config import load_config

    # Load configuration and data
    config = load_config()\
    
    # GRU1
    data1 = pd.read_csv('/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_preprocessing/RNN_models/GRU1.csv')

    analyzer1 = FeatureImportanceAnalyzer(config, data1, 'GRU1')
    analyzer1.calculate_feature_importance()

    # LSTM1
    data2 = pd.read_csv('/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_preprocessing/RNN_models/LSTM1.csv')

    analyzer2 = FeatureImportanceAnalyzer(config, data2, 'LSTM1')
    analyzer2.calculate_feature_importance()

    # LSTM2
    data3 = pd.read_csv('/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_preprocessing/RNN_models/LSTM2.csv')

    analyzer3 = FeatureImportanceAnalyzer(config, data3, 'LSTM2')
    analyzer3.calculate_feature_importance()

    # Simple1
    data4 = pd.read_csv('/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_preprocessing/RNN_models/Simple1.csv')

    analyzer4 = FeatureImportanceAnalyzer(config, data4, 'Simple1')
    analyzer4.calculate_feature_importance()

    # Simple2
    data5 = pd.read_csv('/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_preprocessing/RNN_models/Simple2.csv')

    analyzer5 = FeatureImportanceAnalyzer(config, data5, 'Simple2')
    analyzer5.calculate_feature_importance()
