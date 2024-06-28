import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import logging

# Add the SummerResearch2024 directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class SHAP_Grapher:
    def __init__(self, config):

        # Set up logging
        logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])

        # Make the lists
        self.privatized_datasets = ['basic']
        self.RNN_models = ['GRU1', 'LSTM1', 'Simple1']
        self.tradeoff_models = ['DecisionTree', 'LinearRegression', 'RandomForest']
        self.targets = ['all_columns', 'career_aspirations', 'ethnoracial_group', 'future_topics', 'gender', 'international_status', 'private_columns', 'utility_columns']

        # Define the mapping for the column name endings
        self.column_mapping = {
            '_0': ' to Ethnoracial Group',
            '_1': ' to Gender',
            '_2': ' to International Student Status',
            '_3': ' to Career Aspirations',
            '_4': ' to Future Topics'
        }

    # Function to update column names
    def update_column_names(self, columns, mapping):
        updated_names = []
        for name in columns:
            updated_name = name
            for suffix in mapping:
                if name.endswith(suffix):
                    base_name = name.replace(suffix, '').replace('_', ' ').title()
                    updated_name = base_name + mapping[suffix]
                    break
            else:
                updated_name = name.title()
            updated_names.append(updated_name)
        return updated_names

    def plot_graphs(self, df, dataset, RNN_model, tradeoff_model, target):
        # Create the Graph Titles
        title_target = target.replace('_', ' ').title()
        title_tradeoff_model = tradeoff_model.replace('_', ' ')

        # Update the column names
        updated_feature_names = self.update_column_names(df.columns[:-2], self.column_mapping)

        # Extract SHAP values and feature names
        shap_columns = df.iloc[:, :-2]
        shap_values = shap_columns.values

        # Create SHAP summary plot and save it
        plt.figure(figsize=(10, 15))
        shap.summary_plot(shap_values, updated_feature_names, show=False, max_display=len(updated_feature_names))
        plt.title(f"SHAP Feature Importance Summary Plot for {title_tradeoff_model} Model for {title_target}", wrap=True)
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to make room for the title

        # Save the plot as an image file
        plt.savefig(f"SHAP_graphs_{dataset}/{RNN_model}/{tradeoff_model}/shap_feature_importance_summary_plot_{target}.png")
        print(f"SHAP_graphs_{dataset}/{RNN_model}/{tradeoff_model}/shap_feature_importance_summary_plot_{target}.png")
        plt.close()  # Close the figure to free up memory

        # Calculate mean absolute SHAP value for each feature
        mean_abs_shap_values = np.abs(shap_columns).mean()

        # Convert to a SHAP values object
        shap_values_object = shap.Explanation(values=mean_abs_shap_values.values, 
                                            feature_names=updated_feature_names)

        # Create SHAP summary plot and save it
        plt.figure(figsize=(10, 15))
        shap.plots.bar(shap_values_object, show=False, max_display=len(updated_feature_names))
        plt.title(f"SHAP Feature Importance Bar Graph for {title_tradeoff_model} Model for {title_target}", wrap=True)
        plt.gca().set_aspect('auto')  # Auto aspect ratio
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to make room for the title

        # Save the plot as an image file
        plt.savefig(f"SHAP_graphs_{dataset}/{RNN_model}/{tradeoff_model}/shap_feature_importance_bar_graph_{target}.png")
        print(f"SHAP_graphs_{dataset}/{RNN_model}/{tradeoff_model}/shap_feature_importance_bar_graph_{target}.png")
        plt.close()  # Close the figure to free up memory

    def generate_graphs(self):
        for dataset in self.privatized_datasets:
            for RNN_model in self.RNN_models:
                for tradeoff_model in self.tradeoff_models:
                    for target in self.targets:
                        # Load the CSV file into a dataframe
                        df = pd.read_csv(f'../../calculating_tradeoffs/shap_values_{dataset}/{RNN_model}/{tradeoff_model}/shap_values_{tradeoff_model}_{target}.csv')
                        self.plot_graphs(df, dataset, RNN_model, tradeoff_model, target)

if __name__ == "__main__":
    # Import necessary dependencies
    from config import load_config

    # Load configuration
    config = load_config()

    grapher = SHAP_Grapher(config)
    grapher.generate_graphs()