import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging

from IntelliShield.tradeoffs import (
    ISDecisionTreeClassification,
    ISLogisticRegression,
    ISRandomForestClassification,
    ISDecisionTreeRegressification,
    ISLinearRegressification,
    ISRandomForestRegressification,
    ISDecisionTreeRegression,
    ISLinearRegression,
    ISRandomForestRegression,
    load_feature_importance
)

# TODO: Combine comparisons and just switch the list order
# TODO: Make a list of functions that can be called. Use model_path_dict to get the model path for example the key-value pair ISDecisionTreeRegression: 'decision_tree_regression'

class Comparison:
    def __init__(self, model_list: list, privatization_list: list, reduction_list: list, target_list: list, input_path: str, output_path: str, model_path_dict: dict, logger=None):
        # Initalize inputs
        self.model_list = model_list
        self.privatization_list = privatization_list
        self.reduction_list = reduction_list
        self.target_list = target_list
        self.model_path_dict = model_path_dict
        self.logger = logger

        # Get the paths and ensure they exist
        self.input_path = input_path
        self.output_path = output_path
        self.check_paths()
    
    def check_paths(self):
        # Create the output directory if it doesn't exist
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)
        # Raise error is the input path does not exist
        if not os.path.exists(self.input_path):
            raise FileNotFoundError("The input path does not exist")

def pipeline(Comparison: Comparison, compare_models=False, compare_reduction=False, compare_privatization=False, compare_target=False):
    if compare_models:
        order = [Comparison.model_list, Comparison.privatization_list, Comparison.reduction_list, Comparison.target_list]
        make_comparison(Comparison, order)
    elif compare_reduction:
        order = [Comparison.reduction_list, Comparison.privatization_list, Comparison.target_list, Comparison.model_list]
        make_comparison(Comparison, order)
    elif compare_privatization:
        order = [Comparison.privatization_list, Comparison.reduction_list, Comparison.target_list, Comparison.model_list]
        make_comparison(Comparison, order)
    elif compare_target:
        order = [Comparison.target_list, Comparison.privatization_list, Comparison.reduction_list, Comparison.model_list]
        make_comparison(Comparison, order)
    elif Comparison.logger is not None:
        Comparison.logger.warning("No pipeline was chosen, so nothing was run")

def make_comparison(Comparison: Comparison, order: list, save_files=True, return_files=False):
    list_1 = order[0]
    list_2 = order[1]
    list_3 = order[2]
    list_4 = order[3]
    for elem_1 in list_1:
        metrics_df_list = []
        importance_df_list = []
        for elem_2 in list_2:
            for elem_3 in list_3:
                for elem_4 in list_4:
                    # Determine which element is which
                    model, privatization_type, reduction_type, target, folder_name = get_elements(Comparison, elem_1, elem_2, elem_3, elem_4)

                    # Get the proper output path of the model (i.e. where all the files are stored)
                    output_path = get_output_path(Comparison, model, privatization_type, reduction_type, target)

                    tradeoff_model = model(privatization_type, reduction_type, target, output_path=output_path, model_ran=True)
                    print(type(tradeoff_model))
                    metrics, importance = load_model_metrics(tradeoff_model)

                    # Append the outputs
                    metrics_df_list.append(metrics)
                    importance_df_list.append(importance)

        # Concatenate all DataFrames into one
        combined_metrics_df = pd.concat(metrics_df_list, ignore_index=True)
        combined_features_df = pd.concat(importance_df_list, ignore_index=True)

        # Calculate the metrics for each column
        metrics = {
            'mean': combined_metrics_df.mean(),
            'median': combined_metrics_df.median(),
            'std': combined_metrics_df.std(),
            'min': combined_metrics_df.min(),
            'max': combined_metrics_df.max()
        }

        # Get the final dataframes
        final_features_df = combined_features_df.groupby('Feature').agg(['mean', 'median', 'std', 'min', 'max'])
        final_metrics_df = pd.DataFrame(metrics)

        # Save as CSVs
        if save_files:
            final_metrics_df.to_csv(f'{Comparison.output_path}/{folder_name}/combined_metrics.csv', index=True)
            final_features_df.to_csv(f'{Comparison.output_path}/{folder_name}/combined_feature_importance.csv', index=True)
        
        # Return the files
        if return_files:
            final_metrics_df, final_features_df

def get_elements(Comparison: Comparison, elem_1, elem_2, elem_3, elem_4):
    # Combine the elements into one list
    elem_list = [elem_1, elem_2, elem_3, elem_4]

    for elem in elem_list:
        if elem in Comparison.reduction_list:
            reduction_type = elem
            folder_name = f'dimensinality_reduction/{elem}'
        elif elem in Comparison.privatization_list:
            privatization_type = elem
            folder_name = f'privatization/{elem}'
        elif elem in Comparison.target_list:
            target = elem
            folder_name = f'target/{elem}'
        elif elem in Comparison.model_list:
            model = elem
            model_name = Comparison.model_path_dict.get(model)[0]
            folder_name = f'model/{model_name}'

    return model, privatization_type, reduction_type, target, folder_name

def get_output_path(Comparison: Comparison, Model, privatization_type: str, reduction_type: str, target: str):
    target_name = target.replace(' ','_')
    model_value_list_length = len(Comparison.model_path_dict.get(Model))
    for i in range(model_value_list_length):
        model_folder = Comparison.model_path_dict.get(Model)[i]
        input_path = f'{Comparison.input_path}/{model_folder}'
        output_path = f'{input_path}/outputs/{privatization_type}/{reduction_type}/{target_name}'
        # Check if the path exists
        if os.path.exists(output_path):
            if Comparison.logger is not None:
                if Comparison.logger.isEnabledFor(logging.DEBUG):
                    Comparison.logger.debug(f"The path: {output_path} was chosen")
                break
    return output_path

def plot_metrics(Comparison: Comparison, comparison_df, comparison_type, metrics=['mean', 'median', 'std', 'min', 'max']):
    """
    Plots and saves graphs for specified metrics from the aggregated DataFrame.

    Parameters:
    - Comparison: A comparison object
    - comparison_df: Aggregated DataFrame containing the metrics.
    - metrics: A list of metrics to examine
    """
    for metric in metrics:
        # Extract the specified metric from the comparison DataFrame
        metric_df = comparison_df.xs(metric, level=1, axis=1)

        # Plot the metric
        plt.figure(figsize=(12, 8))
        metric_df.plot(kind='bar')
        plt.title(f'{metric.capitalize()} Importance Comparison Across {comparison_type}')
        plt.xlabel('Feature')
        plt.ylabel(f'{metric.capitalize()} Importance')
        plt.legend(title='Dimension Type')

        # Save the plot
        plt.savefig(f'{Comparison.output_path}{metric}_importance_comparison.png', bbox_inches='tight')
        plt.close()

def load_model_metrics(Model, classname=None) -> tuple:
    print(f"Model type: {type(Model)}")  # Debug print to see the actual type of Model

    # Mapping of class names to expected classes
    classification_regressification_class_names = {
        'ISDecisionTreeClassification',
        'ISLogisticRegression',
        'ISRandomForestClassification',
        'ISDecisionTreeRegressification',
        'ISLinearRegressification',
        'ISRandomForestRegressification'
    }

    regression_class_names = {
        'ISDecisionTreeRegression',
        'ISLinearRegression',
        'ISRandomForestRegression'
    }

    # Get feature importance
    if classname is not None:
        file_path = f'{Model.output_path}/{classname}/feature_importance.csv'
    else:
        file_path = f'{Model.output_path}/feature_importance.csv'
    importance = load_feature_importance(Model, file_path, return_df=True)

    # Check for classification and regressification models
    if Model.__class__.__name__ in classification_regressification_class_names:
        with open(file_path, 'r') as file:
            report = json.load(file)
        return report, importance
    
    # Check for regression models
    if Model.__class__.__name__ in regression_class_names:
        metrics = pd.read_csv(file_path)
        return metrics, importance
    
    # Raise error if Model is not recognized
    raise ValueError("Need a proper Model object")