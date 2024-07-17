import json
import pandas as pd

from IntelliShield.tradeoffs import ISDecisionTreeClassification, ISDecisionTreeRegressification, ISDecisionTreeRegression, ISLogisticRegression, load_feature_importance

class Comparison:
    def __init__(self, privatization_list: list, reduction_list: list, target_list: list, input_path, output_path):
        # Initalize inputs
        self.privatization_list = privatization_list
        self.reduction_list = reduction_list
        self.target_list = target_list
        self.input_path = input_path
        self.output_path = output_path

def compare_dimensionality_reduction(Comparison: Comparison):
    # Classification Models
    compare_dimensionality_reduction_classification(Comparison)

    # Regressification Models
    compare_dimensionality_reduction_regressification(Comparison)

    # Regression Models
    compare_dimensionality_reduction_regression(Comparison)

def compare_dimensionality_reduction_regression(Comparison: Comparison):
    for reduction_type in Comparison.reduction_list:
        reduction_type_metrics_df = []
        reduction_type_importance_df = []
        for privatization_type in Comparison.privatization_list:
            for target in Comparison.target_list:
                target_name = target.replace(' ', '_')

                # Decision Tree Regression
                input_path = f'{Comparison.input_path}/decision_tree_regression'
                output_path = f'{input_path}/outputs/{privatization_type}/{reduction_type}/{target_name}'

                tradeoff_model = ISDecisionTreeRegression(privatization_type, reduction_type, target, output_path=output_path, model_ran=True)
                metrics, importance = load_model_metrics(tradeoff_model)

                # Append the outputs
                reduction_type_metrics_df.append(metrics)
                reduction_type_importance_df.append(importance)

        # Concatenate all DataFrames into one
        combined_metrics_df = pd.concat(reduction_type_metrics_df, ignore_index=True)

        # Calculate the metrics for each column
        metrics = {
            'mean': combined_metrics_df.mean(),
            'median': combined_metrics_df.median(),
            'std': combined_metrics_df.std(),
            'min': combined_metrics_df.min(),
            'max': combined_metrics_df.max(),
            'sum': combined_metrics_df.sum()
        }

        # Convert the metrics dictionary to a DataFrame
        final_metrics_df = pd.DataFrame(metrics)

def compare_dimensionality_reduction_regressification(Comparison: Comparison):
    pass

def compare_dimensionality_reduction_classification(Comparison: Comparison):
    pass

def compare_privatization(Comparison: Comparison):
    pass

def compare_target(Comparison: Comparison):
    pass

def load_model_metrics(Model, classname=None):
    # Get feature importance
    if classname is not None:
        file_path = f'{Model.output_path}/{classname}/feature_importance.csv'
    else:
        file_path = f'{Model.output_path}/feature_importance.csv'
    importance = load_feature_importance(Model, file_path, return_df=True)

    # Get metrics or classification report
    if isinstance(Model, ISDecisionTreeClassification) or isinstance(Model, ISLogisticRegression) or isinstance(Model, ISDecisionTreeRegressification):
        with open(file_path, 'r') as file:
            report = json.load(file)
        
        return report, importance
    elif isinstance(Model, ISDecisionTreeRegression):
        metrics = pd.read_csv(file_path)
        return metrics, importance
