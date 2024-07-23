import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
import seaborn as sns

from IntelliShield.tradeoffs import load_feature_importance

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
    if compare_reduction:
        order = [Comparison.reduction_list, Comparison.privatization_list, Comparison.target_list, Comparison.model_list]
        make_comparison(Comparison, order)
    if compare_privatization:
        order = [Comparison.privatization_list, Comparison.reduction_list, Comparison.target_list, Comparison.model_list]
        make_comparison(Comparison, order)
    if compare_target:
        order = [Comparison.target_list, Comparison.privatization_list, Comparison.reduction_list, Comparison.model_list]
        make_comparison(Comparison, order)

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
                    metrics, importance = load_model_metrics(tradeoff_model)

                    # Remove target column from importance (only present in some classification models)
                    if 'Target' in importance.columns:
                        importance.drop(columns=['Target'], inplace=True)

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
            'max': combined_metrics_df.max(),
            'first_quartile': combined_metrics_df.quantile(0.25),
            'third_quartile': combined_metrics_df.quantile(0.75)
        }

        # Define the aggregate functions for the features
        def custom_agg(group):
            return pd.Series({
                'mean': group.mean(),
                'median': group.median(),
                'std': group.std(),
                'min': group.min(),
                'max': group.max(),
                'first_quartile': group.quantile(0.25),
                'third_quartile': group.quantile(0.75)
            })

        # Get the final dataframes
        final_metrics_df = pd.DataFrame(metrics)
        final_features_df = combined_features_df.groupby('Feature').apply(lambda group: group.apply(custom_agg)).unstack()

        # Save as CSVs
        if save_files:
            output_directory = f'{Comparison.output_path}/{folder_name}'
             # Create the output directory if it doesn't exist
            if not os.path.exists(output_directory):
                os.makedirs(output_directory, exist_ok=True)

            # Save the aggregated files
            final_metrics_df.to_csv(f'{output_directory}/aggregate_metrics.csv', index=True)
            final_features_df.to_csv(f'{output_directory}/aggregate_feature_importance.csv', index=True)

            # Save the non aggregated files
            combined_metrics_df.to_csv(f'{output_directory}/combined_metrics.csv', index=True)
            combined_features_df.to_csv(f'{output_directory}/combined_feature_importance.csv', index=True)
        
        # Return the files
        if return_files:
            final_metrics_df, final_features_df

def get_elements(Comparison: Comparison, elem_1, elem_2, elem_3, elem_4):
    # Combine the elements into one list
    elem_list = [elem_1, elem_2, elem_3, elem_4]

    # Associate each element with the proper list and name the folder based on the first element
    for i, elem in enumerate(elem_list):
        if elem in Comparison.reduction_list:
            reduction_type = elem
            if i == 0:
                folder_name = f'dimensionality_reduction/{elem}'
        elif elem in Comparison.privatization_list:
            privatization_type = elem
            if i == 0:
                folder_name = f'privatization/{elem}'
        elif elem in Comparison.target_list:
            target = elem
            if i == 0:
                folder_name = f'target/{elem}'
        elif elem in Comparison.model_list:
            model = elem
            if i == 0:
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

def load_model_metrics(Model, ave_type='macro') -> tuple:
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
    importance = load_feature_importance(Model, f'{Model.output_path}/feature_importance.csv', return_df=True)

    # Check for classification and regressification models
    if Model.__class__.__name__ in classification_regressification_class_names:
        with open(f'{Model.output_path}/classification_report.json', 'r') as file:
            report = json.load(file)
        
        # Extract relevant values from the file
        metrics = {
            'accuracy': report.get('accuracy'),
            'precision': report.get(f'{ave_type} avg').get('precision'),
            'recall': report.get(f'{ave_type} avg').get('recall'),
            'f1-score': report.get(f'{ave_type} avg').get('f1-score'),
            'support': report.get(f'{ave_type} avg').get('support'),
            'time': report.get('time')
        }
        report_df = pd.DataFrame(metrics, index=[0])
        return report_df, importance
    
    # Check for regression models
    if Model.__class__.__name__ in regression_class_names:
        metrics = pd.read_csv(f'{Model.output_path}/metrics.csv')
        return metrics, importance
    
    # Raise error if Model is not recognized
    raise ValueError("Need a proper Model object")

def find_and_load_csv_files_as_dataframe(directory, filename):
    """
    Iterates through the directory and its subdirectories, finds all CSV files with the specified name,
    and loads their content into pandas DataFrames. The result is a DataFrame with one column
    containing lists of folder names representing the path to each found file, and another column
    containing the DataFrames of these files.
    
    Parameters:
    - directory (str): The root directory to start the search.
    - filename (str): The name of the CSV files to search for.
    
    Returns:
    - pd.DataFrame: A DataFrame where one column contains lists of folder names representing the path to the found files
                    and another column contains the DataFrames of these files.
    """
    folder_paths = []
    dataframes = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file == filename:
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                folder_path = os.path.relpath(root, directory).split(os.sep)
                folder_paths.append(folder_path)
                dataframes.append(df)
    
    result_df = pd.DataFrame({
        'Folder List': folder_paths,
        'DataFrame Content': dataframes
    })
    
    return result_df

def load_files(Comparison: Comparison):
    pass

def boxplot(Comparison: Comparison):
    pass

# Load the CSV files
gru_file_path = '/mnt/data/combined_feature_importance_GRU.csv'
lstm_file_path = '/mnt/data/combined_feature_importance_LSTM.csv'
simple_file_path = '/mnt/data/combined_feature_importance_Simple.csv'

gru_data = pd.read_csv(gru_file_path)
lstm_data = pd.read_csv(lstm_file_path)
simple_data = pd.read_csv(simple_file_path)

# Add a column to identify the model type in each dataset
gru_data['Model'] = 'GRU'
lstm_data['Model'] = 'LSTM'
simple_data['Model'] = 'Simple'

# Combine all datasets into a single DataFrame
combined_data = pd.concat([gru_data, lstm_data, simple_data])

# Drop the unnamed index column
combined_data = combined_data.drop(columns=['Unnamed: 0'])

# Create a box plot
plt.figure(figsize=(14, 8))
sns.boxplot(x='Feature', y='Importance', hue='Model', data=combined_data, palette="Set3")

plt.title('Box Plot of Feature Importance by Model')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.legend(title='Model')
plt.tight_layout()

# Display the plot
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV files for metrics
gru_metrics_file_path = '/mnt/data/combined_metrics_GRU.csv'
lstm_metrics_file_path = '/mnt/data/combined_metrics_LSTM.csv'
simple_metrics_file_path = '/mnt/data/combined_metrics_Simple.csv'

gru_metrics = pd.read_csv(gru_metrics_file_path)
lstm_metrics = pd.read_csv(lstm_metrics_file_path)
simple_metrics = pd.read_csv(simple_metrics_file_path)

# Add a column to identify the model type in each metrics dataset
gru_metrics['Model'] = 'GRU'
lstm_metrics['Model'] = 'LSTM'
simple_metrics['Model'] = 'Simple'

# Combine all metrics datasets into a single DataFrame
combined_metrics = pd.concat([gru_metrics, lstm_metrics, simple_metrics])

# Drop the unnamed index column
combined_metrics = combined_metrics.drop(columns=['Unnamed: 0'])

# Melt the dataframe for easier plotting
melted_metrics = combined_metrics.melt(id_vars=['Model'], var_name='Metric', value_name='Value')

# Create subplots for each metric
metrics = melted_metrics['Metric'].unique()
n_metrics = len(metrics)

fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 8 * n_metrics))

for i, metric in enumerate(metrics):
    sns.boxplot(ax=axes[i], x='Model', y='Value', data=melted_metrics[melted_metrics['Metric'] == metric], palette="Set3")
    axes[i].set_title(f'Comparison of {metric} across Models')
    axes[i].set_xlabel('Model')
    axes[i].set_ylabel(metric)

plt.tight_layout()
plt.show()


def perform_statistical_tests(metrics_files, feature_files):
    import pandas as pd
    from scipy.stats import f_oneway, kruskal, ttest_ind

    def load_and_prepare_data(file_paths, data_type):
        data_frames = []
        for model, file_path in file_paths.items():
            df = pd.read_csv(file_path)
            df['Model'] = model
            data_frames.append(df)
        combined_data = pd.concat(data_frames)
        combined_data = combined_data.drop(columns=['Unnamed: 0'])
        return combined_data

    def run_anova_kruskal(data, group_column, value_column):
        unique_groups = data[group_column].unique()
        grouped_data = [data[data[group_column] == group][value_column] for group in unique_groups]
        anova_stat, anova_p = f_oneway(*grouped_data)
        kruskal_stat, kruskal_p = kruskal(*grouped_data)
        return anova_stat, anova_p, kruskal_stat, kruskal_p

    def run_pairwise_ttests(data, group_column, value_column):
        unique_groups = data[group_column].unique()
        pairwise_results = []
        for i in range(len(unique_groups)):
            for j in range(i + 1, len(unique_groups)):
                group1 = unique_groups[i]
                group2 = unique_groups[j]
                values1 = data[data[group_column] == group1][value_column]
                values2 = data[data[group_column] == group2][value_column]
                t_stat, p_val = ttest_ind(values1, values2)
                pairwise_results.append((f'{group1} vs {group2}', t_stat, p_val))
        return pairwise_results

    # Load and prepare data
    metrics_data = load_and_prepare_data(metrics_files, 'metrics')
    features_data = load_and_prepare_data(feature_files, 'features')

    # Metrics tests
    metrics = ['accuracy', 'precision', 'recall', 'f1-score', 'support', 'time']
    metrics_results = []
    significant_metrics = []

    for metric in metrics:
        anova_stat, anova_p, kruskal_stat, kruskal_p = run_anova_kruskal(metrics_data, 'Model', metric)
        metrics_results.append((metric, anova_stat, anova_p, kruskal_stat, kruskal_p))
        if anova_p < 0.05 or kruskal_p < 0.05:
            significant_metrics.append(metric)

    # Features tests
    features = features_data['Feature'].unique()
    features_results = []
    significant_features = []

    for feature in features:
        anova_stat, anova_p, kruskal_stat, kruskal_p = run_anova_kruskal(features_data[features_data['Feature'] == feature], 'Model', 'Importance')
        features_results.append((feature, anova_stat, anova_p, kruskal_stat, kruskal_p))
        if anova_p < 0.05 or kruskal_p < 0.05:
            significant_features.append(feature)

    # Pairwise comparisons for significant metrics and features
    pairwise_comparisons = []

    for metric in significant_metrics:
        pairwise_comparisons.extend(run_pairwise_ttests(metrics_data, 'Model', metric))

    for feature in significant_features:
        pairwise_comparisons.extend(run_pairwise_ttests(features_data[features_data['Feature'] == feature], 'Model', 'Importance'))

    # Combine results into DataFrames
    metrics_results_df = pd.DataFrame(metrics_results, columns=['Metric', 'ANOVA Statistic', 'ANOVA p-value', 'Kruskal-Wallis Statistic', 'Kruskal-Wallis p-value'])
    features_results_df = pd.DataFrame(features_results, columns=['Feature', 'ANOVA Statistic', 'ANOVA p-value', 'Kruskal-Wallis Statistic', 'Kruskal-Wallis p-value'])
    pairwise_results_df = pd.DataFrame(pairwise_comparisons, columns=['Comparison', 'T-test Statistic', 'p-value'])

    return metrics_results_df, features_results_df, pairwise_results_df

# Define the file paths for metrics and features
metrics_files = {
    'GRU': '/mnt/data/combined_metrics_GRU.csv',
    'LSTM': '/mnt/data/combined_metrics_LSTM.csv',
    'Simple': '/mnt/data/combined_metrics_Simple.csv'
}

feature_files = {
    'GRU': '/mnt/data/combined_feature_importance_GRU.csv',
    'LSTM': '/mnt/data/combined_feature_importance_LSTM.csv',
    'Simple': '/mnt/data/combined_feature_importance_Simple.csv'
}

# Run the statistical tests function
metrics_results_df, features_results_df, pairwise_results_df = perform_statistical_tests(metrics_files, feature_files)
