import logging
from ast import literal_eval
import pandas as pd
from typing import Union
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sn
import joblib
import shap
import numpy as np
import json
import time
import os
import pickle

# TODO: Add the other tradeoff classes
# TODO: Fix error where SHAP values must be saved and reloaded for the graphing to work

# Set the pandas option to avoid silent downcasting
pd.set_option('future.no_silent_downcasting', True)

# Classification Models

class ISLogisticRegression:
    def __init__(self, privatization_type, RNN_model, target, data=None, data_path=None, output_path=None, X_columns=None, classnames=None):
        # Either data or data path must be declared
        if data is None and data_path is None:
            raise ValueError("At least one of 'data' or 'data_path' must be provided.")
        
        # Initialize inputs
        self.privatization_type = privatization_type
        self.RNN_model = RNN_model
        self.target = target
        self.target_name = target.replace(' ', '_')

        # Get the Data
        if data is None:
            self.data = pd.read_csv(data_path, converters={
                'learning style': literal_eval,
                'major': literal_eval,
                'previous courses': literal_eval,
                'course types': literal_eval,
                'course subjects': literal_eval,
                'subjects of interest': literal_eval,
                'extracurricular activities': literal_eval,
                'career aspirations': literal_eval,
                'future topics': literal_eval
            })
        else:
            self.data = data

        # Output paths
        if output_path is not None:
            self.output_path = output_path
        else:
            logging.info("no output path given, so no data will be saved!")

        # Additional Attributes
        self.name = 'logistic_regressor'
        if X_columns is None:
            self.X_columns = ['gpa', 'student semester', 'learning style', 'major', 'previous courses', 'course types', 'course subjects', 'subjects of interest', 'extracurricular activities']
        else:
            self.X_columns = X_columns

        # Initialize empty attributes to be filled.
        self.X = None
        self.X_train = None
        self.X_test = None
        self.y = None
        self.y_train = None
        self.y_test = None
        self.stratified_data = None
        self.tradeoffmodel = None
        self.X_sample = None
        self.cm = None

        if classnames is not None:
            self.classnames = classnames
        elif target == 'ethnoracial group':
            self.classnames = [
                'European American or white', 'Latino/a/x American', 'African American or Black', 'Asian American', 'Multiracial', 'American Indian or Alaska Native', 'Pacific Islander'
            ]
            self.shap_is_list = True
        elif target == 'gender':
            self.classnames = [
                'Female', "Male", 'Nonbinary'
            ]
            self.shap_is_list = True
        elif target == 'international status':
            self.classnames = [
                'Domestic', 'International'
            ]
            self.shap_is_list = False
        elif target == 'socioeconomic status':
            self.classnames = [
                'In poverty', 'Near poverty', 'Lower-middle income', 'Middle income', 'Higher income'
            ]
            self.shap_is_list = True
        else:
            raise ValueError(f"Incorrect target column name {target} for a classification model")
        self.labels = list(range(len(self.classnames)))

class ISDecisionTreeClassification:
    def __init__(self, privatization_type, RNN_model, target, data=None, data_path=None, output_path=None, X_columns=None, classnames=None):
        # Either data or data path must be declared
        if data is None and data_path is None:
            raise ValueError("At least one of 'data' or 'data_path' must be provided.")
        
        # Initialize inputs
        self.privatization_type = privatization_type
        self.RNN_model = RNN_model
        self.target = target
        self.target_name = target.replace(' ', '_')

        # Get the Data
        if data is None:
            self.data = pd.read_csv(data_path, converters={
                'learning style': literal_eval,
                'major': literal_eval,
                'previous courses': literal_eval,
                'course types': literal_eval,
                'course subjects': literal_eval,
                'subjects of interest': literal_eval,
                'extracurricular activities': literal_eval,
                'career aspirations': literal_eval,
                'future topics': literal_eval
            })
        else:
            self.data = data

        # Output paths
        if output_path is not None:
            self.output_path = output_path
        else:
            logging.info("no output path given, so no data will be saved!")

        # Additional Attributes
        self.name = 'decision_tree_classifier'
        self.shap_is_list = True
        if X_columns is None:
            self.X_columns = ['gpa', 'student semester', 'learning style', 'major', 'previous courses', 'course types', 'course subjects', 'subjects of interest', 'extracurricular activities']
        else:
            self.X_columns = X_columns

        # Initialize empty attributes to be filled.
        self.X = None
        self.X_train = None
        self.X_test = None
        self.y = None
        self.y_train = None
        self.y_test = None
        self.stratified_data = None
        self.tradeoffmodel = None
        self.X_sample = None
        self.cm = None

        if classnames is not None:
            self.classnames = classnames
        elif target == 'ethnoracial group':
            self.classnames = [
                'European American or white', 'Latino/a/x American', 'African American or Black', 'Asian American', 'Multiracial', 'American Indian or Alaska Native', 'Pacific Islander'
            ]
        elif target == 'gender':
            self.classnames = [
                'Female', "Male", 'Nonbinary'
            ]
        elif target == 'international status':
            self.classnames = [
                'Domestic', 'International'
            ]
        elif target == 'socioeconomic status':
            self.classnames = [
                'In poverty', 'Near poverty', 'Lower-middle income', 'Middle income', 'Higher income'
            ]
        else:
            raise ValueError(f"Incorrect target column name {target} for a classification model")
        self.labels = list(range(len(self.classnames)))

        # Attributes for getting the best model
        self.models_data = None
        self.ccp_alphas = None
        self.ccp_alpha = None
        self.train_scores = None
        self.test_scores = None
        self.impurities = None
        self.node_counts = None
        self.depth = None
        
# Regressification Models

class ISDecisionTreeRegressification:
    def __init__(self, privatization_type, RNN_model, target, data=None, data_path=None, output_path=None, X_columns=None):
        # Either data or data path must be declared
        if data is None and data_path is None:
            raise ValueError("At least one of 'data' or 'data_path' must be provided.")
        
        # Initialize inputs
        self.privatization_type = privatization_type
        self.RNN_model = RNN_model
        self.target = target
        self.target_name = target.replace(' ', '_')

        # Get the Data
        if data is None:
            self.data = pd.read_csv(data_path, converters={
                'learning style': literal_eval,
                'major': literal_eval,
                'previous courses': literal_eval,
                'course types': literal_eval,
                'course subjects': literal_eval,
                'subjects of interest': literal_eval,
                'extracurricular activities': literal_eval,
                'career aspirations': literal_eval,
                'future topics': literal_eval
            })
        else:
            self.data = data

        self.data = self.data.dropna(subset=[target])

        # Output paths
        if output_path is not None:
            self.output_path = output_path
        else:
            logging.info("no output path given, so no data will be saved!")

        # Additional Attributes
        self.name = 'decision_tree_regressifier'
        self.shap_is_list = False
        if X_columns is None:
            self.X_columns = ['gpa', 'student semester', 'learning style', 'major', 'previous courses', 'course types', 'course subjects', 'subjects of interest', 'extracurricular activities']
        else:
            self.X_columns = X_columns

        # Initialize empty attributes to be filled.
        self.X = None
        self.X_train = None
        self.X_test = None
        self.y = None
        self.y_train = None
        self.y_test = None
        self.tradeoffmodel = None
        self.X_sample = None
        self.cm = None

        # Attributes for getting the best model
        self.models_data = None
        self.ccp_alphas = None
        self.ccp_alpha = None
        self.train_scores = None
        self.test_scores = None
        self.impurities = None
        self.node_counts = None
        self.depth = None

# Regression Models

def make_folders(Model, output_path=None):
    if output_path is None:
        output_path = Model.output_path
    if Model.shap_is_list == False:
        directory = os.path.dirname(f'{output_path}/graphs/feature_scatter_plots/example.py')

        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
    else:
        for name in Model.classnames:
            name = name.replace(' ', '_')
            directory = os.path.dirname(f'{output_path}/graphs/{name}/feature_scatter_plots/example.py')

            # Create the directory if it doesn't exist
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

def split_data(Model, full_model=False):
        if isinstance(Model, ISLogisticRegression) or isinstance(Model, ISDecisionTreeClassification):
            # Define y
            Model.y = Model.data[[Model.target]]

            # Stratify the data by taking only the number of elements the least common values has
            least_common_count = Model.y.value_counts().min()
            if full_model:
                sample_number = least_common_count
            else: # Cut it off at 1,000 if running the ccp alpha calculations
                sample_number = min(1000, least_common_count)
            Model.stratified_data = Model.data.groupby(Model.target, group_keys=False).apply(lambda x: x.sample(sample_number))

            # Set up X
            Model.X = Model.stratified_data[Model.X_columns]

        elif isinstance(Model, ISDecisionTreeRegressification):
            if full_model:
                Model.final_data = Model.data
            else:
                Model.final_data = Model.data.sample(n=2000, random_state=1)
            # Set up X
            Model.X = Model.final_data[Model.X_columns]

        # Change the lists into just the elements within them if the element is a list otherwise just take the element
        for column in Model.X_columns:
            Model.X.loc[:, column] = Model.X[column].apply(lambda x: x[0] if isinstance(x, list) else x)

        if isinstance(Model, ISLogisticRegression):
            # Fill NaN values with the mean of each column
            Model.X = Model.X.fillna(Model.X.mean())

            # Infer objects to avoid future warning
            Model.X = Model.X.infer_objects(copy=False)

        if isinstance(Model, ISLogisticRegression) or isinstance(Model, ISDecisionTreeClassification):
            # Set up y post stratification
            Model.y = Model.stratified_data[[Model.target]]

            # Change the doubles into integers (1.0 -> 1)
            Model.y = Model.y.astype(int)

        elif isinstance(Model, ISDecisionTreeRegressification):
            # Set up y post stratification
            Model.y = Model.final_data[[Model.target]]

            # Change the lists into just the elements within them if the element is a list otherwise just take the element
            for column in Model.y.columns:
                Model.y.loc[:, column] = Model.y[column].apply(lambda x: x[0] if isinstance(x, list) else (int(x) if not np.isnan(x) else x))

        # Test - Train Split
        Model.X_train, Model.X_test, Model.y_train, Model.y_test = train_test_split(Model.X, Model.y, train_size = 0.8, random_state = 1234)

        if isinstance(Model, ISLogisticRegression):
            # Reshape ys to be 1-dimensional
            Model.y_train = Model.y_train.values.ravel()

def score(tradeoffsmodel, X, y, sample_weight=None):
    # Takes in values from a regressor model, but scores them based on a classification (round to the nearest whole number)

    y_pred = tradeoffsmodel.predict(X)

    # Round the predictions and convert to integers
    y_pred_rounded = np.round(y_pred).astype(int)

    return accuracy_score(y, y_pred_rounded, sample_weight=sample_weight)

def get_best_model(Model:Union[ISDecisionTreeClassification], make_graphs=True, return_model=True, return_ccp_alpha=True, save_model=True):
    # Split the data
    split_data(Model)

    # Generate the models with different ccp alphas
    clf = DecisionTreeClassifier(random_state=0)
    path = clf.cost_complexity_pruning_path(Model.X_train, Model.y_train)
    Model.ccp_alphas, Model.impurities = path.ccp_alphas, path.impurities

    # Ensure all ccp_alphas are non-negative
    Model.ccp_alphas = [max(alpha, 0.0) for alpha in Model.ccp_alphas]

    # Fit the models
    clfs = []
    for ccp_alpha in Model.ccp_alphas:
        clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(Model.X_train, Model.y_train)
        clfs.append(clf)
    Model.node_counts = [clf.tree_.node_count for clf in clfs]
    Model.depth = [clf.tree_.max_depth for clf in clfs]

    # Train the models
    Model.train_scores = [clf.score(Model.X_train, Model.y_train) for clf in clfs]
    Model.test_scores = [clf.score(Model.X_test, Model.y_test) for clf in clfs]

    Model.models_data = {
        'models': clfs,
        'ccp alpha': Model.ccp_alphas,
        'train scores': Model.train_scores,
        'test scores': Model.test_scores,
        'impurities': Model.impurities,
        'node count': Model.node_counts,
        'depth': Model.depth
    }
    
    models = pd.DataFrame(Model.models_data)

    if save_model:
        try:
            models.to_csv(f'{Model.output_path}/decision_tree_classifier_models.csv', index=False)
        except OSError:
            make_folders(Model)
            try:
                models.to_csv(f'{Model.output_path}/decision_tree_classifier_models.csv', index=False)
            except:
                raise OSError("The make_folders function is not working. Check to see if the outputs folder can be created.")

    models_sorted = models.sort_values(by='test scores', ascending=False)

    # Pull out the model that has the highest test score and make it the model for training
    Model.tradeoffmodel, Model.ccp_alpha = models_sorted.iloc[0, 0:2]

    if make_graphs:
        graph_impurities(Model)
        graph_nodes_and_depth(Model)
        graph_accuracy(Model)

    # Return the best model and the ccp_alpha
    if return_model and return_ccp_alpha:
        return Model.tradeoffmodel, Model.ccp_alpha
    elif return_model:
        return Model.tradeoffmodel
    elif return_ccp_alpha:
        return_ccp_alpha

def graph_impurities(Model: Union[ISDecisionTreeClassification]):
    # Impurity vs Effective Alpha
    fig, ax = plt.subplots()
    ax.plot(Model.ccp_alphas[:-1], Model.impurities[:-1], marker="o", drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    plt.savefig(f'{Model.output_path}/graphs/effective_alpha_vs_total_impurity.png')
    plt.close()

def graph_nodes_and_depth(Model: Union[ISDecisionTreeClassification]):
    # Model Nodes and Depth vs Alpha
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Model.ccp_alphas, Model.node_counts, marker="o", drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(Model.ccp_alphas, Model.depth, marker="o", drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    fig.tight_layout()
    plt.savefig(f'{Model.output_path}/graphs/effective_alpha_vs_graph_nodes_and_depth.png')
    plt.close()

def graph_accuracy(Model: Union[ISDecisionTreeClassification]):
    # Accuracy vs Alpha
    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(Model.ccp_alphas, Model.train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(Model.ccp_alphas, Model.test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.savefig(f'{Model.output_path}/graphs/effective_alpha_vs_accuracy.png')
    plt.close()

def tree_plotter(Model: Union[ISDecisionTreeClassification], tradeoffmodel=None, save_fig=False, show_fig=False, max_depth=2):
        # Plot the tree using matplotlib
        plt.figure(figsize=(20,10))
        if tradeoffmodel is None:
            tradeoffmodel = Model.tradeoffmodel
        plot_tree(tradeoffmodel, 
                  feature_names=Model.X_columns, 
                  class_names=Model.classnames,
                  filled=True, 
                  rounded=True,
                  max_depth=max_depth) # Prevent too much of the tree from being generated
        if show_fig:
            plt.show()
        if save_fig:
            plt.savefig(f'{Model.output_path}/graphs/{Model.name}.png', bbox_inches="tight")
        plt.close()

def confusion_matrix_plotter(Model: Union[ISDecisionTreeClassification, ISLogisticRegression], matrix=None, matrix_path=None, save_fig=False, show_fig=False):
        if matrix is None and matrix is None:
            matrix = Model.cm
        elif matrix_path is not None:
            # Loading the matrix
            matrix = np.load(matrix_path)
        if matrix is None:
            try:
                matrix = np.load(f'{Model.output_path}/confusion_matrix.npy', allow_pickle=True)
            except FileNotFoundError as e:
                raise ValueError("Undeclared confusion matrix. Add in matrix or matrix path.")
        # Plot the tree using matplotlib
        plt.figure(figsize = (10,7))
        sn.heatmap(matrix, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        if show_fig:
            plt.show()
        if save_fig:
            plt.savefig(f'{Model.output_path}/graphs/confusion_matrix.png', bbox_inches="tight")
            plt.close()
        else:
            plt.close()

def run_model(Model, tradeoffmodel=None, ccp_alpha=None, print_report=False, save_files=True):
    # Split the data
    split_data(Model, full_model=True)

    if tradeoffmodel is not None:
        Model.tradeoffmodel = tradeoffmodel
        Model.y_pred = Model.tradeoffmodel.predict(Model.X_test)
    else:
        if isinstance(Model, ISDecisionTreeClassification):
            if ccp_alpha is not None:
                Model.tradeoffmodel = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
            else:
                Model.tradeoffmodel = DecisionTreeClassifier(random_state=0)
        elif isinstance(Model, ISLogisticRegression):
            Model.tradeoffmodel = LogisticRegression(solver='liblinear', random_state=0)
        
        # Run and time model
        start_time = time.time()
        Model.tradeoffmodel.fit(Model.X_train, Model.y_train)
        Model.y_pred = Model.tradeoffmodel.predict(Model.X_test)
        end_time = time.time()
        runtime = end_time - start_time

    if isinstance(Model, ISDecisionTreeClassification) or isinstance(Model, ISLogisticRegression):
        # Get Metrics
        Model.cm = confusion_matrix(Model.y_test, Model.y_pred)

        if print_report:
            report = classification_report(Model.y_test, Model.y_pred, zero_division=0, labels=Model.labels, target_names=Model.classnames)
            print(report)
        else:
            report = classification_report(Model.y_test, Model.y_pred, zero_division=0, output_dict=True, labels=Model.labels, target_names=Model.classnames)
            if tradeoffmodel is None:
                report['time'] = runtime # Add time to the report dictionary

    # Saving to a JSON file
    if save_files:
        with open(f'{Model.output_path}/classification_report.json', 'w') as json_file:
            json.dump(report, json_file, indent=4)

        # Save confusion matrix
        np.save(f'{Model.output_path}/confusion_matrix.npy', Model.cm)

        # Convert X_test and y_test to DataFrames if they are not already
        X_test_df = pd.DataFrame(Model.X_test)
        y_test_df = pd.DataFrame(Model.y_test, columns=[f'Actual Class: {Model.target}'])
        y_pred_df = pd.DataFrame(Model.y_pred, columns=[f'Predicted Class: {Model.target}'])

        # Concatenate the DataFrames
        combined_df = pd.concat([X_test_df, y_test_df, y_pred_df], axis=1)
        combined_df.to_csv(f'{Model.output_path}/predictions.csv', index=False)

        # Save the model
        save_model(Model, f'{Model.output_path}/{Model.name}_model.pkl')

def calculate_confusion_matrix(Model, y_test=None, y_pred=None, return_matrix=False):
    if y_pred is None or y_test is None:
        y_pred = Model.y_pred
        y_test = Model.y_test
        if y_pred is None or y_test is None:
            file_path = f'{Model.output_path}/predictions.csv'
            load_prediction(Model, file_path)
    
    Model.cm = confusion_matrix(y_test, y_pred)

    # Save confusion matrix
    np.save(f'{Model.output_path}/confusion_matrix.npy', Model.cm)

    if return_matrix:
        return Model.cm
    
def calculate_shap_values(Model, sample_size=1000, return_values=False):
    # Select a random sample from Model.X
    if sample_size < len(Model.X):
        Model.X_sample = Model.X.sample(n=sample_size, random_state=42)
    else:
        Model.X_sample = Model.X
    
    if isinstance(Model, ISDecisionTreeClassification):
        explainer = shap.TreeExplainer(Model.tradeoffmodel)
        shap_values = explainer(Model.X_sample)  # Obtain SHAP values as an Explanation object
    if isinstance(Model, ISLogisticRegression):
        explainer = shap.LinearExplainer(Model.tradeoffmodel, Model.X_train)
        shap_values = explainer(Model.X_sample)  # Obtain SHAP values as an Explanation object

    shap_values_path = f'{Model.output_path}/shap_values.npy'
    np.save(shap_values_path, shap_values)

    if return_values:
        return shap_values
    
def plot_shap_values(Model, shap_values=None, shap_explainer_list=None):
    if shap_explainer_list is not None:
        Model.shap_explainer_list = shap_explainer_list
    if shap_values is not None:
        Model.shap_values = shap_values
    if Model.shap_is_list == False:
        plot_shap_values_one_target(Model, Model.shap_values)
    else:
        for i, name in enumerate(Model.classnames):
            plot_shap_values_one_target(Model, Model.shap_explainer_list[i], name)

def plot_shap_values_one_target(Model, shap_values, classname=None):

    if classname is not None:
        # Ensure proper naming protocol was used
        classname = classname.replace(' ', '_')

    # Generate the SHAP bar plot and capture the axes
    ax = shap.plots.bar(shap_values, show=False)
    
    # Get the figure from the axes
    fig = ax.figure

    # Set the figure size to be wider
    fig = plt.gcf()
    fig.set_size_inches(20, 6)  # Adjust the width and height as needed

    # Save the figure
    if classname is not None:
        fig.savefig(f'{Model.output_path}/graphs/{classname}/shap_bar_plot.png', bbox_inches="tight")
        plt.close(fig)
    else:
        fig.savefig(f'{Model.output_path}/graphs/shap_bar_plot.png', bbox_inches="tight")
        plt.close(fig)

    # Scatter plot for each specific feature
    for i, column in enumerate(Model.X_columns):
        # Create a figure and axis
        fig, ax = plt.subplots()
        if column == 'learning style':
            shap.plots.scatter(shap_values[:, 2], show=False, ax=ax)
            ax.set_xlim(0.488, 0.51)
        else:
            shap.plots.scatter(shap_values[:, i], show=False, ax=ax)

        # Adjust the size
        fig = plt.gcf()
        fig.set_size_inches(15, 6)

        # Save the figure
        if classname is not None:
            plt.savefig(f'{Model.output_path}/graphs/{classname}/feature_scatter_plots/{column}.png', bbox_inches="tight")
            plt.close(fig)
        else:
            plt.savefig(f'{Model.output_path}/graphs/feature_scatter_plots/{column}.png', bbox_inches="tight")
            plt.close(fig)

    # Heatmap plot
    ax = shap.plots.heatmap(shap_values, show=False)

    # Get the figure from the axes
    fig = ax.figure

    # Set the figure size to be wider
    fig = plt.gcf()
    fig.set_size_inches(15, 6)

    # Save the figure
    if classname is not None:
        fig.savefig(f'{Model.output_path}/graphs/{classname}/shap_heatmap.png', bbox_inches="tight")
        plt.close(fig)
    else:
        fig.savefig(f'{Model.output_path}/graphs/shap_heatmap.png', bbox_inches="tight")
        plt.close(fig)

    # Bee swarm plot
    ax = shap.plots.beeswarm(shap_values, show=False)

    # Get the figure from the axes
    fig = ax.figure

    # Set the figure size to be wider
    fig = plt.gcf()
    fig.set_size_inches(22, 6)  # Adjust the width and height as needed

    # Save the figure
    if classname is not None:
        fig.savefig(f'{Model.output_path}/graphs/{classname}/shap_bee_swarm_plot.png', bbox_inches="tight")
        plt.close(fig)
    else:
        fig.savefig(f'{Model.output_path}/graphs/shap_bee_swarm_plot.png', bbox_inches="tight")
        plt.close(fig)

    # Violin plot
    fig = plt.figure()
    shap.plots.violin(shap_values, show=False)

    # Set the figure size to be wider
    fig = plt.gcf()
    fig.set_size_inches(22, 6)  # Adjust the width and height as needed

    # Save the figure
    if classname is not None:
        plt.savefig(f'{Model.output_path}/graphs/{classname}/shap_violin_plot.png', bbox_inches="tight")
        plt.close(fig)
    else:
        plt.savefig(f'{Model.output_path}/graphs/shap_violin_plot.png', bbox_inches="tight")
        plt.close(fig)

    # Ensure all figures were closed
    plt.close('all')

def load_shap_values(Model, file_path, return_values=False):
    if Model.shap_is_list == False:
        # Load the .npy file
        Model.shap_values = np.load(file_path, allow_pickle=True)
        
        # Extract the SHAP values, base values, and data
        values = []
        base_values = []
        data = []

        for element in Model.shap_values:
            elem_values = []
            elem_data = []
            for item in range(len(element)):
                elem_values.append(element[item].values)
                elem_data.append(element[item].data)
            values.append(elem_values)
            base_values.append(element[0].base_values)
            data.append(elem_data)

        # Convert lists to numpy arrays
        values = np.array(values)
        base_values = np.array(base_values)
        data = np.array(data)

        # Create an Explanation object
        Model.shap_values = shap.Explanation(values=values,base_values=base_values, data=data, feature_names=Model.X_columns)

        if return_values:
            return Model.shap_values
    else:
        # Load the .npy file
        shap_values = np.load(file_path, allow_pickle=True)

        Model.shap_explainer_list = []

        for i in range(len(Model.classnames)):
            # Extract the SHAP values, base values, and data
            values = []
            base_values = []
            data = []
            for element in shap_values:
                elem_values = []
                elem_data = []
                for item in range(len(element)):
                    elem_values.append(element[item][i].values)
                    elem_data.append(element[item][i].data)
                values.append(elem_values)
                base_values.append(element[0][i].base_values)
                data.append(elem_data)

            # Convert lists to numpy arrays
            values = np.array(values)
            base_values = np.array(base_values)
            data = np.array(data)

            # Create an Explanation object
            shap_value_explainer = shap.Explanation(values=values,base_values=base_values, data=data, feature_names=Model.X_columns)

            # Add object to list of explainers, one per target
            Model.shap_explainer_list.append(shap_value_explainer)

        if return_values:
            return Model.shap_explainer_list

def load_prediction(Model, file_path, return_predictions=False):
    """
    Loads in the y prediction and y test values as two dataframes
    """
    try:
        prediction = pd.read_csv(file_path, converters={
            f'Actual Class: {Model.target}': literal_eval,
            f'Predicted Class: {Model.target}': literal_eval
        })
    except KeyError: # Legacy CSV style
        prediction = pd.read_csv(file_path, converters={
            f'{Model.target}': literal_eval,
            f'Predicted Class: {Model.target}': literal_eval
        })

    Model.y_pred = prediction.iloc[:, -1].to_frame()
    Model.test = prediction.iloc[:, -2].to_frame()

    if return_predictions:
        return Model.y_pred

def save_model(Model, file_path):
    joblib.dump(Model.tradeoffmodel, file_path)

def load_model(Model, file_path, return_file=False):
    try:
        Model.tradeoffmodel = joblib.load(file_path)
    except Exception as e:
        logging.error(f"Joblib failed to load the model: {e}")
        logging.info("Attempting to load with pickle...")
        try:
            with open(file_path, 'rb') as f:
                Model.tradeoffmodel = pickle.load(f)
        except Exception as pickle_e:
            logging.error(f"Pickle also failed to load the model: {pickle_e}")
            raise pickle_e
        
    if return_file:
        return Model.tradeoffmodel