import logging
from ast import literal_eval
import pandas as pd
from typing import Union
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, median_absolute_error
import matplotlib.pyplot as plt
import seaborn as sn
import joblib
import shap
import numpy as np
import json
import time
import os
import pickle

# TODO: Fix error where SHAP values must be saved and reloaded for the graphing to work

# Set the pandas option to avoid silent downcasting
pd.set_option('future.no_silent_downcasting', True)

# Classification Models

class ISDecisionTreeClassification:
    def __init__(self, privatization_type: str, RNN_model: str, target: str, data=None, data_path=None, output_path=None, X_columns=None, classnames= None, model_ran=False):
        if model_ran:
            self.data = None
        else:
            # Either data or data path must be declared is the model has not already been run
            if data is None and data_path is None:
                raise ValueError("At least one of 'data' or 'data_path' must be provided.")
            
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
        
        # Initialize inputs
        self.privatization_type = privatization_type
        self.RNN_model = RNN_model
        self.target = target
        self.target_name = target.replace(' ', '_')
        self.target_name = self.target_name.replace('/', '_')

        # Output paths
        if output_path is not None:
            self.output_path = output_path
        else:
            logging.info("no output path given, so no data will be saved!")

        # Additional Attributes
        self.name = 'decision_tree_classifier'
        self.shap_is_list = True
        self.feature_importance = []
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
        self.shap_values = None
        self.feature_importance = None
        self.shap_explainer_list = None

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

class ISLogisticRegression:
    def __init__(self, privatization_type: str, RNN_model: str, target: str, data=None, data_path=None, output_path=None, X_columns=None, classnames=None, model_ran=False):
        if model_ran:
            self.data = None
        else:
            # Either data or data path must be declared is the model has not already been run
            if data is None and data_path is None:
                raise ValueError("At least one of 'data' or 'data_path' must be provided.")
            
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
        
        # Initialize inputs
        self.privatization_type = privatization_type
        self.RNN_model = RNN_model
        self.target = target
        self.target_name = target.replace(' ', '_')
        self.target_name = self.target_name.replace('/', '_')

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
        self.shap_values = None
        self.feature_importance = None
        self.shap_explainer_list = None

        if classnames is not None:
            self.classnames = classnames
            # Assumes multi-target for SHAP explainer object construction
            self.shap_is_list = True
            self.feature_importance = []
        elif target == 'ethnoracial group':
            self.classnames = [
                'European American or white', 'Latino/a/x American', 'African American or Black', 'Asian American', 'Multiracial', 'American Indian or Alaska Native', 'Pacific Islander'
            ]
            self.shap_is_list = True
            self.feature_importance = []
        elif target == 'gender':
            self.classnames = [
                'Female', "Male", 'Nonbinary'
            ]
            self.shap_is_list = True
            self.feature_importance = []
        elif target == 'international status':
            self.classnames = [
                'Domestic', 'International'
            ]
            self.shap_is_list = False
            self.feature_importance = None
        elif target == 'socioeconomic status':
            self.classnames = [
                'In poverty', 'Near poverty', 'Lower-middle income', 'Middle income', 'Higher income'
            ]
            self.shap_is_list = True
            self.feature_importance = []
        else:
            raise ValueError(f"Incorrect target column name {target} for a classification model")
        self.labels = list(range(len(self.classnames)))

class ISRandomForestClassification:
    def __init__(self, privatization_type: str, RNN_model: str, target: str, data=None, data_path=None, output_path=None, X_columns=None, classnames=None, model_ran=False):
        if model_ran:
            self.data = None
        else:
            # Either data or data path must be declared is the model has not already been run
            if data is None and data_path is None:
                raise ValueError("At least one of 'data' or 'data_path' must be provided.")
            
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
        
        # Initialize inputs
        self.privatization_type = privatization_type
        self.RNN_model = RNN_model
        self.target = target
        self.target_name = target.replace(' ', '_')
        self.target_name = self.target_name.replace('/', '_')

        # Output paths
        if output_path is not None:
            self.output_path = output_path
        else:
            logging.info("no output path given, so no data will be saved!")

        # Additional Attributes
        self.name = 'random_forest_classifier'
        self.shap_is_list = True
        self.feature_importance = []
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
        self.shap_values = None
        self.feature_importance = None
        self.shap_explainer_list = None

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

# Regressification Models

class ISDecisionTreeRegressification:
    def __init__(self, privatization_type: str, RNN_model: str, target: str, data=None, data_path=None, output_path=None, X_columns=None, model_ran=False):
        if model_ran:
            self.data = None
        else:
            # Either data or data path must be declared is the model has not already been run
            if data is None and data_path is None:
                raise ValueError("At least one of 'data' or 'data_path' must be provided.")
            
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
        
        # Initialize inputs
        self.privatization_type = privatization_type
        self.RNN_model = RNN_model
        self.target = target
        self.target_name = target.replace(' ', '_')
        self.target_name = self.target_name.replace('/', '_')

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
        self.shap_values = None
        self.feature_importance = None

        # Attributes for getting the best model
        self.models_data = None
        self.ccp_alphas = None
        self.ccp_alpha = None
        self.train_scores = None
        self.test_scores = None
        self.impurities = None
        self.node_counts = None
        self.depth = None

class ISLinearRegressification:
    def __init__(self, privatization_type: str, RNN_model: str, target: str, data=None, data_path=None, output_path=None, X_columns=None, model_ran=False):
        if model_ran:
            self.data = None
        else:
            # Either data or data path must be declared is the model has not already been run
            if data is None and data_path is None:
                raise ValueError("At least one of 'data' or 'data_path' must be provided.")
            
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
        
        # Initialize inputs
        self.privatization_type = privatization_type
        self.RNN_model = RNN_model
        self.target = target
        self.target_name = target.replace(' ', '_')
        self.target_name = self.target_name.replace('/', '_')

        # Output paths
        if output_path is not None:
            self.output_path = output_path
        else:
            logging.info("no output path given, so no data will be saved!")

        # Additional Attributes
        self.name = 'linear_regressifier'
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
        self.shap_values = None
        self.feature_importance = None

class ISRandomForestRegressification:
    def __init__(self, privatization_type: str, RNN_model: str, target: str, data=None, data_path=None, output_path=None, X_columns=None, model_ran=False):
        if model_ran:
            self.data = None
        else:
            # Either data or data path must be declared is the model has not already been run
            if data is None and data_path is None:
                raise ValueError("At least one of 'data' or 'data_path' must be provided.")
            
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
        
        # Initialize inputs
        self.privatization_type = privatization_type
        self.RNN_model = RNN_model
        self.target = target
        self.target_name = target.replace(' ', '_')
        self.target_name = self.target_name.replace('/', '_')

        # Output paths
        if output_path is not None:
            self.output_path = output_path
        else:
            logging.info("no output path given, so no data will be saved!")

        # Additional Attributes
        self.name = 'random_forest_regressifier'
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
        self.shap_values = None
        self.feature_importance = None

# Regression Models

class ISDecisionTreeRegression:
    def __init__(self, privatization_type: str, RNN_model: str, target: str, data=None, data_path=None, output_path=None, X_columns=None, model_ran=False):
        if model_ran:
            self.data = None
        else:
            # Either data or data path must be declared is the model has not already been run
            if data is None and data_path is None:
                raise ValueError("At least one of 'data' or 'data_path' must be provided.")
            
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
        
        # Initialize inputs
        self.privatization_type = privatization_type
        self.RNN_model = RNN_model
        self.target = target
        self.target_name = target.replace(' ', '_')
        self.target_name = self.target_name.replace('/', '_')

        # Output paths
        if output_path is not None:
            self.output_path = output_path
        else:
            logging.info("no output path given, so no data will be saved!")

        # Additional Attributes
        self.name = 'decision_tree_regressor'
        self.shap_is_list = False
        self.feature_importance = None
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
        self.final_data = None
        self.tradeoffmodel = None
        self.X_sample = None
        self.shap_values = None
        self.feature_importance = None
        self.r2 = None

        # Attributes for getting the best model
        self.models_data = None
        self.ccp_alphas = None
        self.ccp_alpha = None
        self.train_scores = None
        self.test_scores = None
        self.impurities = None
        self.node_counts = None
        self.depth = None

class ISLinearRegression:
    def __init__(self, privatization_type: str, RNN_model: str, target: str, data=None, data_path=None, output_path=None, X_columns=None, model_ran=False):
        if model_ran:
            self.data = None
        else:
            # Either data or data path must be declared is the model has not already been run
            if data is None and data_path is None:
                raise ValueError("At least one of 'data' or 'data_path' must be provided.")
            
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
        
        # Initialize inputs
        self.privatization_type = privatization_type
        self.RNN_model = RNN_model
        self.target = target
        self.target_name = target.replace(' ', '_')
        self.target_name = self.target_name.replace('/', '_')

        # Output paths
        if output_path is not None:
            self.output_path = output_path
        else:
            logging.info("no output path given, so no data will be saved!")

        # Additional Attributes
        self.name = 'linear_regressor'
        self.shap_is_list = False
        self.feature_importance = None
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
        self.final_data = None
        self.tradeoffmodel = None
        self.X_sample = None
        self.shap_values = None
        self.feature_importance = None
        self.m = None
        self.b = None
        self.r2 = None

class ISRandomForestRegression:
    def __init__(self, privatization_type: str, RNN_model: str, target: str, data=None, data_path=None, output_path=None, X_columns=None, model_ran=False):
        if model_ran:
            self.data = None
        else:
            # Either data or data path must be declared is the model has not already been run
            if data is None and data_path is None:
                raise ValueError("At least one of 'data' or 'data_path' must be provided.")
            
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
        
        # Initialize inputs
        self.privatization_type = privatization_type
        self.RNN_model = RNN_model
        self.target = target
        self.target_name = target.replace(' ', '_')
        self.target_name = self.target_name.replace('/', '_')

        # Output paths
        if output_path is not None:
            self.output_path = output_path
        else:
            logging.info("no output path given, so no data will be saved!")

        # Additional Attributes
        self.name = 'random_forest_regressor'
        self.shap_is_list = False
        self.feature_importance = None
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
        self.final_data = None
        self.tradeoffmodel = None
        self.X_sample = None
        self.shap_values = None
        self.feature_importance = None
        self.r2 = None

# General Functions

def make_folders(Model, output_path=None):
    if output_path is None:
        output_path = Model.output_path
    if Model.shap_is_list == False:
        directories = []
        directories.append(os.path.dirname(f'{output_path}/graphs/feature_scatter_plots/example.py'))
        if isinstance(Model, ISLinearRegression):
            directories.append(os.path.dirname(f'{output_path}/graphs/feature_linear_plots/example.py'))

        for directory in directories:
            # Create the directory if it doesn't exist
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
    else:
        for name in Model.classnames:
            name = name.replace(' ', '_')
            name = name.replace('/', '_')
            directory = os.path.dirname(f'{output_path}/graphs/{name}/feature_scatter_plots/example.py')

            # Create the directories if they doesn't exist
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

def split_data(Model, full_model=False):
        if isinstance(Model, ISDecisionTreeClassification) or isinstance(Model, ISLogisticRegression) or isinstance(Model, ISRandomForestClassification):
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

        elif isinstance(Model, ISDecisionTreeRegressification) or isinstance(Model, ISLinearRegressification) or isinstance(Model, ISRandomForestRegressification) or isinstance(Model, ISDecisionTreeRegression) or isinstance(Model, ISLinearRegression) or isinstance(Model, ISRandomForestRegression):
            if full_model:
                Model.final_data = Model.data
            else:
                Model.final_data = Model.data.sample(n=2000, random_state=1)
            # Set up X
            Model.X = Model.final_data[Model.X_columns]

        else:
            raise ValueError("Need a proper Model object")

        # Change the lists into just the elements within them if the element is a list otherwise just take the element
        for column in Model.X_columns:
            Model.X.loc[:, column] = Model.X[column].apply(lambda x: x[0] if isinstance(x, list) else x)

        if isinstance(Model, ISLogisticRegression) or isinstance(Model, ISLinearRegression) or isinstance(Model, ISLinearRegressification):
            # Fill NaN values with the mean of each column
            Model.X = Model.X.fillna(Model.X.mean())

            # Infer objects to avoid future warning
            Model.X = Model.X.infer_objects(copy=False)

        if isinstance(Model, ISLogisticRegression) or isinstance(Model, ISDecisionTreeClassification) or isinstance(Model, ISRandomForestClassification):
            # Set up y post stratification
            Model.y = Model.stratified_data[[Model.target]]

            # Change the doubles into integers (1.0 -> 1)
            Model.y = Model.y.astype(int)

        elif isinstance(Model, ISDecisionTreeRegressification) or isinstance(Model, ISLinearRegressification) or isinstance(Model, ISRandomForestRegressification) or isinstance(Model, ISDecisionTreeRegression) or isinstance(Model, ISLinearRegression) or isinstance(Model, ISRandomForestRegression):
            # Set up y post stratification
            Model.y = Model.final_data[[Model.target]]

            # Change the lists into just the elements within them if the element is a list otherwise just take the element
            for column in Model.y.columns:
                Model.y.loc[:, column] = Model.y[column].apply(lambda x: x[0] if isinstance(x, list) else (int(x) if not np.isnan(x) else x))

        else:
            raise ValueError("Need a proper Model object")

        # Test - Train Split
        Model.X_train, Model.X_test, Model.y_train, Model.y_test = train_test_split(Model.X, Model.y, train_size = 0.8, random_state = 1234)

        if isinstance(Model, ISLogisticRegression) or isinstance(Model, ISRandomForestRegression) or isinstance(Model, ISRandomForestRegressification) or isinstance(Model, ISRandomForestClassification):
            # Reshape ys to be 1-dimensional
            Model.y_train = Model.y_train.values.ravel()

def score(tradeoffsmodel, X, y, sample_weight=None):
    # Takes in values from a regressor model, but scores them based on a classification (round to the nearest whole number)

    y_pred = tradeoffsmodel.predict(X)

    # Round the predictions and convert to integers
    y_pred_rounded = np.round(y_pred).astype(int)

    return accuracy_score(y, y_pred_rounded, sample_weight=sample_weight)

def get_best_model(Model: Union[ISDecisionTreeClassification, ISDecisionTreeRegressification, ISDecisionTreeRegression], make_graphs=True, return_model=True, return_ccp_alpha=True, save_model:bool = True, show_fig=False, save_fig=True):
    # Split the data
    split_data(Model)

    # Generate the models with different ccp alphas
    if isinstance(Model, ISDecisionTreeClassification):
        clf = DecisionTreeClassifier(random_state=0)
    else:
        clf = DecisionTreeRegressor(random_state=0)
    path = clf.cost_complexity_pruning_path(Model.X_train, Model.y_train)
    Model.ccp_alphas, Model.impurities = path.ccp_alphas, path.impurities

    # Ensure all ccp_alphas are non-negative
    Model.ccp_alphas = [max(alpha, 0.0) for alpha in Model.ccp_alphas]

    # Fit the models
    clfs = []
    for ccp_alpha in Model.ccp_alphas:
        if isinstance(Model, ISDecisionTreeClassification):
            clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        else:
            clf = DecisionTreeRegressor(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(Model.X_train, Model.y_train)
        clfs.append(clf)
    Model.node_counts = [clf.tree_.node_count for clf in clfs]
    Model.depth = [clf.tree_.max_depth for clf in clfs]

    # Train the models
    if isinstance(Model, ISDecisionTreeRegressification):
        Model.train_scores = [score(clf, Model.X_train, Model.y_train) for clf in clfs]
        Model.test_scores = [score(clf, Model.X_test, Model.y_test) for clf in clfs]
    else:
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
            models.to_csv(f'{Model.output_path}/{Model.name}_models.csv', index=False)
        except OSError:
            make_folders(Model)
            try:
                models.to_csv(f'{Model.output_path}/{Model.name}_models.csv', index=False)
            except:
                raise OSError("The make_folders function is not working. Check to see if the outputs folder can be created.")

    models_sorted = models.sort_values(by='test scores', ascending=False)

    # Pull out the model that has the highest test score and make it the model for training
    Model.tradeoffmodel, Model.ccp_alpha = models_sorted.iloc[0, 0:2]

    if make_graphs:
        graph_impurities(Model, show_fig, save_fig)
        graph_nodes_and_depth(Model, show_fig, save_fig)
        if isinstance(Model, ISDecisionTreeRegression):
            graph_R2(Model, show_fig, save_fig)
        else:
            graph_accuracy(Model, show_fig, save_fig)

    # Return the best model and the ccp_alpha
    if return_model and return_ccp_alpha:
        return Model.tradeoffmodel, Model.ccp_alpha
    elif return_model:
        return Model.tradeoffmodel
    elif return_ccp_alpha:
        return_ccp_alpha

def graph_impurities(Model: Union[ISDecisionTreeClassification, ISDecisionTreeRegressification, ISDecisionTreeRegression], show_fig=False, save_fig=True):
    # Impurity vs Effective Alpha
    fig, ax = plt.subplots()
    ax.plot(Model.ccp_alphas[:-1], Model.impurities[:-1], marker="o", drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    if show_fig:
        plt.show()
    if save_fig:
        plt.savefig(f'{Model.output_path}/graphs/effective_alpha_vs_total_impurity.png')
    plt.close()

def graph_nodes_and_depth(Model: Union[ISDecisionTreeClassification, ISDecisionTreeRegressification, ISDecisionTreeRegression], show_fig=False, save_fig=True):
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
    if show_fig:
        plt.show()
    if save_fig:
        plt.savefig(f'{Model.output_path}/graphs/effective_alpha_vs_graph_nodes_and_depth.png')
    plt.close()

def graph_accuracy(Model: Union[ISDecisionTreeClassification, ISDecisionTreeRegressification], show_fig=False, save_fig=True):
    # Accuracy vs Alpha
    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(Model.ccp_alphas, Model.train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(Model.ccp_alphas, Model.test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    if show_fig:
        plt.show()
    if save_fig:
        plt.savefig(f'{Model.output_path}/graphs/effective_alpha_vs_accuracy.png')
    plt.close()

def graph_R2(Model: Union[ISDecisionTreeRegression], show_fig=False, save_fig=True):
    # Accuracy vs Alpha
    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("r squared")
    ax.set_title("R squared vs alpha for training and testing sets")
    ax.plot(Model.ccp_alphas, Model.train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(Model.ccp_alphas, Model.test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    if show_fig:
        plt.show()
    if save_fig:
        plt.savefig(f'{Model.output_path}/graphs/effective_alpha_vs_r2.png')
    plt.close()

def tree_plotter(Model: Union[ISDecisionTreeClassification, ISDecisionTreeRegressification, ISDecisionTreeRegression, ISRandomForestClassification, ISRandomForestRegressification, ISRandomForestRegression], tradeoffmodel=None, save_fig=False, show_fig=False, max_depth: int = 2):
        # Plot the tree using matplotlib
        plt.figure(figsize=(20,10))
        if tradeoffmodel is None:
            tradeoffmodel = Model.tradeoffmodel
        if isinstance(Model, ISDecisionTreeClassification):
            plot_tree(tradeoffmodel, 
                    feature_names=Model.X_columns, 
                    class_names=Model.classnames,
                    filled=True, 
                    rounded=True,
                    max_depth=max_depth) # Prevent too much of the tree from being generated
        elif isinstance(Model, ISDecisionTreeRegressification) or isinstance(Model, ISDecisionTreeRegression):
            plot_tree(tradeoffmodel, 
                    feature_names=Model.X_columns,
                    filled=True, 
                    rounded=True,
                    max_depth=max_depth) # Prevent too much of the tree from being generated
        elif isinstance(Model, ISRandomForestRegressification) or isinstance(Model, ISRandomForestRegression):
            plot_tree(tradeoffmodel.estimators_[0],
                  feature_names=Model.X_columns,
                  filled=True, 
                  rounded=True,
                  max_depth=max_depth) # Prevent too much of the tree from being generated
        elif isinstance(Model, ISRandomForestClassification):
            plot_tree(tradeoffmodel.estimators_[0],
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

def confusion_matrix_plotter(Model: Union[ISDecisionTreeClassification, ISLogisticRegression, ISRandomForestClassification, ISDecisionTreeRegressification, ISLinearRegressification, ISRandomForestRegressification], matrix: np.array = None, matrix_path=None, save_fig=False, show_fig=False):
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
        if isinstance(Model, ISDecisionTreeRegressification) or isinstance(Model, ISLinearRegressification) or isinstance(Model, ISRandomForestRegressification):
            plt.figure(figsize=(20, 15))
            sn.heatmap(matrix, annot=False, fmt='d')
        elif isinstance(Model, ISDecisionTreeClassification) or isinstance(Model, ISLogisticRegression):
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

def linear_regression_plotter(Model: Union[ISLinearRegression]):
    # Plot the model
    # TODO: Add capability to load metrics CSV to get the m,b,r2 values
    # TODO: Add connection to load_predictions CSV so that the X and y can be loaded
    for name in Model.X_columns:
        lr_plotter_one_target(Model, name, save_fig=True)

def lr_plotter_one_target(Model: Union[ISLinearRegression], column: str, tradeoffmodel=None, save_fig=False, show_fig=False):
        # Plot the regression using matplotlib
        plt.figure(figsize=(20,10))
        if tradeoffmodel is None:
            tradeoffmodel = Model.tradeoffmodel
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,7))

        X_col = Model.X[[column]]

        ax.scatter(X_col, Model.y, color='black')

        ax.plot(X_col, tradeoffmodel.predict(Model.X), color='red',linewidth=3)
        # TODO: could tradeoffmodel.predict(Model.X) be replaced with Model.y_pred?
        ax.grid(True,
                axis = 'both',
                zorder = 0,
                linestyle = ':',
                color = 'k')
        ax.tick_params(labelsize = 18)
        ax.set_xlabel('x', fontsize = 24)
        ax.set_ylabel('y', fontsize = 24)

        # Extract single values for formatting
        m_value = Model.m[0][0] if isinstance(Model.m, np.ndarray) else Model.m
        b_value = Model.b[0] if isinstance(Model.b, np.ndarray) else Model.b
        r2_value = Model.r2[0] if isinstance(Model.r2, np.ndarray) else Model.r2


        ax.set_title("Linear Regression Line with Intercept y = {:.2f}x + {:.2f} (R2 = {:.2f})".format(m_value, b_value, r2_value), fontsize = 16 )
        fig.tight_layout()
        if show_fig:
            plt.show()
        if save_fig:
            plt.savefig(f'{Model.output_path}/graphs/feature_linear_plots/{column}.png', bbox_inches="tight")
            plt.close(fig)
        else:
            plt.close(fig)

def run_model(Model, tradeoffmodel=None, ccp_alpha=None, print_report=False, save_files=True, print_results=False, n_estimators: int = 100, min_samples_split: int = 10):
    # Split the data
    split_data(Model, full_model=True)

    if tradeoffmodel is not None:
        Model.tradeoffmodel = tradeoffmodel
        Model.y_pred = Model.tradeoffmodel.predict(Model.X_test)
        if isinstance(Model, ISDecisionTreeRegressification) or isinstance(Model, ISLinearRegressification) or isinstance(Model, ISRandomForestRegressification):
            # Round the predictions and convert to integers
            Model.y_pred = np.round(Model.y_pred).astype(int)
    else:
        if isinstance(Model, ISDecisionTreeClassification):
            if ccp_alpha is not None:
                Model.tradeoffmodel = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
            else:
                Model.tradeoffmodel = DecisionTreeClassifier(random_state=0)
        elif isinstance(Model, ISLogisticRegression):
            Model.tradeoffmodel = LogisticRegression(solver='liblinear', random_state=0)
        elif isinstance(Model, ISDecisionTreeRegressification) or isinstance(Model, ISDecisionTreeRegression):
            if ccp_alpha is not None:
                Model.tradeoffmodel = DecisionTreeRegressor(random_state=0, ccp_alpha=ccp_alpha)
            else:
                Model.tradeoffmodel = DecisionTreeRegressor(random_state=0)
        elif isinstance(Model, ISLinearRegressification) or isinstance(Model, ISLinearRegression):
            Model.tradeoffmodel = LinearRegression(fit_intercept=True)
        elif isinstance(Model, ISRandomForestRegression) or isinstance(Model, ISRandomForestRegressification):
            Model.tradeoffmodel = RandomForestRegressor(n_estimators=n_estimators, min_samples_split=min_samples_split, random_state=0)
        elif isinstance(Model, ISRandomForestClassification):
            Model.tradeoffmodel = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, random_state=0)
        else:
            raise ValueError("Need a proper Model object")
        # Run and time model
        start_time = time.time()
        Model.tradeoffmodel.fit(Model.X_train, Model.y_train)
        Model.y_pred = Model.tradeoffmodel.predict(Model.X_test)
        if isinstance(Model, ISDecisionTreeRegressification) or isinstance(Model, ISLinearRegressification) or isinstance(Model, ISRandomForestRegressification):
            # Round the predictions and convert to integers
            Model.y_pred = np.round(Model.y_pred).astype(int)
        end_time = time.time()
        runtime = end_time - start_time

     # Get Equation of the line
    if isinstance(Model, ISLinearRegression):
        Model.m = Model.tradeoffmodel.coef_[0][0]
        Model.b = Model.tradeoffmodel.intercept_[0]

    # Get Metrics
    if isinstance(Model, ISDecisionTreeClassification) or isinstance(Model, ISLogisticRegression) or isinstance(Model, ISRandomForestClassification) or isinstance(Model, ISDecisionTreeRegressification) or isinstance(Model, ISLinearRegressification) or isinstance(Model, ISRandomForestRegressification):
        Model.cm = confusion_matrix(Model.y_test, Model.y_pred)

        if print_report:
            if isinstance(Model, ISDecisionTreeRegressification) or isinstance(Model, ISLinearRegressification) or isinstance(Model, ISRandomForestRegressification):
                report = classification_report(Model.y_test, Model.y_pred, zero_division=0)
            elif isinstance(Model, ISDecisionTreeClassification) or isinstance(Model, ISLogisticRegression) or isinstance(Model, ISRandomForestClassification):
                report = classification_report(Model.y_test, Model.y_pred, zero_division=0, labels=Model.labels, target_names=Model.classnames)
            print(report)
        else:
            if isinstance(Model, ISDecisionTreeRegressification) or isinstance(Model, ISLinearRegressification) or isinstance(Model, ISRandomForestRegressification):
                report = classification_report(Model.y_test, Model.y_pred, zero_division=0, output_dict=True)
            elif isinstance(Model, ISDecisionTreeClassification) or isinstance(Model, ISLogisticRegression) or isinstance(Model, ISRandomForestClassification):
                report = classification_report(Model.y_test, Model.y_pred, zero_division=0, output_dict=True, labels=Model.labels, target_names=Model.classnames)
            if tradeoffmodel is None:
                report['time'] = runtime # Add time to the report dictionary
    elif isinstance(Model, ISDecisionTreeRegression) or isinstance(Model, ISLinearRegression) or isinstance(Model, ISRandomForestRegression):
        # Get Metrics
        mse = mean_squared_error(Model.y_test, Model.y_pred, multioutput='raw_values')[0]
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(Model.y_test, Model.y_pred, multioutput='raw_values')[0]
        medae = median_absolute_error(Model.y_test, Model.y_pred)
        Model.r2 = r2_score(Model.y_test, Model.y_pred, multioutput='raw_values')[0]
        evs = explained_variance_score(Model.y_test, Model.y_pred, multioutput='raw_values')[0]
        mbd = np.mean(np.array(Model.y_pred) - np.array(Model.y_test)) # Mean Bias Deviation

        # Put the resutls into a dataframe
        if isinstance(Model, ISLinearRegression):
            results = {
                "Slope": [Model.m], "Y-intercept": [Model.b], "MSE": [mse], "RMSE": [rmse], "MAE": [mae], "MedAE": [medae], "R2": [Model.r2], "Explained Variance": [evs], "MBD": [mbd], "Runtime": [runtime]
            }
        elif isinstance(Model, ISDecisionTreeRegression) or isinstance(Model, ISRandomForestRegression):
            results = {
                "MSE": [mse], "RMSE": [rmse], "MAE": [mae], "MedAE": [medae], "R2": [Model.r2], "Explained Variance": [evs], "MBD": [mbd], "Runtime": [runtime]
            }
        
        results_df = pd.DataFrame(results)

        if print_results:
            print(results)
    
    else:
        raise ValueError("Need a proper Model object")

    # Saving to a JSON file
    if save_files:
        if isinstance(Model, ISDecisionTreeClassification) or isinstance(Model, ISLogisticRegression) or isinstance(Model, ISRandomForestClassification) or isinstance(Model, ISDecisionTreeRegressification) or isinstance(Model, ISLinearRegressification) or isinstance(Model, ISRandomForestRegressification):
            with open(f'{Model.output_path}/classification_report.json', 'w') as json_file:
                json.dump(report, json_file, indent=4)

            # Save confusion matrix
            np.save(f'{Model.output_path}/confusion_matrix.npy', Model.cm)

        elif isinstance(Model, ISDecisionTreeRegression) or isinstance(Model, ISLinearRegression) or isinstance(Model, ISRandomForestRegression):
            results_df.to_csv(f'{Model.output_path}/metrics.csv', index=False)

        else:
            raise ValueError("Need a proper Model object")

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
    
def calculate_shap_values(Model, sample_size: int = 1000, return_values=False, save_values=True):
    # Ensure the model has been run
    if Model.tradeoffmodel is None:
        raise ValueError("You need to run the model or load the model.")
    # Select a random sample from Model.X
    if sample_size < len(Model.X):
        Model.X_sample = Model.X.sample(n=sample_size, random_state=42)
    else:
        Model.X_sample = Model.X
    
    if isinstance(Model, ISDecisionTreeClassification) or isinstance(Model, ISDecisionTreeRegressification) or isinstance(Model, ISDecisionTreeRegression) or isinstance(Model, ISRandomForestRegression) or isinstance(Model, ISRandomForestClassification) or isinstance(Model, ISRandomForestRegressification):
        explainer = shap.TreeExplainer(Model.tradeoffmodel)
        Model.shap_values = explainer(Model.X_sample)  # Obtain SHAP values as an Explanation object
    elif isinstance(Model, ISLogisticRegression) or isinstance(Model, ISLinearRegression) or isinstance(Model, ISLinearRegressification):
        explainer = shap.LinearExplainer(Model.tradeoffmodel, Model.X_train)
        Model.shap_values = explainer(Model.X_sample)  # Obtain SHAP values as an Explanation object
    else:
        raise ValueError("Need a proper Model object")

    if save_values:
        shap_values_path = f'{Model.output_path}/shap_values.npy'
        np.save(shap_values_path, Model.shap_values)

    if return_values:
        return Model.shap_values
    
def plot_shap_values(Model, shap_values=None, shap_explainer_list=None, save_figs=True):
    if shap_explainer_list is not None:
        Model.shap_explainer_list = shap_explainer_list
    if shap_values is not None:
        Model.shap_values = shap_values
    if Model.shap_is_list == False:
        plot_shap_values_one_target(Model, Model.shap_values, save_figs=save_figs)
    else:
        for i, name in enumerate(Model.classnames):
            plot_shap_values_one_target(Model, Model.shap_explainer_list[i], name, save_figs=save_figs)

def plot_shap_values_one_target(Model, shap_values, classname=None, save_figs=True):

    if classname is not None:
        # Ensure proper naming protocol was used
        classname = classname.replace(' ', '_')
        classname = classname.replace('/', '_')

    # Generate the SHAP bar plot and capture the axes
    ax = shap.plots.bar(shap_values, show=False)
    
    # Get the figure from the axes
    fig = ax.figure

    # Set the figure size to be wider
    fig = plt.gcf()
    fig.set_size_inches(20, 6)  # Adjust the width and height as needed

    # Save the figure
    if save_figs:
        if classname is not None:
            fig.savefig(f'{Model.output_path}/graphs/{classname}/shap_bar_plot.png', bbox_inches="tight")
            plt.close(fig)
        else:
            fig.savefig(f'{Model.output_path}/graphs/shap_bar_plot.png', bbox_inches="tight")
            plt.close(fig)
    else:
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
        if save_figs:
            if classname is not None:
                plt.savefig(f'{Model.output_path}/graphs/{classname}/feature_scatter_plots/{column}.png', bbox_inches="tight")
                plt.close(fig)
            else:
                plt.savefig(f'{Model.output_path}/graphs/feature_scatter_plots/{column}.png', bbox_inches="tight")
                plt.close(fig)
        else:
            plt.close(fig)

    # Heatmap plot
    ax = shap.plots.heatmap(shap_values, show=False)

    # Get the figure from the axes
    fig = ax.figure

    # Set the figure size to be wider
    fig = plt.gcf()
    fig.set_size_inches(15, 6)

    # Save the figure
    if save_figs:
        if classname is not None:
            fig.savefig(f'{Model.output_path}/graphs/{classname}/shap_heatmap.png', bbox_inches="tight")
            plt.close(fig)
        else:
            fig.savefig(f'{Model.output_path}/graphs/shap_heatmap.png', bbox_inches="tight")
            plt.close(fig)
    else:
        plt.close(fig)

    # Bee swarm plot
    ax = shap.plots.beeswarm(shap_values, show=False)

    # Get the figure from the axes
    fig = ax.figure

    # Set the figure size to be wider
    fig = plt.gcf()
    fig.set_size_inches(22, 6)  # Adjust the width and height as needed

    # Save the figure
    if save_figs:
        if classname is not None:
            fig.savefig(f'{Model.output_path}/graphs/{classname}/shap_bee_swarm_plot.png', bbox_inches="tight")
            plt.close(fig)
        else:
            fig.savefig(f'{Model.output_path}/graphs/shap_bee_swarm_plot.png', bbox_inches="tight")
            plt.close(fig)
    else:
        plt.close(fig)

    # Violin plot
    fig = plt.figure()
    shap.plots.violin(shap_values, show=False)

    # Set the figure size to be wider
    fig = plt.gcf()
    fig.set_size_inches(22, 6)  # Adjust the width and height as needed

    # Save the figure
    if save_figs:
        if classname is not None:
            plt.savefig(f'{Model.output_path}/graphs/{classname}/shap_violin_plot.png', bbox_inches="tight")
            plt.close(fig)
        else:
            plt.savefig(f'{Model.output_path}/graphs/shap_violin_plot.png', bbox_inches="tight")
            plt.close(fig)
    else:
        plt.close(fig)

    # Ensure all figures were closed
    plt.close('all')

def load_shap_values(Model, file_path: str, return_values=False):
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

def load_prediction(Model, file_path: str, return_predictions=False):
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
    
def get_feature_importance(Model, shap_values=None, shap_explainer_list=None, save_values=True):
    if shap_explainer_list is not None:
        Model.shap_explainer_list = shap_explainer_list
    if shap_values is not None:
        Model.shap_values = shap_values
    if Model.shap_is_list == False:
        Model.feature_importance = get_single_feature_importance(Model, Model.shap_values, return_df=True)
    else:
        feature_importance_list = []
        names = []
        for i, name in enumerate(Model.classnames):
            feature_importance_list.append(get_single_feature_importance(Model, Model.shap_explainer_list[i], return_df=True))
            # Ensure proper naming protocol was used
            name = name.replace(' ', '_')
            name = name.replace('/', '_')
            names.append(name)
        Model.feature_importance = pd.concat(feature_importance_list, keys=names)

        # Reset index to turn multi-index into columns
        Model.feature_importance.reset_index(level=0, inplace=True)
        Model.feature_importance.rename(columns={'level_0': 'Target'}, inplace=True)
    
    if save_values:
        Model.feature_importance.to_csv(f'{Model.output_path}/feature_importance.csv', index=False)

def get_single_feature_importance(Model, shap_values, return_df=False):
    # Summarize the feature importance
    feature_importance = np.abs(shap_values.values).mean(axis=0)

    # Get feature names if available
    feature_names = shap_values.feature_names if shap_values.feature_names is not None else [f'Feature {i}' for i in range(len(feature_importance))]

    # Create a DataFrame to display feature importance
    import pandas as pd
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    # Return the DataFrame
    if return_df:
        return feature_importance_df
    
def load_feature_importance(Model, file_path: str, return_df=False):
    Model.feature_importance = pd.read_csv(file_path)
    
    if return_df:
        return Model.feature_importance

def save_model(Model, file_path: str):
    joblib.dump(Model.tradeoffmodel, file_path)

def load_model(Model, file_path: str, return_file=False):
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
    
def pipeline(Model, full_run=False, run_shap=False, plot_graphs=False, feature_importance=False, pipeline_list=None):
    """
    Pipeline function that runs the basic variations. Always saves all the files.
    """
    if pipeline_list is not None:
        for function in pipeline_list:
            try:
                function(Model)
            except TypeError:
                raise TypeError(f"{function} needs more than just the model as an input. This function cannot be used in a user generated pipeline.")
    else:
        # Ensure at least one pipeline was chosen
        if not (full_run or run_shap or plot_graphs or feature_importance):
            raise ValueError("At least one of 'full_run', 'run_shap', 'plot_graphs', or 'feature_importance' must be set to True or the pipeline will do nothing.")
        
        # Ensure the output path was declared
        if Model.output_path is None:
            raise ValueError("Pipeline function requires output path as all files are saved.")
        
        # Make the folders
        make_folders(Model)

        # The full pipeline
        if full_run:
            if isinstance(Model, ISDecisionTreeClassification) or isinstance(Model, ISDecisionTreeRegressification) or isinstance(Model, ISDecisionTreeRegression):
                get_best_model(Model)
            
            # Run the main model
            run_model(Model)

            if isinstance(Model, ISDecisionTreeClassification) or isinstance(Model, ISDecisionTreeRegressification) or isinstance(Model, ISDecisionTreeRegression) or isinstance(Model, ISRandomForestClassification) or isinstance(Model, ISRandomForestRegressification) or isinstance(Model, ISRandomForestRegression):
                tree_plotter(Model, save_fig=True)

            if isinstance(Model, ISDecisionTreeClassification) or isinstance(Model, ISDecisionTreeRegressification) or isinstance(Model, ISLogisticRegression) or isinstance(Model, ISLinearRegressification) or isinstance(Model, ISRandomForestClassification) or isinstance(Model, ISRandomForestRegressification):
                confusion_matrix_plotter(Model, save_fig=True)

            if isinstance(Model, ISLinearRegression):
                linear_regression_plotter(Model)

            # Get the SHAP values
            calculate_shap_values(Model)
            load_shap_values(Model, f'{Model.output_path}/shap_values.npy')
            plot_shap_values(Model)

            # Get feature importance
            get_feature_importance(Model)
        
        # The graphing pipeline
        if plot_graphs:

            if isinstance(Model, ISDecisionTreeClassification) or isinstance(Model, ISDecisionTreeRegressification) or isinstance(Model, ISDecisionTreeRegression) or isinstance(Model, ISRandomForestClassification) or isinstance(Model, ISRandomForestRegressification) or isinstance(Model, ISRandomForestRegression):
                try:
                    load_model(Model, f'{Model.output_path}/{Model.name}_model.pkl')
                except FileNotFoundError:
                    raise FileNotFoundError("You need to run the model or adjust the given output path.")
                tree_plotter(Model, save_fig=True)

            if isinstance(Model, ISDecisionTreeClassification) or isinstance(Model, ISDecisionTreeRegressification) or isinstance(Model, ISLogisticRegression) or isinstance(Model, ISLinearRegressification) or isinstance(Model, ISRandomForestClassification) or isinstance(Model, ISRandomForestRegressification):
                try:
                    load_prediction(Model, f'{Model.output_path}/predictions.csv')
                except FileNotFoundError:
                    raise FileNotFoundError("You need to run the model or adjust the given output path.")
                calculate_confusion_matrix(Model)
                confusion_matrix_plotter(Model, save_fig=True)

        # The SHAP values pipeline
        if run_shap:
            try:
                load_model(Model, f'{Model.output_path}/{Model.name}_model.pkl')
            except FileNotFoundError:
                raise FileNotFoundError("You need to run the model or adjust the given output path.")
            calculate_shap_values(Model)
            load_shap_values(Model, f'{Model.output_path}/shap_values.npy')
            plot_shap_values(Model)
            get_feature_importance(Model)

        # The feature importance pipeline
        if feature_importance:
            try:
                load_shap_values(Model, f'{Model.output_path}/shap_values.npy')
            except FileNotFoundError:
                raise FileNotFoundError("You need to calculate the SHAP values or adjust the given output path.")
            get_feature_importance(Model)