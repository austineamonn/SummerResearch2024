import pandas as pd
import numpy as np
import os
import sys
import logging
import time
import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, accuracy_score, 
                             classification_report, precision_score, recall_score, f1_score, cohen_kappa_score, 
                             roc_auc_score, log_loss, explained_variance_score, median_absolute_error)
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

# Add the SummerResearch2024 directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class ModelTrainer:
    def __init__(self, config, data_path):
        """
        Initialize the ModelTrainer with configuration and data path.
        """
        # Set up logging
        self.setup_logging(config["logging"])

        self.data = pd.read_csv(data_path, nrows=100)  # Load only first 100 rows to test for errors
        self.models = {}
        self.results = {}

    def setup_logging(self, logging_config):
        """
        Set up logging configuration.
        """
        # Set up the root logger
        logging.basicConfig(level=getattr(logging, logging_config["level"].upper()),
                            format=logging_config["format"],
                            handlers=[logging.StreamHandler(sys.stdout)])

        # Set up specific logger for this class
        self.logger = logging.getLogger('ModelTrainer')
        self.logger.setLevel(getattr(logging, logging_config["level"].upper()))

        # Configure TensorFlow logging
        tf_logger = logging.getLogger('tensorflow')
        tf_logger.setLevel(getattr(logging, logging_config["level"].upper()))
        tf_logger.propagate = True

        # Test logging configuration
        self.logger.info("Logger initialized successfully.")

    def preprocess_data(self):
        """
        Preprocess the data by handling missing values, scaling numerical features,
        and encoding categorical features.
        """
        target_columns = ['career aspirations', 'future topics', 'ethnoracial group', 'gender', 'international status']
        for col in target_columns:
            if col not in self.data.columns:
                raise ValueError(f"Missing target column: {col}")

        # Identify categorical and numerical columns excluding target columns
        categorical_cols = ['ethnoracial group', 'gender', 'international status']
        numerical_cols = ['learning style', 'gpa', 'student semester', 'major', 'previous courses', 'course types', 'course subjects', 'subjects of interest', 'extracurricular activities', 'career aspirations', 'future topics']

        self.logger.info(f"Categorical columns: {categorical_cols}")
        self.logger.info(f"Numerical columns: {numerical_cols}")

        # Define the preprocessing pipeline for numerical and categorical data
        numerical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, numerical_cols),
                ('cat', categorical_pipeline, categorical_cols)
            ])

        # Fit and transform the data using the preprocessor
        X_transformed = self.preprocessor.fit_transform(self.data)

        # Store column names for later use
        num_col_names = numerical_cols
        cat_col_names = self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols).tolist()
        all_col_names = num_col_names + cat_col_names
        self.target_columns = ['career aspirations', 'future topics'] + cat_col_names

        # Convert the resulting numpy array to a DataFrame
        X_transformed_df = pd.DataFrame(X_transformed, columns=all_col_names)

        # Separate features and target columns
        self.X = X_transformed_df.drop(columns=self.target_columns, errors='ignore')
        self.y_continuous = X_transformed_df[['career aspirations', 'future topics']]
        self.y_categorical = X_transformed_df[cat_col_names]

        # Combine the continuous and categorical targets
        y_combined = pd.concat([self.y_continuous, self.y_categorical], axis=1)

        # Split the data into training and testing sets
        X_train_full, X_test_full, y_train, y_test = train_test_split(self.X, y_combined, test_size=0.2, random_state=42)

        # Separate the continuous and categorical targets after splitting
        self.y_cont_train = y_train[self.y_continuous.columns]
        self.y_cont_test = y_test[self.y_continuous.columns]
        self.y_cat_train = y_train[self.y_categorical.columns]
        self.y_cat_test = y_test[self.y_categorical.columns]

        # Store column names for later use
        self.feature_names = self.X.columns.tolist()

        # Store the processed training and testing data as a dataframe
        self.X_train = pd.DataFrame(X_train_full, columns=self.feature_names)
        self.X_test = pd.DataFrame(X_test_full, columns=self.feature_names)

        # Log shapes for debugging purposes
        self.logger.info(f"X_train shape: {self.X_train.shape}")
        self.logger.info(f"y_train (combined) shape: {y_train.shape}")
        self.logger.info(f"X_test shape: {self.X_test.shape}")
        self.logger.info(f"y_test (combined) shape: {y_test.shape}")
        self.logger.info(f"y_cont_train shape: {self.y_cont_train.shape}")
        self.logger.info(f"y_cont_test shape: {self.y_cont_test.shape}")
        self.logger.info(f"y_cat_train shape: {self.y_cat_train.shape}")
        self.logger.info(f"y_cat_test shape: {self.y_cat_test.shape}")

    def define_models(self):
        """
        Define the models for both regression and classification tasks.
        """
        self.regression_models = {
            #'Linear Regression': MultiOutputRegressor(estimator=LinearRegression()),
            'Decision Tree Regressor': MultiOutputRegressor(estimator=DecisionTreeRegressor()),
            'Random Forest Regressor': MultiOutputRegressor(estimator=RandomForestRegressor(random_state=42)),
            #"Neural Network Regressor": "Neural Network"
        }

        self.classification_models = {
            'Decision Tree Classifier': MultiOutputClassifier(estimator=DecisionTreeClassifier()),
            'Random Forest Classifier': MultiOutputClassifier(estimator=RandomForestClassifier(random_state=42)),
            #'Logistic Regression': MultiOutputClassifier(estimator=LogisticRegression(max_iter=1000)),
            #"Neural Network Classifier": "Neural Network"
        }

    def create_nn_regression(self, input_shape, output_shape):
        """
        Create a neural network model for regression.
        """
        inputs = Input(shape=(input_shape,))
        x = Dense(64, activation='relu')(inputs)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(2, activation='linear')(x)
        model = Model(inputs, outputs)
        metrics=['mae', 'mse'] * output_shape
        model.compile(optimizer=Adam(), loss='mse', metrics=metrics)
        return model

    def create_nn_classification(self, input_shape, output_shape):
        inputs = Input(shape=(input_shape,))
        x = Dense(64, activation='relu')(inputs)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        if isinstance(output_shape, int):
            output_shape = [output_shape]
        outputs = [Dense(output_dim, activation='softmax')(x) for output_dim in output_shape]
        model = Model(inputs, outputs)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score'] * output_shape
            
        model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=metrics)
        return model
    
    def make_predictions(self, model, X_test):
        """
        Preprocess the new data and make predictions using the trained model.
        """
        y_pred = model.predict(X_test)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        return y_pred

    def train_and_evaluate_classification_model(self, model, scenario, y_train, y_test, name):
        self.logger.info(f"Training {name} on {scenario}...")
        try:
            # Log initial states and perform checks
            self.logger.debug(f"Model: {model}")
            self.logger.debug(f"Scenario: {scenario}")
            self.logger.debug(f"y_train shape: {y_train.shape}")
            self.logger.debug(f"y_test shape: {y_test.shape}")
            self.logger.debug(f"y_train columns: {y_train.columns}")
            self.logger.debug(f"y_test columns: {y_test.columns}")

            if y_train is None or y_test is None:
                self.logger.error(f"y_train or y_test is None for {name} on {scenario}")
                return

            if model is None:
                self.logger.error(f"Model {name} is None for {scenario}")
                return

            start_time = time.time()
            self.logger.info(f"Feature names before fitting: {self.X_train.columns.tolist()}")
            if "Neural Network" in name:
                y_train_list = [y_train[col].values for col in y_train.columns]
                model.fit(self.X_train, y_train_list, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
                y_pred_list = model.predict(self.X_test)
                if isinstance(y_pred_list, list):
                    y_pred = np.column_stack([
                        np.argmax(pred, axis=1) if pred.ndim > 1 else pred 
                        for pred in y_pred_list
                    ])
                else:
                    if y_pred_list.ndim > 1:
                        y_pred = np.argmax(y_pred_list, axis=1)
                    else:
                        y_pred = y_pred_list  # Handle the 1-dimensional case directly
            elif "Logistic Regression" in name:
                model.fit(self.X_train, y_train)
                X_test_values = self.X_test.values
                y_pred = self.make_predictions(model, X_test_values)
                if y_pred.ndim == 1:
                    y_pred = y_pred.reshape(-1, 1)
            else:
                model.fit(self.X_train, y_train)
                y_pred = self.make_predictions(model, self.X_test)
                if y_pred.ndim == 1:
                    y_pred = y_pred.reshape(-1, 1)

            end_time = time.time()
            runtime = end_time - start_time

            # Logging predictions
            self.logger.debug(f"y_pred shape: {y_pred.shape}")
            self.logger.debug(f"y_pred: {y_pred}")

            acc = [accuracy_score(y_test.iloc[:, i], y_pred[:, i]) if y_pred.shape[1] > 1 else accuracy_score(y_test, y_pred) for i in range(y_test.shape[1])]
            precision = [precision_score(y_test.iloc[:, i], y_pred[:, i], average='weighted', zero_division=0) if y_pred.shape[1] > 1 else precision_score(y_test, y_pred, average='weighted', zero_division=0) for i in range(y_test.shape[1])]
            recall = [recall_score(y_test.iloc[:, i], y_pred[:, i], average='weighted', zero_division=0) if y_pred.shape[1] > 1 else recall_score(y_test, y_pred, average='weighted', zero_division=0) for i in range(y_test.shape[1])]
            f1 = [f1_score(y_test.iloc[:, i], y_pred[:, i], average='weighted', zero_division=0) if y_pred.shape[1] > 1 else f1_score(y_test, y_pred, average='weighted', zero_division=0) for i in range(y_test.shape[1])]
            kappa = [cohen_kappa_score(y_test.iloc[:, i], y_pred[:, i]) if y_pred.shape[1] > 1 else cohen_kappa_score(y_test, y_pred) for i in range(y_test.shape[1])]
            roc_auc = []
            for i in range(y_test.shape[1]):
                if len(np.unique(y_test.iloc[:, i])) > 1:
                    known_labels = np.unique(y_test.iloc[:, i])
                    lb = LabelBinarizer()
                    lb.fit(known_labels)
                    y_test_binarized = lb.transform(y_test.iloc[:, i])
                    y_pred_binarized = lb.transform(y_pred[:, i])
                    if y_pred.shape[1] > 1:
                        roc_auc.append(roc_auc_score(y_test_binarized, y_pred_binarized, average='weighted', multi_class='ovo'))
                    else:
                        roc_auc.append(roc_auc_score(y_test_binarized, y_pred_binarized, average='weighted', multi_class='ovo'))
                else:
                    roc_auc.append(None)
            logloss = [log_loss(y_test.iloc[:, i], y_pred[:, i]) if y_pred.shape[1] > 1 else log_loss(y_test, y_pred) for i in range(y_test.shape[1])]
            clf_report = classification_report(y_test, y_pred, output_dict=True) if y_pred.shape[1] > 1 else classification_report(y_test, y_pred, output_dict=True)

            self.results[f"{name} on {scenario}"] = {
                "Accuracy": acc, "Precision": precision, "Recall": recall, "F1 Score": f1, 
                "Cohen's Kappa": kappa, "ROC AUC": roc_auc, "Log Loss": logloss, "Classification Report": clf_report, "Runtime": runtime
            }

            # Save the model
            if "Neural Network" in name:
                model.save(f'outputs/{name}_{scenario}_model.keras')
            else:
                joblib.dump(model, f'outputs/{name}_{scenario}_model.joblib')

            self.calculate_and_save_shap_values(model, scenario, name)
        except Exception as e:
            self.logger.error(f"Error training {name} on {scenario}: {e}")

    def train_and_evaluate_regression_model(self, model, scenario, y_train, y_test, name):
        self.logger.info(f"Training {name} on {scenario}...")
        try:
            start_time = time.time()
            self.logger.info(f"Feature names before fitting: {self.X_train.columns.tolist()}")
            if "Neural Network" in name:
                model.fit(self.X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
                y_pred = model.predict(self.X_test)
            elif "Linear Regression" in name:
                model.fit(self.X_train, y_train)
                X_test_values = self.X_test.values
                y_pred = self.make_predictions(model, X_test_values)
            else:
                model.fit(self.X_train, y_train)
                y_pred = self.make_predictions(model, self.X_test)

            end_time = time.time()
            runtime = end_time - start_time

            mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
            medae = median_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred, multioutput='raw_values')
            evs = explained_variance_score(y_test, y_pred, multioutput='raw_values')
            mbd = np.mean(y_pred - y_test)
            self.results[f"{name} on {scenario}"] = {
                "MSE": mse.tolist(), "RMSE": rmse.tolist(), "MAE": mae.tolist(), "MedAE": medae.tolist(), 
                "R2": r2.tolist(), "Explained Variance": evs.tolist(), "MBD": mbd.tolist(), "Runtime": runtime
            }
            # Save the model
            if "Neural Network" in name:
                model.save(f'outputs/{name}_{scenario}_model.keras')
            else:
                joblib.dump(model, f'outputs/{name}_{scenario}_model.joblib')
            self.calculate_and_save_shap_values(model, scenario, name)
        except Exception as e:
            self.logger.error(f"Error training {name} on {scenario}: {e}")

    def calculate_and_save_shap_values(self, model, scenario, name):
        """
        Calculate and save SHAP values for model interpretation.
        """
        try:
            if "Tree" in name or "Forest" in name:
                explainer = shap.TreeExplainer(model.estimators_[0] if hasattr(model, 'estimators_') else model)
                shap_values = explainer.shap_values(self.X_test)
            else:
                explainer = shap.KernelExplainer(model.predict, self.X_train)
                shap_values = explainer.shap_values(self.X_test, nsamples=100)

            if isinstance(shap_values, list):
                for i, shap_val in enumerate(shap_values):
                    if len(shap_val.shape) == 2:
                        shap_values_df = pd.DataFrame(shap_val, columns=[f"Feature_{j}" for j in range(self.X_test.shape[1])])
                        shap_values_df.to_csv(f'outputs/{name}_{scenario}_shap_values_class_{i}.csv', index=False)
                    elif len(shap_val.shape) == 3:
                        for k in range(shap_val.shape[2]):
                            shap_values_df = pd.DataFrame(shap_val[:, :, k], columns=[f"Feature_{j}" for j in range(self.X_test.shape[1])])
                            shap_values_df.to_csv(f'outputs/{name}_{scenario}_shap_values_output_{k}_class_{i}.csv', index=False)
            else:
                if len(shap_values.shape) == 2:
                    shap_values_df = pd.DataFrame(shap_values, columns=[f"Feature_{i}" for i in range(self.X_test.shape[1])])
                    shap_values_df.to_csv(f'outputs/{name}_{scenario}_shap_values.csv', index=False)
                elif len(shap_values.shape) == 3:
                    for i in range(shap_values.shape[2]):
                        shap_values_df = pd.DataFrame(shap_values[:, :, i], columns=[f"Feature_{j}" for j in range(self.X_test.shape[1])])
                        shap_values_df.to_csv(f'outputs/{name}_{scenario}_shap_values_output_{i}.csv', index=False)
        except Exception as e:
            self.logger.error(f"Error calculating SHAP values for {name} on {scenario}: {e}")

    def train_and_evaluate_models(self):
        """
        Train and evaluate all defined models on specified scenarios.
        """
        self.logger.debug("Starting train_and_evaluate_models")

        # Log the state of the datasets
        self.logger.debug(f"y_cont_train: {self.y_cont_train}")
        self.logger.debug(f"y_cat_train: {self.y_cat_train}")
        self.logger.debug(f"y_cont_test: {self.y_cont_test}")
        self.logger.debug(f"y_cat_test: {self.y_cat_test}")
        self.logger.debug(f"y_continuous: {self.y_continuous}")
        self.logger.debug(f"y_categorical: {self.y_categorical}")

        # Check if any of the main datasets are None
        if self.y_cont_train is None or self.y_cat_train is None or self.y_cont_test is None or self.y_cat_test is None:
            self.logger.error("One or more of the training or testing datasets are None")
            return

        cont_scenarios = {
            "All Targets": (pd.concat([self.y_cont_train, self.y_cat_train], axis=1), 
                            pd.concat([self.y_cont_test, self.y_cat_test], axis=1)),
            "Continuous Targets": (self.y_cont_train, self.y_cont_test),
        }

        cat_scenarios = {
            "All Targets": (pd.concat([self.y_cont_train, self.y_cat_train], axis=1), 
                            pd.concat([self.y_cont_test, self.y_cat_test], axis=1)),
            "Categorical Targets": (self.y_cat_train, self.y_cat_test)
        }

        # Include individual targets in scenarios
        for target in self.y_continuous.columns:
            if self.y_cont_train[[target]].isnull().values.any() or self.y_cont_test[[target]].isnull().values.any():
                self.logger.error(f"Target {target} in continuous data contains None values")
                continue
            cont_scenarios[f"Target {target}"] = (self.y_cont_train[[target]], self.y_cont_test[[target]])
        for target in self.y_categorical.columns:
            if self.y_cat_train[[target]].isnull().values.any() or self.y_cat_test[[target]].isnull().values.any():
                self.logger.error(f"Target {target} in categorical data contains None values")
                continue
            cat_scenarios[f"Target {target}"] = (self.y_cat_train[[target]], self.y_cat_test[[target]])

        """These are the categorical scenario keys: dict_keys(['All Targets', 'Categorical Targets', 'Target ethnoracial group_0', 'Target ethnoracial group_1', 'Target ethnoracial group_2', 'Target ethnoracial group_3', 'Target ethnoracial group_4', 'Target gender_0', 'Target gender_1', 'Target gender_2', 'Target international status_0', 'Target international status_1'])"""

        # Check for None models in regression and classification models
        for name, model in self.regression_models.items():
            if model is None:
                self.logger.error(f"Regression model {name} is None")
        for name, model in self.classification_models.items():
            if model is None:
                self.logger.error(f"Classification model {name} is None")

        # Train and evaluate continuous models
        for scenario, (y_train, y_test) in cont_scenarios.items():
            self.logger.debug(f"Processing continous scenario: {scenario}")
            self.logger.debug(f"y_train: {y_train}")
            self.logger.debug(f"y_test: {y_test}")
            if y_train is None or y_test is None:
                self.logger.error(f"Scenario {scenario} has None values: y_train={y_train}, y_test={y_test}")
                continue

            for name, model in self.regression_models.items():
                if model is None:
                    self.logger.error(f"Skipping training for regression model {name} as it is None")
                    continue
                elif model == 'Neural Network':
                    model = self.create_nn_regression(self.X_train.shape[1], y_train.shape[1])
                try:
                    self.train_and_evaluate_regression_model(model, scenario, y_train.iloc[:, :self.y_cont_train.shape[1]], y_test.iloc[:, :self.y_cont_test.shape[1]], name)
                except Exception as e:
                    self.logger.error(f"Error in training regression model {name} for scenario {scenario}: {e}")
        
        # Train and evaluate categorical models
        for scenario, (y_train, y_test) in cat_scenarios.items():
            self.logger.debug(f"Processing categorical scenario: {scenario}")
            self.logger.debug(f"y_train: {y_train}")
            self.logger.debug(f"y_test: {y_test}")
            if y_train is None or y_test is None:
                self.logger.error(f"Scenario {scenario} has None values: y_train={y_train}, y_test={y_test}")
                continue

            for name, model in self.classification_models.items():
                if model is None:
                    self.logger.error(f"Skipping training for classification model {name} as it is None")
                    continue
                elif model == 'Neural Network':
                    model = self.create_nn_classification(self.X_train.shape[1], y_train.shape[1])
                try:
                    if "All Targets" in scenario:
                        y_train_cat = y_train.iloc[:, self.y_cont_train.shape[1]:]
                        y_test_cat = y_test.iloc[:, self.y_cont_test.shape[1]:]
                        if y_train_cat is not None and y_test_cat is not None:
                            self.train_and_evaluate_classification_model(model, scenario, y_train_cat, y_test_cat, name)
                    else:
                        self.train_and_evaluate_classification_model(model, scenario, y_train, y_test, name)
                except Exception as e:
                    self.logger.error(f"Error in training classification model {name} for scenario {scenario}: {e}")

    def save_results(self):
        if not isinstance(self.results, dict):
            self.logger.error("self.results is not a dictionary or is None.")
            return

        for model_name, metrics in self.results.items():
            try:
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        if not isinstance(value, list):
                            metrics[key] = [value]
                    df = pd.DataFrame.from_dict(metrics, orient='index').T
                else:
                    df = pd.DataFrame(metrics)
                df.to_csv(f'outputs/{model_name}_metrics.csv', index=False)
            except Exception as e:
                self.logger.error(f"Error saving results for {model_name}: {e}")

if __name__ == "__main__":
    # Import necessary dependencies
    from config import load_config

    # Load configuration
    config = load_config()

    # List of RNN models to run
    RNN_model_list = ['GRU1']#, 'LSTM1', 'Simple1']

    # List of Privatization Types to run
    privatization_types = ['NoPrivatization']#, 'Basic_DP', 'Basic_DP_LLC', 'Uniform', 'Uniform_LLC', 'Shuffling', 'Complete_Shuffling']

    for privatization_type in privatization_types:
        logging.info(f"Starting {privatization_type}")
        for RNN_model in RNN_model_list:
            logging.info(f"Starting {RNN_model}")

            # Get Data Paths
            data_path = config["running_model"]["completely_preprocessed_data_paths"][privatization_type][f"{RNN_model}_combined"]

            # Instantiate the class
            trainer = ModelTrainer(config, data_path)
            trainer.preprocess_data()
            trainer.define_models()
            trainer.train_and_evaluate_models()
            trainer.save_results()
            logging.info("Models trained and evaluated. Metrics, SHAP values, and models are saved to CSV files.")

            
            # save model, history, ypred, ytest, and metrics
            # tune hyperparameters on like 1,000 datavalues, then calculate best model with the full 100,000 values
            # code in parallel with n_jobs
            # use Pipelines

            # use tensorboard for neural networks

            # feature selection not needed for decision trees because they ignore unimportant features
            # first feature split is the most important feature for a decision tree

            # Logisting Regression - muticlass='ovr' or one vs rest for multilabel binary classification

            # Change the main file to be a follow along able jupyter notebook instead that just uses a small part of the data to explain the whole data process ../../saved_research_files/Dataset.csv