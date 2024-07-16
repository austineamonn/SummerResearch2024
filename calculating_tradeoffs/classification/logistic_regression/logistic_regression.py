import logging
from ast import literal_eval
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib
import shap
import numpy as np
import json
import time
import os
import pickle
import seaborn as sn

# Set the pandas option to avoid silent downcasting
pd.set_option('future.no_silent_downcasting', True)

class LogisticRegressor:
    def __init__(self, privatization_type, RNN_model, target='ethnoracial group', data=None, output_path=None):
        # Initiate inputs
        self.target = target # Set 'ethnoracial group' as the target if one is not chosen, other options include: 'gender', 'international status', and 'socioeconomic status'
        self.target_name = target.replace(' ', '_')
        self.privatization_type = privatization_type
        self.RNN_model = RNN_model

        if target == 'ethnoracial group':
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
            raise ValueError(f"Incorrect target column name {target}")
        self.labels = list(range(len(self.classnames)))
        
        # Get Data Paths
        if data is None:
            data_path = f'../../../data_preprocessing/reduced_dimensionality_data/{self.privatization_type}/{self.RNN_model}_combined.csv'

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
        self.X_columns = ['gpa', 'student semester', 'learning style', 'major', 'previous courses', 'course types', 'course subjects', 'subjects of interest', 'extracurricular activities']


        # Set up Output Paths
        if output_path is not None:
            self.output_path = output_path
        else:
            self.output_path = f'outputs/{self.privatization_type}/{self.RNN_model}/{self.target_name}'

        # Make the necessary folers
        self.make_folders()

    def make_folders(self, directory=None):
        if directory is None:
            if self.target == 'international status':
                directory = os.path.dirname(f'{self.output_path}/graphs/feature_scatter_plots/example.py')

                # Create the directory if it doesn't exist
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
            else:
                for name in self.classnames:
                    name = name.replace(' ', '_')
                    directory = os.path.dirname(f'{self.output_path}/graphs/{name}/feature_scatter_plots/example.py')

                    # Create the directory if it doesn't exist
                    if not os.path.exists(directory):
                        os.makedirs(directory, exist_ok=True)
        else:
            # Create the directory if it doesn't exist
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

    def split_data(self, full_model=False):
        # Define y
        self.y = self.data[[self.target]]

        # Stratify the data by taking only the number of elements the least common values has
        least_common_count = self.y.value_counts().min()
        if full_model:
            sample_number = least_common_count
        else: # Cut it off at 1,000 if running the ccp alpha calculations
            sample_number = min(1000, least_common_count)
        self.stratified_data = self.data.groupby(self.target, group_keys=False).apply(lambda x: x.sample(sample_number))

        # Set up X
        self.X = self.stratified_data[self.X_columns]
        # Change the lists into just the elements within them if the element is a list otherwise just take the element
        for column in self.X_columns:
            self.X.loc[:, column] = self.X[column].apply(lambda x: x[0] if isinstance(x, list) else x)

        # Fill NaN values with the mean of each column
        self.X = self.X.fillna(self.X.mean())

        # Infer objects to avoid future warning
        self.X = self.X.infer_objects(copy=False)

        # Set up y post stratification
        self.y = self.stratified_data[[self.target]]

        # Change the doubles into integers (1.0 -> 1)
        self.y = self.y.astype(int)

        # Test - Train Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size = 0.8, random_state = 1234)

        # Reshape ys to be 1-dimensional
        self.y_train = self.y_train.values.ravel()

    def plotter(self, matrix=None, matrix_path=None, save_fig=False, show_fig=False):
        if matrix_path is not None:
            # Loading the matrix
            matrix = np.load(matrix_path)
        # Plot the tree using matplotlib
        plt.figure(figsize = (10,7))
        sn.heatmap(matrix, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        if show_fig:
            plt.show()
        if save_fig:
            plt.savefig(f'{self.output_path}/graphs/confusion_matrix.png', bbox_inches="tight")
            plt.close()
        else:
            plt.close()

    def run_model(self, model=None, print_report=False, save_files=True, plot_files=True, get_shap=True):
        # Split the data
        self.split_data(full_model=True)

        if model is not None:
            self.model = model
            self.y_pred = self.model.predict(self.X_test)
        else:
            self.model = LogisticRegression(solver='liblinear',
                         random_state = 0)
            
            # Run and time model
            start_time = time.time()
            self.model.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)
            end_time = time.time()
            runtime = end_time - start_time

        # Get Metrics
        cm = confusion_matrix(self.y_test, self.y_pred)

        if print_report:
            report = classification_report(self.y_test, self.y_pred, zero_division=0, labels=self.labels, target_names=self.classnames)
            print(report)
        else:
            report = classification_report(self.y_test, self.y_pred, zero_division=0, output_dict=True, labels=self.labels,target_names=self.classnames)
            if model is None:
                report['time'] = runtime # Add time to the report dictionary


        # Saving to a JSON file
        if save_files:
            with open(f'{self.output_path}/classification_report.json', 'w') as json_file:
                json.dump(report, json_file, indent=4)

            # Save confusion matrix
            np.save(f'{self.output_path}/confusion_matrix.npy', cm)

            # Create a new csv file with the predictions
            y_pred_df = pd.concat([self.X_test, self.y_test], axis=1)
            y_pred_df[f'Predicted Class: {self.target}'] = self.y_pred
            y_pred_df.to_csv(f'{self.output_path}/predictions.csv', index=False)

            # Save the model
            self.save_model(f'{self.output_path}/logistic_regressor_model.pkl')

        if plot_files:
            # Plot the confusion matrix
            self.plotter(matrix=cm, save_fig=True)

        if get_shap:
            # Calculate and plot SHAP values
            self.calculate_shap_values()
            self.plot_shap_values()
        
        # Ensure all figures were closed
        plt.close('all')

    def calculate_shap_values(self, sample_size=1000, return_values=False):
        # Select a random sample from self.X
        if sample_size < len(self.X):
            self.X_sample = self.X.sample(n=sample_size, random_state=42)
        else:
            self.X_sample = self.X
        
        explainer = shap.LinearExplainer(self.model, self.X_train)
        shap_values = explainer(self.X_sample)  # Obtain SHAP values as an Explanation object

        shap_values_path = f'{self.output_path}/shap_values.npy'
        np.save(shap_values_path, shap_values)

        if return_values:
            return shap_values
        
    def plot_shap_values(self, shap_values=None, shap_explainer_list=None):
        if shap_explainer_list is not None:
            self.shap_explainer_list = shap_explainer_list
        if shap_values is not None:
            self.shap_values = shap_values
        if self.target == 'international status':
            # Generate the SHAP bar plot and capture the axes
            ax = shap.plots.bar(self.shap_values, show=False)
            
            # Get the figure from the axes
            fig = ax.figure

            # Set the figure size to be wider
            fig = plt.gcf()
            fig.set_size_inches(20, 6)  # Adjust the width and height as needed

            # Save the figure
            fig.savefig(f'{self.output_path}/graphs/shap_bar_plot.png', bbox_inches="tight")
            plt.close(fig)

            # Scatter plot for each specific feature
            for i, column in enumerate(self.X_columns):
                # Create a figure and axis
                fig, ax = plt.subplots()
                # TODO: Fix the width of learning style graphs to match better
                if column == 'learning style':
                    shap.plots.scatter(self.shap_values[:, 2], show=False, ax=ax)
                    ax.set_xlim(0.488, 0.51)
                else:
                    shap.plots.scatter(self.shap_values[:, i], show=False, ax=ax)

                # Adjust the size
                fig = plt.gcf()
                fig.set_size_inches(15, 6)

                # Save the figure
                plt.savefig(f'{self.output_path}/graphs/feature_scatter_plots/{column}.png', bbox_inches="tight")
                plt.close(fig)

            # Heatmap plot
            ax = shap.plots.heatmap(self.shap_values, show=False)

            # Get the figure from the axes
            fig = ax.figure

            # Set the figure size to be wider
            fig = plt.gcf()
            fig.set_size_inches(15, 6)  # Adjust the width and height as needed

            # Save the figure
            fig.savefig(f'{self.output_path}/graphs/shap_heatmap.png', bbox_inches="tight")
            plt.close(fig)

            # Bee swarm plot
            ax = shap.plots.beeswarm(self.shap_values, show=False)

            # Get the figure from the axes
            fig = ax.figure

            # Set the figure size to be wider
            fig = plt.gcf()
            fig.set_size_inches(22, 6)  # Adjust the width and height as needed

            # Save the figure
            fig.savefig(f'{self.output_path}/graphs/shap_bee_swarm_plot.png', bbox_inches="tight")
            plt.close(fig)

            # Violin plot
            fig = plt.figure()
            shap.plots.violin(self.shap_values, show=False)

            # Set the figure size to be wider
            fig = plt.gcf()
            fig.set_size_inches(22, 6)  # Adjust the width and height as needed

            # Save the figure
            plt.savefig(f'{self.output_path}/graphs/shap_violin_plot.png', bbox_inches="tight")
            plt.close(fig)
        else:
            for i, name in enumerate(self.classnames):
                self.plot_shap_values_one_target(self.shap_explainer_list[i], name)

    def plot_shap_values_one_target(self, shap_values, classname):

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
        fig.savefig(f'{self.output_path}/graphs/{classname}/shap_bar_plot.png', bbox_inches="tight")
        plt.close(fig)

        # Scatter plot for each specific feature
        for i, column in enumerate(self.X_columns):
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
            plt.savefig(f'{self.output_path}/graphs/{classname}/feature_scatter_plots/{column}.png', bbox_inches="tight")
            plt.close(fig)

        # Heatmap plot
        ax = shap.plots.heatmap(shap_values, show=False)

        # Get the figure from the axes
        fig = ax.figure

        # Set the figure size to be wider
        fig = plt.gcf()
        fig.set_size_inches(15, 6)  # Adjust the width and height as needed

        # Save the figure
        fig.savefig(f'{self.output_path}/graphs/{classname}/shap_heatmap.png', bbox_inches="tight")
        plt.close(fig)

        # Bee swarm plot
        ax = shap.plots.beeswarm(shap_values, show=False)

        # Get the figure from the axes
        fig = ax.figure

        # Set the figure size to be wider
        fig = plt.gcf()
        fig.set_size_inches(22, 6)  # Adjust the width and height as needed

        # Save the figure
        fig.savefig(f'{self.output_path}/graphs/{classname}/shap_bee_swarm_plot.png', bbox_inches="tight")
        plt.close(fig)

        # Violin plot
        fig = plt.figure()
        shap.plots.violin(shap_values, show=False)

        # Set the figure size to be wider
        fig = plt.gcf()
        fig.set_size_inches(22, 6)  # Adjust the width and height as needed

        # Save the figure
        plt.savefig(f'{self.output_path}/graphs/{classname}/shap_violin_plot.png', bbox_inches="tight")
        plt.close(fig)

        # Ensure all figures were closed
        plt.close('all')

    def load_shap_values(self, file_path, return_values=False):
        if self.target == 'international status':
            # Load the .npy file
            self.shap_values = np.load(file_path, allow_pickle=True)
            
            # Extract the SHAP values, base values, and data
            values = []
            base_values = []
            data = []

            for element in self.shap_values:
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
            self.shap_values = shap.Explanation(values=values,base_values=base_values, data=data, feature_names=self.X_columns)

            if return_values:
                return self.shap_values
        else:
            # Load the .npy file
            shap_values = np.load(file_path, allow_pickle=True)

            self.shap_explainer_list = []

            for i in range(len(self.classnames)):
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
                shap_value_explainer = shap.Explanation(values=values,base_values=base_values, data=data, feature_names=self.X_columns)

                # Add object to list of explainers, one per target
                self.shap_explainer_list.append(shap_value_explainer)

            if return_values:
                return self.shap_explainer_list

    def save_model(self, file_path):
        joblib.dump(self.model, file_path)

    def load_model(self, file_path, return_file=False):
        try:
            self.model = joblib.load(file_path)
        except Exception as e:
            logging.error(f"Joblib failed to load the model: {e}")
            logging.info("Attempting to load with pickle...")
            try:
                with open(file_path, 'rb') as f:
                    self.model = pickle.load(f)
            except Exception as pickle_e:
                logging.error(f"Pickle also failed to load the model: {pickle_e}")
                raise pickle_e
            
        if return_file:
            return self.model

# Main execution
if __name__ == "__main__":

    # Import necessary dependencies
    import cProfile
    import pstats

    # List of RNN models to run
    RNN_model_list = ['GRU1', 'LSTM1', 'Simple1']

    # List of Privatization Types to run
    privatization_types = ['NoPrivatization', 'Basic_DP', 'Basic_DP_LLC', 'Uniform', 'Uniform_LLC', 'Shuffling', 'Complete_Shuffling']

    # List of targets for the model 
    targets = ['ethnoracial group', 'gender', 'international status', 'socioeconomic status']

    # Get the runtime values for the function
    profiler = cProfile.Profile()
    profiler.enable()

    for privatization_type in privatization_types:
        logging.info(f"Starting {privatization_type}")
        for RNN_model in RNN_model_list:
            logging.info(f"Starting {RNN_model}")
            
            # Skip if privatization_type is 'NoPrivatization' and RNN_model is 'GRU1'
            if privatization_type == 'NoPrivatization' and RNN_model == 'GRU1':
                logging.info(f"Skipping {privatization_type} and {RNN_model}")
                continue

            for target in targets:
                logging.info(f"Starting {target}")

                target_name = target.replace(' ', '_')

                # Initiate regressor
                regressor = LogisticRegressor(privatization_type, RNN_model, target)
                regressor.run_model(get_shap=False)
                #regressor.split_data(full_model=True)
                #regressor.load_model(f'outputs/{privatization_type}/{RNN_model}/{target_name}/logistic_regressor_model.pkl')
                regressor.calculate_shap_values()
                regressor.load_shap_values(f'outputs/{privatization_type}/{RNN_model}/{target_name}/shap_values.npy')
                regressor.plot_shap_values()

    # Save the profiling stats to a file
    profile_stats_file = "profile_stats.txt"
    with open(profile_stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats()