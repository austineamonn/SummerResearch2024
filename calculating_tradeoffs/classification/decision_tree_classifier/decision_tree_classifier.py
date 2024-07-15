import logging
from ast import literal_eval
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import joblib
import shap
import numpy as np
import json
import time
import os
import pickle

class DTClassifier:
    def __init__(self, privatization_type, RNN_model, target='ethnoracial group', data=None, output_paths=None):
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
        if output_paths is not None:
            self.ccp_alpha_models_path = output_paths[""]

        # Make the necessary folers
        self.make_folders()

    def make_folders(self, directory=None):
        if directory is None:
            directory = os.path.dirname(f'outputs/{self.privatization_type}/{self.RNN_model}/{self.target_name}/graphs/feature_scatter_plots/example.py')
            
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

        # Set up y post stratification
        self.y = self.stratified_data[[self.target]]

        # Change the doubles into integers (1.0 -> 1)
        self.y = self.y.astype(int)

        # Test - Train Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size = 0.8, random_state = 1234)
    
    def get_best_model(self, make_graphs=True, return_model=True, return_ccp_alpha=True, save_model=True):
        # Split the data
        self.split_data()

        # Generate the models with different ccp alphas
        clf = DecisionTreeClassifier(random_state=0)
        path = clf.cost_complexity_pruning_path(self.X_train, self.y_train)
        self.ccp_alphas, self.impurities = path.ccp_alphas, path.impurities

        # Ensure all ccp_alphas are non-negative
        self.ccp_alphas = [max(alpha, 0.0) for alpha in self.ccp_alphas]

        # Fit the models
        clfs = []
        for ccp_alpha in self.ccp_alphas:
            clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
            clf.fit(self.X_train, self.y_train)
            clfs.append(clf)
        self.node_counts = [clf.tree_.node_count for clf in clfs]
        self.depth = [clf.tree_.max_depth for clf in clfs]

        # Train the models
        self.train_scores = [clf.score(self.X_train, self.y_train) for clf in clfs]
        self.test_scores = [clf.score(self.X_test, self.y_test) for clf in clfs]

        self.models_data = {
            'models': clfs,
            'ccp alpha': self.ccp_alphas,
            'train scores': self.train_scores,
            'test scores': self.test_scores,
            'impurities': self.impurities,
            'node count': self.node_counts,
            'depth': self.depth
        }
        
        models = pd.DataFrame(self.models_data)

        if save_model:
            models.to_csv(f'outputs/{self.privatization_type}/{self.RNN_model}/{self.target_name}/decision_tree_classifier_models.csv', index=False)

        models_sorted = models.sort_values(by='test scores', ascending=False)

        # Pull out the model that has the highest test score and make it the model for training
        self.model, self.ccp_alpha = models_sorted.iloc[0, 0:2]

        if make_graphs:
            self.graph_impurities()
            self.graph_nodes_and_depth()
            self.graph_accuracy()

        # Return the best model and the ccp_alpha
        if return_model and return_ccp_alpha:
            return self.model, self.ccp_alpha
        elif return_model:
            return self.model
        elif return_ccp_alpha:
            return_ccp_alpha
    
    def graph_impurities(self):
        # Impurity vs Effective Alpha
        fig, ax = plt.subplots()
        ax.plot(self.ccp_alphas[:-1], self.impurities[:-1], marker="o", drawstyle="steps-post")
        ax.set_xlabel("effective alpha")
        ax.set_ylabel("total impurity of leaves")
        ax.set_title("Total Impurity vs effective alpha for training set")
        plt.savefig(f'outputs/{self.privatization_type}/{self.RNN_model}/{self.target_name}/graphs/effective_alpha_vs_total_impurity.png')
        plt.close()

    def graph_nodes_and_depth(self):
        # Model Nodes and Depth vs Alpha
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(self.ccp_alphas, self.node_counts, marker="o", drawstyle="steps-post")
        ax[0].set_xlabel("alpha")
        ax[0].set_ylabel("number of nodes")
        ax[0].set_title("Number of nodes vs alpha")
        ax[1].plot(self.ccp_alphas, self.depth, marker="o", drawstyle="steps-post")
        ax[1].set_xlabel("alpha")
        ax[1].set_ylabel("depth of tree")
        ax[1].set_title("Depth vs alpha")
        fig.tight_layout()
        plt.savefig(f'outputs/{self.privatization_type}/{self.RNN_model}/{self.target_name}/graphs/effective_alpha_vs_graph_nodes_and_depth.png')
        plt.close()

    def graph_accuracy(self):
        # Accuracy vs Alpha
        fig, ax = plt.subplots()
        ax.set_xlabel("alpha")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy vs alpha for training and testing sets")
        ax.plot(self.ccp_alphas, self.train_scores, marker="o", label="train", drawstyle="steps-post")
        ax.plot(self.ccp_alphas, self.test_scores, marker="o", label="test", drawstyle="steps-post")
        ax.legend()
        plt.savefig(f'outputs/{self.privatization_type}/{self.RNN_model}/{self.target_name}/graphs/effective_alpha_vs_accuracy.png')
        plt.close()

    def plotter(self, model=None, save_fig=False, show_fig=False):
        # Plot the tree using matplotlib
        plt.figure(figsize=(20,10))
        if model is None:
            model = self.model
        plot_tree(model, 
                  feature_names=self.X_columns, 
                  class_names=self.classnames,
                  filled=True, 
                  rounded=True)
        if show_fig:
            plt.show()
        if save_fig:
            plt.savefig(f'outputs/{self.privatization_type}/{self.RNN_model}/{self.target_name}/graphs/decision_tree_classifier.png', bbox_inches="tight")
        plt.close()

    def run_model(self, model=None, ccp_alpha=None, print_report=False, save_files=True, plot_files=True, get_shap=True):
        # Split the data
        self.split_data(full_model=True)

        if model is not None:
            self.model = model
            self.y_pred = self.model.predict(self.X_test)
        else:
            if ccp_alpha is not None:
                self.model = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
            else:
                self.model = DecisionTreeClassifier(random_state=0)
            
            # Run and time model
            start_time = time.time()
            self.model.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)
            end_time = time.time()
            runtime = end_time - start_time

        # Get Metrics
        report = classification_report(self.y_test, self.y_pred, zero_division=0, output_dict=True)
        if model is None:
            report['time'] = runtime # Add time to the report dictionary
        if print_report:
            print(report)

        # Saving to a JSON file
        if save_files:
            with open(f'outputs/{self.privatization_type}/{self.RNN_model}/{self.target_name}/classification_report.json', 'w') as json_file:
                json.dump(report, json_file, indent=4)

            # Create a new csv file with the predictions
            y_pred_df = pd.concat([self.X_test, self.y_test], axis=1)
            y_pred_df[f'Predicted Class: {self.target}'] = self.y_pred
            y_pred_df.to_csv(f'outputs/{self.privatization_type}/{self.RNN_model}/{self.target_name}/predictions.csv', index=False)

            # Save the model
            self.save_model(f'outputs/{self.privatization_type}/{self.RNN_model}/{self.target_name}/decision_tree_classifier_model.pkl')

        if plot_files:
            # Plot the model
            self.plotter(save_fig=True)

        if get_shap:
            # Calculate and plot SHAP values
            self.calculate_shap_values()
            self.plot_shap_values()

    def calculate_shap_values(self, sample_size=1000, return_values=False):
        # Select a random sample from self.X
        if sample_size < len(self.X):
            self.X_sample = self.X.sample(n=sample_size, random_state=42)
        else:
            self.X_sample = self.X
        
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer(self.X_sample)  # Obtain SHAP values as an Explanation object
        self.shap_values = shap.Explanation(shap_values.values, base_values=shap_values.base_values, data=shap_values.data, feature_names=self.X_sample.columns)

        shap_values_path = f'outputs/{self.privatization_type}/{self.RNN_model}/{self.target_name}/shap_values.npy'
        np.save(shap_values_path, self.shap_values)

        if return_values:
            return self.shap_values

    def plot_shap_values(self, shap_values=None):
        if shap_values is not None:
            self.shap_values = shap_values

        # Generate the SHAP bar plot and capture the axes
        ax = shap.plots.bar(self.shap_values, show=False)
        
        # Get the figure from the axes
        fig = ax.figure

        # Set the figure size to be wider
        fig = plt.gcf()
        fig.set_size_inches(20, 6)  # Adjust the width and height as needed

        # Save the figure
        fig.savefig(f'outputs/{self.privatization_type}/{self.RNN_model}/{self.target_name}/graphs/shap_bar_plot.png', bbox_inches="tight")
        plt.close(fig)

        # Scatter plot for each specific feature
        for i, column in enumerate(['learning style']):#self.X_columns):
            # Create a figure and axis
            fig, ax = plt.subplots()
            if column == 'learning style':
                shap.plots.scatter(self.shap_values[:, 2], show=False, ax=ax)
                ax.set_xlim(0.4885, 0.4915)
            else:
                shap.plots.scatter(self.shap_values[:, i], show=False, ax=ax)

            # Adjust the size
            fig = plt.gcf()
            fig.set_size_inches(15, 6)

            # Save the figure
            plt.savefig(f'outputs/{self.privatization_type}/{self.RNN_model}/{self.target_name}/graphs/feature_scatter_plots/{column}.png', bbox_inches="tight")
            plt.close()

        # Heatmap plot
        ax = shap.plots.heatmap(self.shap_values, show=False)

        # Get the figure from the axes
        fig = ax.figure

        # Set the figure size to be wider
        fig = plt.gcf()
        fig.set_size_inches(15, 6)  # Adjust the width and height as needed

        # Save the figure
        fig.savefig(f'outputs/{self.privatization_type}/{self.RNN_model}/{self.target_name}/graphs/shap_heatmap.png', bbox_inches="tight")
        plt.close(fig)

        # Bee swarm plot
        ax = shap.plots.beeswarm(self.shap_values, show=False)

        # Get the figure from the axes
        fig = ax.figure

        # Set the figure size to be wider
        fig = plt.gcf()
        fig.set_size_inches(22, 6)  # Adjust the width and height as needed

        # Save the figure
        fig.savefig(f'outputs/{self.privatization_type}/{self.RNN_model}/{self.target_name}/graphs/shap_bee_swarm_plot.png', bbox_inches="tight")
        plt.close(fig)

        # Violin plot
        fig = plt.figure()
        shap.plots.violin(self.shap_values, show=False)

        # Set the figure size to be wider
        fig = plt.gcf()
        fig.set_size_inches(22, 6)  # Adjust the width and height as needed

        # Save the figure
        plt.savefig(f'outputs/{self.privatization_type}/{self.RNN_model}/{self.target_name}/graphs/shap_violin_plot.png', bbox_inches="tight")
        plt.close()

    def load_shap_values(self, file_path, return_values=False):
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
        self.shap_values = shap.Explanation(values=values,base_values=base_values, data=data)

        if return_values:
            return self.shap_values

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
            for target in targets:
                logging.info(f"Starting {target}")
                # Initiate classifier
                classifier = DTClassifier(privatization_type, RNN_model, target)
                ccp_alpha = classifier.get_best_model(return_model=False)
                # Don't forget you left get_shap at false!!!
                classifier.run_model(ccp_alpha=ccp_alpha, get_shap=False)

    # Save the profiling stats to a file
    profile_stats_file = "profile_stats.txt"
    with open(profile_stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats()