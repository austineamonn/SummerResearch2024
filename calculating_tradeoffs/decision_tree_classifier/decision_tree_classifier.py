import logging
import sys
import os
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

# Add the SummerResearch2024 directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class DTClassifier:
    def __init__(self, privatization_type, RNN_model, target='ethnoracial group'):
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

    def read_data(self, nrows=1000):
        # Get Data Paths
            data_path = f'../../data_preprocessing/reduced_dimensionality_data/{self.privatization_type}/{self.RNN_model}_combined.csv'

            self.data = pd.read_csv(data_path, nrows=nrows, converters={
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

    def split_data(self):
        # Set up X
        self.X = self.data[['learning style', 'major', 'previous courses', 'course types', 
                'course subjects', 'subjects of interest', 'extracurricular activities']]
        # Change the lists into just the elements within them
        for column in self.X.columns:
            self.X.loc[:,column] = self.X[column].apply(lambda x: x[0])

        # Set up y
        self.y = self.data[[self.target]]

        # Change the doubles into integers
        self.y = self.y.astype(int)

        # Test - Train Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size = 0.8, random_state = 1234)
    
    def get_best_model(self, make_graphs=True, return_model=True, return_ccp_alpha=True):
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
                  feature_names=self.X.columns, 
                  class_names=self.classnames,
                  filled=True, 
                  rounded=True)
        if show_fig:
            plt.show()
        if save_fig:
            plt.savefig(f'outputs/{self.privatization_type}/{self.RNN_model}/{self.target_name}/graphs/decision_tree_classifier.png')
        plt.close()

    def run_model(self, model=None, ccp_alpha=None, print_report=False, save_files=True, plot_files=True, get_shap=True):
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

            # Save y_preds to a csv file
            y_pred_df = pd.DataFrame(self.y_pred, columns=[f'Predicted Class: {self.target}'])
            y_pred_df.to_csv(f'outputs/{self.privatization_type}/{self.RNN_model}/{self.target_name}/predictions.csv', index=False)

            # Save the model
            self.save_model(f'outputs/{self.privatization_type}/{self.RNN_model}/{self.target_name}/decision_tree_classifier_model.pkl')

        if plot_files:
            # Plot the model
            self.plotter()

        if get_shap:
            # Calculate and plot SHAP values
            self.calculate_shap_values()
            self.plot_shap_values()

    def calculate_shap_values(self, sample_size=1000):
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

    def plot_shap_values(self):
        # Bar plot
        shap.plots.bar(self.shap_values)
        plt.savefig(f'outputs/{self.privatization_type}/{self.RNN_model}/{self.target_name}/graphs/shap_bar_plot.png')
        plt.close()

        # Scatter plot for specific feature
        for column in self.X.columns:
            shap.plots.scatter(self.shap_values[:, column])
            plt.savefig(f'outputs/{self.privatization_type}/{self.RNN_model}/{self.target_name}/graphs/feature_scatter_plots/{column}.png')
            plt.close()

        # Heatmap plot
        shap.plots.heatmap(self.shap_values)
        plt.savefig(f'outputs/{self.privatization_type}/{self.RNN_model}/{self.target_name}/graphs/shap_heatmap.png')
        plt.close()

        # Bee swarm plot
        shap.plots.beeswarm(self.shap_values)
        plt.savefig(f'outputs/{self.privatization_type}/{self.RNN_model}/{self.target_name}/graphs/shap_bee_swarm_plot.png')
        plt.close()

        # Violin plot
        shap.plots.violin(self.shap_values)
        plt.savefig(f'outputs/{self.privatization_type}/{self.RNN_model}/{self.target_name}/graphs/shap_violin_plot.png')
        plt.close()

    def save_model(self, file_path):
        joblib.dump(self.model, file_path)

    def load_model(self, file_path, return_file=False):
        self.model = joblib.load(file_path)
        if return_file:
            return self.model

# Main execution
if __name__ == "__main__":
    # List of RNN models to run
    RNN_model_list = ['GRU1', 'LSTM1', 'Simple1']

    # List of Privatization Types to run
    privatization_types = ['NoPrivatization', 'Basic_DP', 'Basic_DP_LLC', 'Uniform', 'Uniform_LLC', 'Shuffling', 'Complete_Shuffling']

    # List of targets for the model 
    targets = ['ethnoracial group', 'gender', 'international status', 'socioeconomic status']

    for privatization_type in privatization_types:
        logging.info(f"Starting {privatization_type}")
        for RNN_model in RNN_model_list:
            logging.info(f"Starting {RNN_model}")
            for target in targets:
                logging.info(f"Starting {target}")
                # Initiate first classifier
                classifier = DTClassifier(privatization_type, RNN_model, target)
                classifier.read_data(10000)
                classifier.split_data()
                ccp_alpha = classifier.get_best_model(return_model=False)
                classifier.read_data(100000)
                classifier.split_data()
                # Don't forget you left get_shap at false!!!
                classifier.run_model(ccp_alpha=ccp_alpha, get_shap=False)