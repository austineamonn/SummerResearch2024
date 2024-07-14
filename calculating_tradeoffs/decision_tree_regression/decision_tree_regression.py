import logging
from ast import literal_eval
import pandas as pd
from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import joblib
import shap
import numpy as np
import json
import time

class DTRegressor:
    def __init__(self, privatization_type, RNN_model, target='career aspirations', data=None, output_paths=None):
        # Initiate inputs
        self.target = target # Set 'career aspirations' as the target if one is not chosen, other options include: 'future topics'
        self.target_name = target.replace(' ', '_')
        self.privatization_type = privatization_type
        self.RNN_model = RNN_model
        
        # Get Data Paths
        if data is None:
            data_path = f'../../data_preprocessing/reduced_dimensionality_data/{self.privatization_type}/{self.RNN_model}_combined.csv'

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

        # Set up Output Paths
        if output_paths is not None:
            self.ccp_alpha_models_path = output_paths[""]

    def split_data(self, full_model=False):
        # Set up X
        self.X = self.data[['gpa', 'student semester', 'learning style', 'major', 'previous courses', 'course types', 
                'course subjects', 'subjects of interest', 'extracurricular activities']]
        # Change the lists into just the elements within them if the element is a list otherwise just take the element
        for column in self.X.columns:
            self.X.loc[:, column] = self.X[column].apply(lambda x: x[0] if isinstance(x, list) else x)

        # Set up y post stratification
        self.y = self.data[[self.target]]

        # Change the doubles into integers
        self.y = self.y.astype(int)

        # Test - Train Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size = 0.8, random_state = 1234)
    
    def get_best_model(self, make_graphs=True, return_model=True, return_ccp_alpha=True, save_model=True):
        # Split the data
        self.split_data()

        # Generate the models with different ccp alphas
        regressor = DecisionTreeRegressor(random_state=0)
        path = regressor.cost_complexity_pruning_path(self.X_train, self.y_train)
        self.ccp_alphas, self.impurities = path.ccp_alphas, path.impurities

        # Ensure all ccp_alphas are non-negative
        self.ccp_alphas = [max(alpha, 0.0) for alpha in self.ccp_alphas]

        # Fit the models
        regressors = []
        for ccp_alpha in self.ccp_alphas:
            regressor = DecisionTreeRegressor(random_state=0, ccp_alpha=ccp_alpha)
            regressor.fit(self.X_train, self.y_train)
            regressors.append(regressor)
        self.node_counts = [regressor.tree_.node_count for regressor in regressors]
        self.depth = [regressor.tree_.max_depth for regressor in regressors]

        # Train the models
        self.train_scores = [regressor.score(self.X_train, self.y_train) for regressors in regressors]
        self.test_scores = [regressor.score(self.X_test, self.y_test) for regressors in regressors]

        self.models_data = {
            'models': regressors,
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
                  feature_names=self.X.columns,
                  filled=True, 
                  rounded=True)
        if show_fig:
            plt.show()
        if save_fig:
            plt.savefig(f'outputs/{self.privatization_type}/{self.RNN_model}/{self.target_name}/graphs/decision_tree_classifier.png')
        plt.close()

    def run_model(self, model=None, ccp_alpha=None, print_report=False, save_files=True, plot_files=True, get_shap=True):
        # Split the data
        self.split_data(full_model=True)

        if model is not None:
            self.model = model
            self.y_pred = self.model.predict(self.X_test)
        else:
            if ccp_alpha is not None:
                self.model = DecisionTreeRegressor(random_state=0, ccp_alpha=ccp_alpha)
            else:
                self.model = DecisionTreeRegressor(random_state=0)
            
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
                # Initiate regressor
                classifier = DTRegressor(privatization_type, RNN_model, target)
                ccp_alpha = classifier.get_best_model(return_model=False)
                # Don't forget you left get_shap at false!!!
                classifier.run_model(ccp_alpha=ccp_alpha, get_shap=False)

    # Save the profiling stats to a file
    profile_stats_file = "profile_stats.txt"
    with open(profile_stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats()