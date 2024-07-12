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

# Add the SummerResearch2024 directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class DTClassifier:
    def __init__(self, privatization_type, RNN_model, target='ethnoracial group'):
        # Initiate inputs
        self.target = target # Set 'ethnoracial group' as the target if one is not chosen, other options include: 'gender' and 'international status'
        self.target_name = target.replace(' ', '_')
        self.privatization_type = privatization_type
        self.RNN_model = RNN_model

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

    def plotter(self):
        # Plot the tree using matplotlib
        plt.figure(figsize=(20,10))
        plot_tree(self.model, 
                  feature_names=self.X.columns, 
                  class_names=self.y.columns,
                  filled=True, 
                  rounded=True)
        plt.savefig(f'outputs/{self.privatization_type}/{self.RNN_model}/{self.target_name}/graphs/decision_tree_classifier.png')
        plt.close()

    def run_model(self, model=None, ccp_alpha=None):
        if model is not None:
            clf = model
        elif ccp_alpha is not None:
            clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        else:
            clf = DecisionTreeClassifier(random_state=0)
        clf.fit(self.X_train, self.y_train)

        self.y_pred = clf.predict(self.X_test)

        # Get Metrics
        report = classification_report(self.y_test, self.y_pred, zero_division=0)
        report_df = pd.DataFrame(report)
        report_df.to_csv(f'outputs/{self.privatization_type}/{self.RNN_model}/{self.target_name}/classification_report.csv', index=False)

        # Plot the model
        self.plotter()
        # Save the model
        self.save_model(f'outputs/{self.privatization_type}/{self.RNN_model}/{self.target_name}/decision_tree_classifier_model.pkl')

    def save_model(self, file_path):
        joblib.dump(self.model, file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path, return_file=False):
        self.model = joblib.load(file_path)
        print(f"Model loaded from {file_path}")
        if return_file:
            return self.model

# Main execution
if __name__ == "__main__":
    # List of RNN models to run
    RNN_model_list = ['GRU1']#, 'LSTM1', 'Simple1']

    # List of Privatization Types to run
    privatization_types = ['NoPrivatization']#, 'Basic_DP', 'Basic_DP_LLC', 'Uniform', 'Uniform_LLC', 'Shuffling', 'Complete_Shuffling']
    targets = ['ethnoracial group']#, 'gender', 'international status']

    for privatization_type in privatization_types:
        logging.info(f"Starting {privatization_type}")
        for RNN_model in RNN_model_list:
            logging.info(f"Starting {RNN_model}")
            for target in targets:
                logging.info(f"Starting {target}")
                # Initiate first classifier
                classifier = DTClassifier(privatization_type, RNN_model, target)
                classifier.read_data(1000)
                classifier.split_data()
                ccp_alpha = classifier.get_best_model(return_model=False)
                classifier.read_data(100000)
                classifier.split_data()
                classifier.run_model(ccp_alpha=ccp_alpha)