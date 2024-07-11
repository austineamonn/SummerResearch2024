import logging
import sys
import os
from ast import literal_eval
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

# Add the SummerResearch2024 directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class DTClassifier:
    def __init__(self, df):
        print(df.columns)
        # Set up data
        self.X = df['learning style', 'major', 'previous courses', 'course types', 
                'course subjects', 'subjects of interest', 'extracurricular activities']
        self.y = df['ethnoracial group', 'gender', 
                'international status']
        print(self.X.head())
        print(self.y.head())

    def classifier(self):
        clf = DecisionTreeClassifier()
        self.model = clf.fit(self.X, self.Y)

    def plotter(self):
        plot_tree(self.model)

# Main execution
if __name__ == "__main__":
    # List of RNN models to run
    RNN_model_list = ['GRU1']#, 'LSTM1', 'Simple1']

    # List of Privatization Types to run
    privatization_types = ['NoPrivatization']#, 'Basic_DP', 'Basic_DP_LLC', 'Uniform', 'Uniform_LLC', 'Shuffling', 'Complete_Shuffling']

    for privatization_type in privatization_types:
        logging.info(f"Starting {privatization_type}")
        for RNN_model in RNN_model_list:
            logging.info(f"Starting {RNN_model}")

            # Get Data Paths
            data_path = f'../../data_preprocessing/reduced_dimensionality_data/{privatization_type}/{RNN_model}_combined.csv'

            data = pd.read_csv(data_path, nrows=10000, converters={
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

            # Initiate Classifier
            classifier = DTClassifier(data)