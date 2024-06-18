import logging
import numpy as np
import pandas as pd
import sys
import os
import ast
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

# Add the SummerResearch2024 directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class PreProcessing:
    def __init__(self, config, data) -> None:
        # Set up logging
        logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])

        # Privatized - Utility split
        self.Xp = config["privacy"]["Xp_list"]
        self.X = config["privacy"]["X_list"]
        self.Xu = config["privacy"]["Xu_list"]
        self.numerical_cols = config["privacy"]["numerical_columns"]

        # Data
        self.data = data

    def stringlist_to_binarylist(self, stringlist, col):
        binarylist = []

        for rowlist in col:
            # Ensure rowlist is a list of strings
            if isinstance(rowlist, str):
                if isinstance(rowlist, str):
                    try:
                        rowlist = ast.literal_eval(rowlist)
                        if not isinstance(rowlist, list):
                            raise ValueError
                    except (ValueError, SyntaxError):
                        # Handle the case where the string cannot be converted to a list
                        rowlist = rowlist.strip("[]").replace("'", "").split(", ")

            logging.debug(f"Processing row: {rowlist}")
            # Create a list of zeroes the length of the stringlist
            zeros_list = np.zeros(len(stringlist), dtype=int).tolist()

            # Iterate through each item in the list
            for item in rowlist:
                item = item.strip()
                logging.debug(f"Processing item: {item}")
                if item in stringlist:
                    item_spot = stringlist.index(item)
                    zeros_list[item_spot] = 1
                else:
                    logging.debug(f"Error: '{item}' is not in list")

            # Add the list to the binary list
            binarylist.append(zeros_list)

        return binarylist

    def string_list_to_numberedlist(self, stringlist, col):
        numberedlist = []
        unknown_item_list = []

        for rowlist in col:
            # Ensure rowlist is a list of strings
            if isinstance(rowlist, str):
                try:
                    rowlist = ast.literal_eval(rowlist)
                    if not isinstance(rowlist, list):
                        raise ValueError
                except (ValueError, SyntaxError):
                    # Handle the case where the string cannot be converted to a list
                    rowlist = rowlist.strip("[]").replace("'", "").split(", ")

            logging.debug(f"Processing row: {rowlist}")
            # Initialize a new list
            newlist = []
            for item in rowlist:
                item = item.strip()
                logging.debug(f"Processing item: {item}")
                if item in stringlist:
                    # For each item in the list add its index to the new list
                    newlist.append(stringlist.index(item))
                else:
                    unknown_item_list.append(item)
                    logging.debug(f"Error: '{item}' is not in list")
            
            # Add the list to the numbered list
            numberedlist.append(newlist)

            unknown_item_list = list(set(unknown_item_list))

        return numberedlist, unknown_item_list
    
    def PCA(self, df, n_components=100):
        """
        Input: dataframe (columns to run PCA analysis on), number of components
        Output: PCA analyzed dataframe, PCA
        """
        # Save the locations of NaNs
        nan_mask = df.isna()

        # Impute NaNs with the mean for PCA
        imputer = SimpleImputer(strategy='mean')
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        # Standardizing the features
        features = df_imputed.columns
        df = StandardScaler().fit_transform(df_imputed)

        # Convert standardized data back to a DataFrame
        df_standardized = pd.DataFrame(data=df, columns=features)

        # Initialize PCA
        pca = PCA(n_components=n_components)

        # Fit and transform the standardized data
        principal_components = pca.fit_transform(df_standardized)

        # Create a DataFrame with the principal components
        principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)])

        # Restore NaNs to the PCA results where they originally were
        for col in principal_df.columns:
            principal_df[col] = principal_df[col].where(~nan_mask.any(axis=1), np.nan)

        return principal_df, pca
    
    def analyze_PCA(self, pca):
        """
        Input: PCA
        Output: none
        """
        # Explained variance
        explained_variance = pca.explained_variance_ratio_
        logging.debug(f'Explained variance by each component: {explained_variance}')

        # Plotting the explained variance
        plt.figure(figsize=(8, 5))
        plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center', label='individual explained variance')
        plt.step(range(1, len(explained_variance) + 1), explained_variance.cumsum(), where='mid', label='cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.legend(loc='best')
        plt.tight_layout()
        
        # Save the plot as a file
        plt.savefig(self.config["running_model"]["PCA explained variance path"])
        logging.info("PCA explained variance graph saved to 'explained_variance_plot.png'")
    
    def preprocess_columns(self, df, col):
        sequences = df[col].apply(lambda x: str(x)).tolist()
        # Ensure all sequences are lists of integers
        sequences = [[int(item) for item in seq.strip('[]').split(',') if item] for seq in sequences]
        # Determine the maximum length of the sequences in this column
        max_len = max(len(seq) for seq in sequences)
        # Pad the sequences
        padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
        return padded_sequences
    
    def create_rnn_model(input_length, output_dim=5):
        model = Sequential()
        model.add(Embedding(input_dim=5000, output_dim=64, input_length=input_length))
        model.add(SimpleRNN(50, return_sequences=False))
        model.add(Dense(output_dim, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        return model

    def preprocess_dataset(self, df):
        """
        Input: dataset
        Output: preprocessed dataset
        """
        list_data = {}

        # Drop the Xp columns
        df = df.drop(columns=self.Xp)
        logging.debug(f"{self.Xp} were dropped")

        # Iterate through the columns
        for col in self.X + self.Xu:
            if col not in self.numerical_cols:
                # X
                if col == 'learning style':
                    stringlist = self.data.learning_style()['learning_style_list']
                    df[col] = self.stringlist_to_binarylist(stringlist, df[col])
                elif col == 'major':
                    stringlist = self.data.major()['majors_list']
                    df[col], unknown_list = self.string_list_to_numberedlist(stringlist, df[col])
                    if not unknown_list:
                        logging.debug("No unknown elements in majors.")
                    else:
                        logging.error('majors: %s', unknown_list)
                elif col == 'previous courses':
                    stringlist = self.data.course()['course_names_list']
                    df[col], unknown_list = self.string_list_to_numberedlist(stringlist, df[col])
                    if not unknown_list:
                        logging.debug("No unknown elements in previous courses.")
                    else:
                        logging.error('previous courses: %s', unknown_list)
                elif col == 'course types':
                    stringlist = self.data.course()['course_type_list']
                    df[col], unknown_list = self.string_list_to_numberedlist(stringlist, df[col])
                    if not unknown_list:
                        logging.debug("No unknown elements in course types.")
                    else:
                        logging.error('course types: %s', unknown_list)
                elif col == 'course subjects':
                    stringlist = self.data.course()['course_subject']
                    df[col], unknown_list = self.string_list_to_numberedlist(stringlist, df[col])
                    if not unknown_list:
                        logging.debug("No unknown elements in course subjects.")
                    else:
                        logging.error('course subjects: %s', unknown_list)
                elif col == 'subjects of interest':
                    stringlist = self.data.subjects()['subjects_list']
                    df[col], unknown_list = self.string_list_to_numberedlist(stringlist, df[col])
                    if not unknown_list:
                        logging.debug("No unknown elements in subjects of interest.")
                    else:
                        logging.error('subjects of interest: %s', unknown_list)
                elif col == 'extracurricular activities':
                    stringlist = self.data.extracurricular_activities()['complete_activity_list']
                    df[col], unknown_list = self.string_list_to_numberedlist(stringlist, df[col])
                    if not unknown_list:
                        logging.debug("No unknown elements in extracurricular activities.")
                    else:
                        logging.error('extracurricular activities: %s', unknown_list)
                # Xu
                elif col == 'career aspirations':
                    stringlist = self.data.careers()['careers_list']
                    df[col], unknown_list = self.string_list_to_numberedlist(stringlist, df[col])
                    if not unknown_list:
                        logging.debug("No unknown elements in career aspirations.")
                    else:
                        logging.error('career aspirations: %s', unknown_list)
                elif col == 'future topics':
                    stringlist = self.data.future_topics()['future_topics']
                    df[col], unknown_list = self.string_list_to_numberedlist(stringlist, df[col])
                    if not unknown_list:
                        logging.debug("No unknown elements in future topics.")
                    else:
                        logging.error('future topics: %s', unknown_list)
                else:
                    logging.error(f"{col} is not a known column name")
                
                preprocessed_col = self.preprocess_columns(df, col)

                # Create and train the model
                input_length = preprocessed_col.shape[1]
                rnn_model = self.create_rnn_model(input_length)
                
                # Normally, you would train the model with appropriate labels, but for this example, we'll use dummy data
                dummy_labels = np.random.rand(len(preprocessed_col), 5)  # Dummy target vectors of length 5
                rnn_model.fit(preprocessed_col, dummy_labels, epochs=10, batch_size=32)
                
                # Transform the sequences
                transformed_sequences = rnn_model.predict(preprocessed_col)
                df[col] = transformed_sequences

        return df

# Main execution
if __name__ == "__main__":
    # Import necessary dependencies
    from datafiles_for_data_construction.data import Data
    from config import load_config

    # Load configuration and data
    config = load_config()
    data = Data()

    # Import synthetic dataset and preprocess it
    df = pd.read_csv(config["running_model"]["data path"])
    preprocessor = PreProcessing(config, data)
    df = preprocessor.preprocess_dataset(df)
    df.to_csv(config["running_model"]["preprocessed data path"], index=False)
