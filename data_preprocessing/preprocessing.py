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
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Bidirectional, LSTM, GRU, Dropout # type: ignore
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
                # Only strip if item is a string
                if isinstance(item, str):
                    item = item.strip()
                logging.debug(f"Processing item: {item}")
                if item in stringlist:
                    # For each item in the list add its index to the new list
                    newlist.append(stringlist.index(item))
                else:
                    logging.debug(f"Error: '{item}' is not in list")
                    logging.debug(f"Current stringlist: {stringlist}")
            
            # Add the list to the numbered list
            numberedlist.append(newlist)

        return numberedlist

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
        max_len = max(5, max(len(seq) for seq in sequences))
        # Pad the sequences
        padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
        return padded_sequences
    
    def create_simple_rnn_model(self, output_dim=1, num_layers=2, dropout_rate=0.2):
        model = Sequential()
        model.add(Embedding(input_dim=5000, output_dim=64))

        for _ in range(num_layers - 1):
            model.add(Bidirectional(SimpleRNN(50, return_sequences=True)))
            model.add(Dropout(dropout_rate))

        model.add(SimpleRNN(50, return_sequences=False))
        model.add(Dense(output_dim, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def create_lstm_model(self, output_dim=1, num_layers=2, dropout_rate=0.2):
        model = Sequential()
        model.add(Embedding(input_dim=5000, output_dim=64))
        
        for _ in range(num_layers - 1):
            model.add(Bidirectional(LSTM(50, return_sequences=True)))
            model.add(Dropout(dropout_rate))
        
        model.add(Bidirectional(LSTM(50, return_sequences=False)))
        model.add(Dense(output_dim, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        return model

    def create_gru_model(self, output_dim=1, num_layers=2, dropout_rate=0.2):
        model = Sequential()
        model.add(Embedding(input_dim=5000, output_dim=64))

        for _ in range(num_layers - 1):
            model.add(Bidirectional(GRU(50, return_sequences=True)))
            model.add(Dropout(dropout_rate))

        model.add(Bidirectional(GRU(50, return_sequences=False)))
        model.add(Dense(output_dim, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        return model

    def preprocess_dataset(self, df):
        """
        Input: dataset
        Output: preprocessed dataset
        """

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
                    df[col] = self.string_list_to_numberedlist(stringlist, df[col])
                elif col == 'previous courses':
                    stringlist = self.data.course()['course_names_list']
                    df[col] = self.string_list_to_numberedlist(stringlist, df[col])
                elif col == 'course types':
                    stringlist = self.data.course()['course_type_list']
                    df[col] = self.string_list_to_numberedlist(stringlist, df[col])
                elif col == 'course subjects':
                    stringlist = self.data.course()['course_subject']
                    df[col] = self.string_list_to_numberedlist(stringlist, df[col])
                elif col == 'subjects of interest':
                    stringlist = self.data.subjects()['subjects_list']
                    df[col] = self.string_list_to_numberedlist(stringlist, df[col])
                elif col == 'extracurricular activities':
                    stringlist = self.data.extracurricular_activities()['complete_activity_list']
                    df[col] = self.string_list_to_numberedlist(stringlist, df[col])
                # Xu
                elif col == 'career aspirations':
                    stringlist = self.data.careers()['careers_list']
                    df[col] = self.string_list_to_numberedlist(stringlist, df[col])
                elif col == 'future topics':
                    stringlist = self.data.future_topics()['future_topics']
                    df[col] = self.string_list_to_numberedlist(stringlist, df[col])
                else:
                    logging.error(f"{col} is not a known column name")

                df[col] = self.preprocess_columns(df, col)

        return df
    
    def run_RNN_models(self, df, model, layers=2):
        # Iterate through the columns
        for col in self.X + self.Xu:
            if col not in self.numerical_cols:

                # Create and train the model
                if model == 'Simple':
                    rnn_model = self.create_simple_rnn_model(num_layers=layers)
                elif model == 'LSTM':
                    rnn_model = self.create_lstm_model(num_layers=layers)
                elif model == 'GRU':
                    rnn_model = self.create_gru_model(num_layers=layers)
                else:
                    raise ValueError("Did not choose an RNN model type.")
                
                # Normally, you would train the model with appropriate labels, but for this example, we'll use dummy data
                dummy_labels = np.random.rand(len(col), 1)  # Dummy target vectors of length 1
                rnn_model.fit(col, dummy_labels, epochs=10, batch_size=32)
            
                # Transform the sequences
                transformed_sequences = rnn_model.predict(col)
                df[col] = transformed_sequences
        
        return df
    
    def create_RNN_models(self, df):
        for layer in range(1, 4):
            result = self.run_RNN_models(self, df, 'Simple', layer)
            name = '/RNN_models/Simple'+str(layer)+'.csv'
            result.to_csv(name, index=False)
            self.run_RNN_models(self, df, 'LSTM', layer)
            name = '/RNN_models/LSTM'+str(layer)+'.csv'
            result.to_csv(name, index=False)
            self.run_RNN_models(self, df, 'GRU', layer)
            name = '/RNN_models/GRU'+str(layer)+'.csv'
            result.to_csv(name, index=False)

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

    # Create the RNN models and save them to their files
    preprocessor.create_RNN_models(df)
