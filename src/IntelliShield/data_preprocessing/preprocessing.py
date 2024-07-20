import logging
import numpy as np
import pandas as pd
import os
import ast
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Bidirectional, LSTM, GRU, Dropout # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from IntelliShield.data_generation.data import Data

class PreProcessing:
    def __init__(self, data: Data, privatization_type: str = None, logger=None):
        # Set up logging
        self.logger = logger

        # Privatized - Utility split
        self.Xp = [
            'first name','last name','ethnoracial group','gender',
            'international status','socioeconomic status'
        ]
        self.X = [
            'learning style', 'gpa', 'student semester' ,'major' ,
            'previous courses','course types','course subjects',
            'subjects of interest', 'extracurricular activities'
        ]
        self.Xu = [
            'career aspirations', 'future topics'
        ]
        self.numerical_cols = [
            "gpa", "student semester"
        ]

        # Data
        self.data = data

        # Privatization type
        self.privatization_type = privatization_type

    def stringlist_to_binarylist(self, stringlist, col):
        binarylist = []

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
    
    def fix_course_types(self, course_types):
        unique_types = set()
        for sublist in course_types:
            for item in sublist:
                unique_types.add(item)
        return sorted(list(unique_types))
  
    def preprocess_columns(self, df, col):
        sequences = df[col].apply(lambda x: ast.literal_eval(str(x))).tolist()
        # Pad the sequences
        sequences = pad_sequences(sequences, padding='post', truncating='post')
        return sequences
    
    def create_simple_rnn_model(self, output_dim=1, num_layers=2, dropout_rate=0.2):
        model = Sequential()
        model.add(Embedding(input_dim=5000, output_dim=64))

        for _ in range(num_layers - 1):
            model.add(Bidirectional(SimpleRNN(50, return_sequences=True)))
            model.add(Dropout(dropout_rate))

        model.add(Bidirectional(SimpleRNN(50, return_sequences=False)))
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
        # Copy the dataframe
        new_df = df.copy()

        # Drop the Xp columns
        new_df = new_df.drop(columns=self.Xp)
        logging.debug(f"{self.Xp} were dropped")

        # Iterate through the columns
        for col in self.X + self.Xu:
            if col not in self.numerical_cols:
                # X
                if col == 'learning style':
                    stringlist = self.data.learning_style()['learning_style_list']
                    new_df[col] = self.stringlist_to_binarylist(stringlist, new_df[col])
                elif col == 'major':
                    stringlist = self.data.major()['majors_list']
                    new_df[col] = self.string_list_to_numberedlist(stringlist, new_df[col])
                elif col == 'previous courses':
                    stringlist = self.data.course()['course_names_list']
                    new_df[col] = self.string_list_to_numberedlist(stringlist, new_df[col])
                elif col == 'course types':
                    # Apply special function to fix the 'course types' column to be one nonrepeating list
                    new_df[col] = new_df[col].astype(str).apply(eval).apply(self.fix_course_types)
                    stringlist = self.data.course()['course_type_list']
                    new_df[col] = self.string_list_to_numberedlist(stringlist, new_df[col])
                elif col == 'course subjects':
                    stringlist = self.data.course()['course_subject']
                    new_df[col] = self.string_list_to_numberedlist(stringlist, new_df[col])
                elif col == 'subjects of interest':
                    stringlist = self.data.subjects()['subjects_list']
                    new_df[col] = self.string_list_to_numberedlist(stringlist, new_df[col])
                elif col == 'extracurricular activities':
                    stringlist = self.data.extracurricular_activities()['complete_activity_list']
                    new_df[col] = self.string_list_to_numberedlist(stringlist, new_df[col])
                # Xu
                elif col == 'career aspirations':
                    stringlist = self.data.careers()['careers_list']
                    new_df[col] = self.string_list_to_numberedlist(stringlist, new_df[col])
                elif col == 'future topics':
                    stringlist = self.data.future_topics()['future_topics']
                    new_df[col] = self.string_list_to_numberedlist(stringlist, new_df[col])
                else:
                    logging.error(f"{col} is not a known column name")
                
        return new_df
    
    def run_RNN_models(self, df, model, layers=2, utility=False):
        df_copy = df.copy()

        if utility: # Reduce dimensionality of utility columns
            columns = self.Xu
        else: # Reduce dimensionality of regular columns
            columns = self.X

        for col in columns:
            if col not in self.numerical_cols:
                preprocessed_col = self.preprocess_columns(df_copy, col)

                if model == 'Simple':
                    rnn_model = self.create_simple_rnn_model(num_layers=layers)
                elif model == 'LSTM':
                    rnn_model = self.create_lstm_model(num_layers=layers)
                elif model == 'GRU':
                    rnn_model = self.create_gru_model(num_layers=layers)
                else:
                    raise ValueError("Did not choose an RNN model type.")

                dummy_labels = np.random.rand(len(preprocessed_col), 1)
                early_stopping = EarlyStopping(monitor='loss', patience=3)
                rnn_model.fit(preprocessed_col, dummy_labels, epochs=10, batch_size=32, callbacks=[early_stopping])

                transformed_sequences = rnn_model.predict(preprocessed_col)
                df_copy[col] = list(transformed_sequences)

        return df_copy
    
    def create_RNN_models(self, df, end_range=1, save_files=False, utility=False, simple=True, LSTM=True, GRU=True, dir_path: str = None):
        results = []

        for layer in range(1, end_range + 1):
            if simple:
                simple_df = self.run_RNN_models(df, 'Simple', layer, utility)
                if save_files:
                    simple_df.to_csv(os.path.join(dir_path, 'reduced_dimensionality_data', self.privatization_type, f'Simple{layer}.csv'), index=False)
                else:
                    results.append(simple_df)

            if LSTM:
                LSTM_df = self.run_RNN_models(df, 'LSTM', layer, utility)
                if save_files:
                    LSTM_df.to_csv(os.path.join(dir_path, 'reduced_dimensionality_data', self.privatization_type, f'LSTM{layer}.csv'), index=False)
                else:
                    results.append(LSTM_df)

            if GRU:
                GRU_df = self.run_RNN_models(df, 'GRU', layer, utility)
                if save_files:
                    GRU_df.to_csv(os.path.join(dir_path,'reduced_dimensionality_data', self.privatization_type, f'GRU{layer}.csv'), index=False)
                else:
                    results.append(GRU_df)
        
        if not save_files:
            return results
        
# Main execution
if __name__ == "__main__":
    # Import necessary dependencies
    from IntelliShield.logger import setup_logger

    # Load logger and data
    logger = setup_logger('preprocessing_logger', 'preprocessing.log')
    data = Data()

    # Options: NoPrivatization, Basic_DP, Basic_DP_LLC, Uniform, Uniform_LLC, Shuffling, Complete_Shuffling
    privatization_type = 'NoPrivatization'

    # Import synthetic dataset and preprocess it
    df = pd.read_csv('../data_generation/Dataset')
    preprocessor = PreProcessing(data, privatization_type, logger)
    df = preprocessor.preprocess_dataset(df)
    df.to_csv('../../../outputs/examples/preprocessed_data/Preprocessed_Dataset.csv')

    # Dimensionality Reduction for each privatization type
    df = pd.read_csv(f'../../../outputs/examples/privatized_datasets/{privatization_type}_Dataset')
    preprocessor.create_RNN_models(df, save_files=True)
