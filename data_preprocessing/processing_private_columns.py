import os
import sys
import pandas as pd
import logging
import ast

# Add the SummerResearch2024 directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class PrivateColumns:
    def __init__(self, config, data) -> None:
        self.config = config
        self.data = data

    def string_to_numberedlist(self, stringlist, col):
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
            for item in rowlist:
                # Only strip if item is a string
                if isinstance(item, str):
                    item = item.strip()
                logging.debug(f"Processing item: {item}")
                if item in stringlist:
                    # For each item in the list add its index to the new list
                    number = stringlist.index(item)
                else:
                    logging.debug(f"Error: '{item}' is not in list")
                    logging.debug(f"Current stringlist: {stringlist}")
            
            # Add the list to the numbered list
            numberedlist.append(number)

        return numberedlist

    def get_private_cols(self, df):
        privacy_cols = self.config["calculating_tradeoffs"]["privacy_cols"]
        privacy_cols_df = df[privacy_cols].copy()
        for col in privacy_cols:
            if col == 'ethnoracial group':
                stringlist = self.data.ethnoracial_group()["ethnoracial_list"]
            elif col == 'gender':
                stringlist = self.data.gender()["gender_list"]
            elif col == 'international status':
                stringlist = self.data.international_status()["international_status_list"]
            privacy_cols_df.loc[:, col] = self.string_to_numberedlist(stringlist, privacy_cols_df[col])
        
        return privacy_cols_df

# Main execution
if __name__ == "__main__":
    # Import necessary dependencies
    #from datafiles_for_data_construction.data import Data
    from config import load_config

    # Load configuration and data
    config = load_config()
    """data = Data()

    # Import synthetic dataset
    df = pd.read_csv(config["running_model"]["data path"])

    # Create private columns class
    private_cols = PrivateColumns(config, data)

    # Get Private Columns and save them
    privacy_cols_df = private_cols.get_private_cols(df)
    privacy_cols_df.to_csv(config["running_model"]["private columns path"], index=False)"""

    # Combine dimensionality reduced data with privatized columns and utility columns
    privatization_type_list = ['NoPrivatization', 'Basic_DP', 'Basic_DP_LLC', 'Uniform', 'Uniform_LLC', 'Shuffling', 'Complete_Shuffling']
    layers = [1]

    # Private Columns pathways
    privacy_cols_df = pd.read_csv('Private_Columns.csv')

    # Utility Column Pathways
    GRU_utility_cols_df = pd.read_csv('reduced_dimensionality_data/NoPrivatization/GRUPrivate_Cols.csv')
    LSTM_utility_cols_df = pd.read_csv('reduced_dimensionality_data/NoPrivatization/LSTMPrivate_Cols.csv')
    Simple_utility_cols_df = pd.read_csv('reduced_dimensionality_data/NoPrivatization/SimplePrivate_Cols.csv')

    # Combine the Pathways
    for privatization_type in privatization_type_list:
        for layer in layers:
            GRU_df = pd.read_csv(f'reduced_dimensionality_data/{privatization_type}/GRU{layer}.csv')
            GRU_combined_df = pd.concat([GRU_df, GRU_utility_cols_df,privacy_cols_df], axis=1)
            GRU_combined_df.to_csv(f'reduced_dimensionality_data/{privatization_type}/GRU{layer}_combined.csv', index=False)

            LSTM_df = pd.read_csv(f'reduced_dimensionality_data/{privatization_type}/LSTM{layer}.csv')
            LSTM_combined_df = pd.concat([LSTM_df, LSTM_utility_cols_df, privacy_cols_df], axis=1)
            LSTM_combined_df.to_csv(f'reduced_dimensionality_data/{privatization_type}/LSTM{layer}_combined.csv', index=False)

            Simple_df = pd.read_csv(f'reduced_dimensionality_data/{privatization_type}/Simple{layer}.csv')
            Simple_combined_df = pd.concat([Simple_df, Simple_utility_cols_df, privacy_cols_df], axis=1)
            Simple_combined_df.to_csv(f'reduced_dimensionality_data/{privatization_type}/Simple{layer}_combined.csv', index=False)
