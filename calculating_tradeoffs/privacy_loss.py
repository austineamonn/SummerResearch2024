import os
import sys
import pandas as pd

# Add the SummerResearch2024 directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preprocessing.preprocessing import string_list_to_numberedlist

class PrivacyLoss:
    def __init__(self, config, data) -> None:
        self.config = config
        self.data = data

    def get_private_cols(self, df):
        privacy_cols = self.config["calculatig_tradeoffs"]["privacy_cols"]
        privacy_cols_df = df[privacy_cols].copy()
        for col in privacy_cols:
            if col == 'ethnoracial group':
                stringlist = self.data.ethnoracial_group()["ethnoracial_list"]
            elif col == 'gender':
                stringlist = self.data.gender()["gender_list"]
            elif col == 'international status':
                stringlist = self.data.international_status()["international_status_list"]
            privacy_cols_df.loc[:, col] = string_list_to_numberedlist(stringlist, privacy_cols_df[col])
        
        return privacy_cols_df

# Main execution
if __name__ == "__main__":
    # Import necessary dependencies
    from datafiles_for_data_construction.data import Data
    from config import load_config

    # Load configuration and data
    config = load_config()
    data = Data()

    # Import synthetic dataset
    df = pd.read_csv(config["running_model"]["data path"])

    # Create Loss Calculator
    loss_calculator = PrivacyLoss(config, data)

    # Get Private Columns and save them
    privacy_cols_df = loss_calculator.get_private_cols(df)
    privacy_cols_df.to_csv(config["running_model"]["private columns path"], index=False)
