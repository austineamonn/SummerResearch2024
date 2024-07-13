import pandas as pd
import logging

class PrivacyMetrics:
    def __init__(self, config):
        self.config = config

        # Privatization Method and its associated parameters
        self.style = config["privacy"]["style"]
        self.parameters = config["privacy"][self.style]

        # Numerical Columns
        self.numerical_cols = config["privacy"]["numerical_columns"]

    def load_data():
        pass

    def calculate_basic_metrics(self, o_dataset: pd.DataFrame, p_dataset: pd.DataFrame):
        """
        Calculate various privacy metrics for the privatized dataset.
        """
        logging.info("Calculating privacy metrics...")

        