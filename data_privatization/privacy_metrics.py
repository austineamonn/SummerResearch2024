import pandas as pd
import logging
from typing import  Dict
from config import load_config

# Load configuration
config = load_config()

# Set up logging
logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])

class PrivacyMetrics:
    def __init__(self, config):
        self.config = config

        # Privatization Method and its associated parameters
        self.style = config["privacy"]["style"]
        self.parameters = config["privacy"][self.style]

        # Numerical Columns
        self.numerical_cols = config["privacy"]["numerical_columns"]

    def calculate_privacy_metrics(self, o_dataset: pd.DataFrame, p_dataset: pd.DataFrame) -> Dict:
        """
        Calculate various privacy metrics for the privatized dataset.
        """
        logging.info("Calculating privacy metrics...")

        try:
            # Compare statistics for the 'student semester' column
            statistics_comparison_df = self.compare_statistics(o_dataset, p_dataset)

            statistics_comparison_df.to_csv(config["running_model"]["statistics comparison path"], index=False)
            logging.info("Privatization comparison data saved to 'Stats_Comparison_Dataset.csv'")

            calculated_metrics = {
                "privatization method": self.style
            }
            # Demographics to Extracurriculars - combine the 3 dictionaries into one
            privacy_metrics = {k: v for d in (calculated_metrics, self.parameters) for k, v in d.items()}
            
            return privacy_metrics

        except Exception as e:
            logging.error("Error during calculating privacy metrics: %s", e)
            raise

    def compare_statistics(self, original: pd.DataFrame, anonymized: pd.DataFrame) -> Dict:
        """
        Compare statistical properties between the original and anonymized datasets.
        Input: preprocessed dataset, privatized dataset
        Output: statistical comparison dataset
        """

        try:
            # Intialize the lists
            original_means = []
            anonymized_means = []
            original_stds = []
            anonymized_stds = []
            original_sums = []
            anonymized_sums = []

            # Find the utility columns
            suffixes = ['_career aspirations', '_future topics']

            # Find columns that end with any of the specified suffixes
            utility_cols = [col for col in original.columns if col.endswith(tuple(suffixes))]

            # Cut out the utility columns
            original = original.drop(columns=utility_cols)
            anonymized = anonymized.drop(columns=utility_cols)

            # Get the column names
            column_names = original.columns.tolist()

            # Iterate through each column
            for col in column_names:
                print(col)
                logging.debug("Comparing statistics for column: %s", col)
                # Numerical Columns
                if col in self.numerical_cols:
                    # Mean
                    original_mean = original[col].mean()
                    original_means.append(original_mean)
                    anonymized_mean = anonymized[col].mean()
                    anonymized_means.append(anonymized_mean)

                    # Standard Deviation
                    original_std = original[col].std()
                    original_stds.append(original_std)
                    anonymized_std = anonymized[col].std()
                    anonymized_stds.append(anonymized_std)

                    # Sum - empty as it makes no sense for the numerical columns
                    original_sums.append(None)
                    anonymized_sums.append(None)

                # Nonnumerical Columns
                else:
                    # Mean
                    original_mean = original[col].mean()
                    original_means.append(original_mean)
                    anonymized_mean = anonymized[col].mean()
                    anonymized_means.append(anonymized_mean)

                    # Standard Deviation - empty because it makes no sense for binarized data
                    original_stds.append(None)
                    anonymized_stds.append(None)

                    # Sum
                    original_sum = original[col].sum()
                    original_sums.append(original_sum)
                    anonymized_sum = anonymized[col].sum()
                    anonymized_sums.append(anonymized_sum)

            # Creating a DataFrame
            data = {
                'Column': column_names,
                'Original Mean': original_means,
                'Anonymized Mean': anonymized_means,
                'Original Std': original_stds,
                'Anonymized Std': anonymized_stds,
                'Original Sum': original_sums,
                'Anonymized Sum': anonymized_sums
            }

            df = pd.DataFrame(data)

            return df

        except Exception as e:
            logging.error("Error during comparing statistics for %s: %s", col, e)
            raise
