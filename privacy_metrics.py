import pandas as pd
import logging
from typing import Tuple, Dict
from config import load_config

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
config = load_config()

class PrivacyMetrics:
    def __init__(self,config):
        self.config = config
        self.mechanism = self.config["privacy"]["mechanism"]

    def calculate_privacy_metrics(self, o_dataset: pd.DataFrame, p_dataset: pd.DataFrame, parameters: list) -> Dict:
        """
        Calculate various privacy metrics for the privatized dataset.
        """
        logging.info("Calculating privacy metrics...")

        try:
            # Compare statistics for the 'class year' column
            statistics_comparison = self.compare_statistics(o_dataset, p_dataset, 'class year')

            # Define quasi-identifiers and sensitive attributes for the dataset
            quasi_identifiers = ['race_ethnicity', 'gender', 'international']
            sensitive_attribute = 'class year'

            # Calculate k-anonymity and l-diversity of the privatized dataset
            k_anonymity = self.calculate_k_anonymity(p_dataset, quasi_identifiers)
            l_diversity = self.calculate_l_diversity(p_dataset, quasi_identifiers, sensitive_attribute)

            if self.mechanism == "Pufferfish":
                privacy_metrics = {
                "k-anonymity": k_anonymity,
                "l-diversity": l_diversity,
                "statistics": statistics_comparison,
                "secrets": parameters[0],
                "discriminative pairs": parameters[1],
                "epsilon": parameters[2]
            }
            elif self.mechanism == "DDP":
                privacy_metrics = {
                "k-anonymity": k_anonymity,
                "l-diversity": l_diversity,
                "statistics": statistics_comparison,
                "secrets": parameters[0],
                "epsilon": parameters[1],
                "correlation coefficient": parameters[2]
            }
            elif self.mechanism == "CBA":
                privacy_metrics = {
                "k-anonymity": k_anonymity,
                "l-diversity": l_diversity,
                "statistics": statistics_comparison,
                "secrets": parameters[0],
                "epsilon": parameters[1],
                "coupling strength": parameters[2]
            }  
            else:
                privacy_metrics = {
                    "k-anonymity": k_anonymity,
                    "l-diversity": l_diversity,
                    "statistics": statistics_comparison,
                    "mechanism": parameters[0],
                    "sensitivity": parameters[1],
                    "epsilon": parameters[2],
                    "delta": parameters[3]
                }
            
            return privacy_metrics

        except Exception as e:
            logging.error("Error during calculating privacy metrics: %s", e)
            raise

    def compare_statistics(self, original: pd.DataFrame, anonymized: pd.DataFrame, column: str) -> Dict:
        """
        Compare statistical properties between the original and anonymized datasets.
        """
        logging.debug("Comparing statistics for column: %s", column)
        try:
            original_mean = original[column].mean()
            anonymized_mean = anonymized[column].mean()
            original_std = original[column].std()
            anonymized_std = anonymized[column].std()
            return {
                "original_mean": original_mean,
                "anonymized_mean": anonymized_mean,
                "original_std": original_std,
                "anonymized_std": anonymized_std
            }

        except Exception as e:
            logging.error("Error during comparing statistics for %s: %s", column, e)
            raise

    def calculate_k_anonymity(self, df: pd.DataFrame, quasi_identifiers: list) -> int:
        """
        Calculate k-anonymity for the given quasi-identifiers.
        """
        logging.debug("Calculating k-anonymity for quasi-identifiers: %s", quasi_identifiers)
        try:
            equivalence_classes = df.groupby(quasi_identifiers).size()
            k_anonymity = equivalence_classes.min()
            return k_anonymity

        except Exception as e:
            logging.error("Error during calculating k-anonymity: %s", e)
            raise

    def calculate_l_diversity(self, df: pd.DataFrame, quasi_identifiers: list, sensitive_attribute: str) -> int:
        """
        Calculate l-diversity for the given quasi-identifiers and sensitive attribute.
        """
        logging.debug("Calculating l-diversity for quasi-identifiers: %s and sensitive attribute: %s", quasi_identifiers, sensitive_attribute)
        try:
            equivalence_classes = df.groupby(quasi_identifiers)
            l_diversity_list = [len(group[sensitive_attribute].value_counts()) for _, group in equivalence_classes]
            l_diversity = min(l_diversity_list)
            return l_diversity

        except Exception as e:
            logging.error("Error during calculating l-diversity: %s", e)
            raise

# Export the function for external use
def calculate_privacy_metrics(o_dataset: pd.DataFrame, p_dataset: pd.DataFrame, parameters: list) -> Dict:
    privacy_metrics_calculator = PrivacyMetrics(config)
    return privacy_metrics_calculator.calculate_privacy_metrics(o_dataset, p_dataset, parameters)
