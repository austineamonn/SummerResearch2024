import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message=s')

class PrivacyMetrics:
    def calculate_privacy_metrics(self, o_dataset, p_dataset, parameters):
        """
        Calculate various privacy metrics for the privatized dataset.
        """
        logging.info("Calculating privacy metrics...")

        # Compare statistics for the 'class year' column
        statistics_comparison = self.compare_statistics(o_dataset, p_dataset, 'class year')

        # Define quasi-identifiers and sensitive attributes for the dataset
        quasi_identifiers = ['race_ethnicity', 'gender', 'international']
        sensitive_attribute = 'class year'

        # Calculate k-anonymity and l-diversity of the privatized dataset
        k_anonymity = self.calculate_k_anonymity(p_dataset, quasi_identifiers)
        l_diversity = self.calculate_l_diversity(p_dataset, quasi_identifiers, sensitive_attribute)

        privacy_metrics = {
            "k-anonymity": k_anonymity,
            "l-diversity": l_diversity,
            "statistics": statistics_comparison,
            "mechanism": parameters[0],
            "epsilon": parameters[1],
            "delta": parameters[2]
        }

        logging.info("Privacy metrics calculated: %s", privacy_metrics)
        return privacy_metrics

    def compare_statistics(self, original, anonymized, column):
        """
        Compare statistical properties between the original and anonymized datasets.
        """
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

    def calculate_k_anonymity(self, df, quasi_identifiers):
        """
        Calculate k-anonymity for the given quasi-identifiers.
        """
        equivalence_classes = df.groupby(quasi_identifiers).size()
        k_anonymity = equivalence_classes.min()
        return k_anonymity

    def calculate_l_diversity(self, df, quasi_identifiers, sensitive_attribute):
        """
        Calculate l-diversity for the given quasi-identifiers and sensitive attribute.
        """
        equivalence_classes = df.groupby(quasi_identifiers)
        l_diversity_list = [len(group[sensitive_attribute].value_counts()) for _, group in equivalence_classes]
        l_diversity = min(l_diversity_list)
        return l_diversity

# Export the function for external use
def calculate_privacy_metrics(o_dataset, p_dataset, parameters):
    privacy_metrics_calculator = PrivacyMetrics()
    return privacy_metrics_calculator.calculate_privacy_metrics(o_dataset, p_dataset, parameters)
