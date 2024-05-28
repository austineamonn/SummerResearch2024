import pandas as pd
from collections import Counter

# Calculate the privacy metrics
def calculate_privacy_metrics(o_dataset, p_dataset, parameters):
    # Compare statistics for the 'student class year' column
    statistics_comparison = compare_statistics(o_dataset, p_dataset, 'student class year')

    # Define quasi-identifiers and sensitive attributes for the dataset
    quasi_identifiers = ['race/ethnicity', 'gender', 'international']
    sensitive_attribute = 'student class year'

    # Calculate k-anonymity and l-diversity of the privatized dataset
    k_anonymity = calculate_k_anonymity(p_dataset, quasi_identifiers)
    l_diversity = calculate_l_diversity(p_dataset, quasi_identifiers, sensitive_attribute)

    privacy_metrics = {
        "k-anonymity": k_anonymity,
        "l-diversity": l_diversity,
        "statistics": statistics_comparison,
        "mechanism": parameters[0],
        "epsilon": parameters[1],
        "delta": parameters[2]
    }

    # Return the privacy metrics
    return privacy_metrics

# Define utility metric functions
def compare_statistics(original, anonymized, column):
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

def calculate_k_anonymity(df, quasi_identifiers):
    """Calculate k-anonymity for the given quasi-identifiers."""
    equivalence_classes = df.groupby(quasi_identifiers).size()
    k_anonymity = equivalence_classes.min()
    return k_anonymity

def calculate_l_diversity(df, quasi_identifiers, sensitive_attribute):
    """Calculate l-diversity for the given quasi-identifiers and sensitive attribute."""
    equivalence_classes = df.groupby(quasi_identifiers)
    l_diversity_list = []
    
    for name, group in equivalence_classes:
        sensitive_counts = group[sensitive_attribute].value_counts()
        l_diversity_list.append(len(sensitive_counts))
    
    l_diversity = min(l_diversity_list)
    return l_diversity
