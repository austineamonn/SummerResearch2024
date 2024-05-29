
import numpy as np
import pandas as pd

# Remove any potential direct identifiers (assuming 'student class year' is the closest direct identifier)
# Anonymize the data while keeping a necessary level of utility.

def privatizing_dataset(dataset):
    # Create a copy to build the privatized version
    private_dataset = dataset.copy()

    # Reversing one-hot encoding of race_ethnicity
    private_dataset = reverse_one_hot_encoding(private_dataset, 'race_ethnicity')
    
    # Privatizing Ethnoracial Group
    private_dataset['race_ethnicity'] = private_dataset['race_ethnicity'].apply(aggregate_race_ethnicity)

    # Reversing one-hot encoding of gender
    private_dataset = reverse_one_hot_encoding(private_dataset, 'gender')

    # Pseudonymizing Gender
    gender_mapping = {"Male": "G1", "Female": "G2"}
    private_dataset['gender'] = private_dataset['gender'].map(gender_mapping)

    # Reversing one-hot encoding of International Status
    private_dataset = reverse_one_hot_encoding(private_dataset, 'international')

    # Pseudonymizing International Status
    international_mapping = {"International": "S1", "Domestic": "S2"}
    private_dataset['international'] = private_dataset['international'].map(international_mapping)

    # Privatizing Class Year
    private_dataset, parameters = privitizing_class_year(dataset, private_dataset)

    # Return the privatized dataset and parameters used
    return private_dataset, parameters

# Reversing one-hot encoding for a specific feature
def reverse_one_hot_encoding(df, feature_prefix):
    feature_cols = [col for col in df.columns if col.startswith(feature_prefix)]
    df[feature_prefix] = df[feature_cols].idxmax(axis=1).apply(lambda x: x[len(feature_prefix) + 1:])
    df = df.drop(columns=feature_cols)
    return df

# Aggregating race_ethnicity into broader categories
def aggregate_race_ethnicity(race):
    if race in ["European American or white"]:
        return "Non-Minority"
    else:
        return "Minority"

# Adding random noise to class year
def add_random_noise(value, noise_level=1):
    return value + np.random.randint(-noise_level, noise_level+1)

# Function to add Gaussian noise
def add_gaussian_noise(value, sensitivity, epsilon, delta):
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = np.random.normal(0, sigma, 1)[0]
    return value + noise

# Function to add Laplace noise
def add_laplace_noise(value, sensitivity, epsilon):
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, 1)[0]
    return value + noise

# Defining Parameters for differential privacy
# Sensitivity - Sensitivity of the class year data
# Epsilon - Privacy parameter (smaller values provide more privacy)
# Delta - A small probability of the mechanism failing

def privitizing_class_year(dataset, private_dataset, mechanism='Random', sensitivity=1, epsilon=0.5, delta=1e-5):
    if mechanism == 'Random':
        # Applying Random noise
        private_dataset['student class year'] = dataset['student class year'].apply(lambda x: add_random_noise(x, noise_level=1))
    elif mechanism == 'Gaussian':
        # Applying the Gaussian mechanism
        private_dataset['student class year'] = dataset['student class year'].apply(lambda x: add_gaussian_noise(x, sensitivity, epsilon, delta))
    elif mechanism == 'Laplace':
        # Applying the Laplace mechanism
        private_dataset['student class year'] = dataset['student class year'].apply(lambda x: add_laplace_noise(x, sensitivity, epsilon))
    parameters = [mechanism, epsilon, delta]
    return private_dataset, parameters
