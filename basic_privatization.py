import numpy as np
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPrivatizer:
    def __init__(self):
        self.gender_mapping = {"Male": "G1", "Female": "G2"}
        self.international_mapping = {"International": "S1", "Domestic": "S2"}

    def privatizing_dataset(self, dataset):
        """
        Main function to privatize the dataset.
        """
        logging.info("Starting dataset privatization...")
        
        private_dataset = dataset.copy()

        # Reversing one-hot encoding and privatizing features
        private_dataset = self.reverse_one_hot_encoding(private_dataset, 'race_ethnicity')
        private_dataset['race_ethnicity'] = private_dataset['race_ethnicity'].apply(self.aggregate_race_ethnicity)

        private_dataset = self.reverse_one_hot_encoding(private_dataset, 'gender')
        private_dataset['gender'] = private_dataset['gender'].map(self.gender_mapping)

        private_dataset = self.reverse_one_hot_encoding(private_dataset, 'international')
        private_dataset['international'] = private_dataset['international'].map(self.international_mapping)

        private_dataset, parameters = self.privatizing_class_year(dataset, private_dataset)
        
        logging.info("Dataset privatization complete.")
        return private_dataset, parameters

    def reverse_one_hot_encoding(self, df, feature_prefix):
        """
        Reverses one-hot encoding for a specific feature.
        """
        feature_cols = [col for col in df.columns if col.startswith(feature_prefix)]
        df[feature_prefix] = df[feature_cols].idxmax(axis=1).apply(lambda x: x[len(feature_prefix) + 1:])
        df = df.drop(columns=feature_cols)
        return df

    def aggregate_race_ethnicity(self, race):
        """
        Aggregates race/ethnicity into broader categories.
        """
        return "Non-Minority" if race in ["European American or white"] else "Minority"

    def add_random_noise(self, value, noise_level=1):
        """
        Adds random noise to a value.
        """
        return value + np.random.randint(-noise_level, noise_level + 1)

    def add_gaussian_noise(self, value, sensitivity, epsilon, delta):
        """
        Adds Gaussian noise to a value.
        """
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        noise = np.random.normal(0, sigma, 1)[0]
        return value + noise

    def add_laplace_noise(self, value, sensitivity, epsilon):
        """
        Adds Laplace noise to a value.
        """
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale, 1)[0]
        return value + noise

    def privatizing_class_year(self, dataset, private_dataset, mechanism='Random', sensitivity=1, epsilon=0.5, delta=1e-5):
        """
        Adds noise to the class year based on the chosen mechanism.
        """
        if mechanism == 'Random':
            private_dataset['class year'] = dataset['class year'].apply(lambda x: self.add_random_noise(x, noise_level=1))
        elif mechanism == 'Gaussian':
            private_dataset['class year'] = dataset['class year'].apply(lambda x: self.add_gaussian_noise(x, sensitivity, epsilon, delta))
        elif mechanism == 'Laplace':
            private_dataset['class year'] = dataset['class year'].apply(lambda x: self.add_laplace_noise(x, sensitivity, epsilon))
        parameters = [mechanism, epsilon, delta]
        return private_dataset, parameters

# Export the function for external use
def privatizing_dataset(dataset):
    privatizer = DataPrivatizer()
    return privatizer.privatizing_dataset(dataset)
