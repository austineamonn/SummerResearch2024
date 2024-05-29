import numpy as np
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPrivatizer:
    def __init__(self, config):
        self.gender_mapping = {"Male": "G1", "Female": "G2"}
        self.international_mapping = {"International": "S1", "Domestic": "S2"}
        self.config = config

    def privatizing_dataset(self, dataset: pd.DataFrame) -> (pd.DataFrame, list): # type: ignore
        """
        Main function to privatize the dataset.
        """
        logging.info("Starting dataset privatization...")
        
        private_dataset = dataset.copy()

        try:
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

        except Exception as e:
            logging.error("Error during dataset privatization: %s", e)
            raise

    def reverse_one_hot_encoding(self, df: pd.DataFrame, feature_prefix: str) -> pd.DataFrame:
        """
        Reverses one-hot encoding for a specific feature.
        """
        logging.debug("Reversing one-hot encoding for feature: %s", feature_prefix)
        try:
            feature_cols = [col for col in df.columns if col.startswith(feature_prefix)]
            df[feature_prefix] = df[feature_cols].idxmax(axis=1).apply(lambda x: x[len(feature_prefix) + 1:])
            df = df.drop(columns=feature_cols)
            return df

        except Exception as e:
            logging.error("Error during reversing one-hot encoding for %s: %s", feature_prefix, e)
            raise

    def aggregate_race_ethnicity(self, race: str) -> str:
        """
        Aggregates race/ethnicity into broader categories.
        """
        return "Non-Minority" if race in ["European American or white"] else "Minority"

    def add_random_noise(self, value: int, noise_level: int = 1) -> int:
        """
        Adds random noise to a value.
        """
        return value + np.random.randint(-noise_level, noise_level + 1)

    def add_gaussian_noise(self, value: float, sensitivity: float, epsilon: float, delta: float) -> float:
        """
        Adds Gaussian noise to a value.
        """
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        noise = np.random.normal(0, sigma, 1)[0]
        return value + noise

    def add_laplace_noise(self, value: float, sensitivity: float, epsilon: float) -> float:
        """
        Adds Laplace noise to a value.
        """
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale, 1)[0]
        return value + noise

    def privatizing_class_year(self, dataset: pd.DataFrame, private_dataset: pd.DataFrame) -> (pd.DataFrame, list): # type: ignore
        """
        Adds noise to the class year based on the chosen mechanism.
        """
        mechanism = self.config["privacy"]["mechanism"]
        sensitivity = self.config["privacy"]["sensitivity"]
        epsilon = self.config["privacy"]["epsilon"]
        delta = self.config["privacy"]["delta"]

        logging.debug("Privatizing class year with mechanism: %s", mechanism)
        try:
            if mechanism == 'Random':
                private_dataset['class year'] = dataset['class year'].apply(lambda x: self.add_random_noise(x, noise_level=1))
            elif mechanism == 'Gaussian':
                private_dataset['class year'] = dataset['class year'].apply(lambda x: self.add_gaussian_noise(x, sensitivity, epsilon, delta))
            elif mechanism == 'Laplace':
                private_dataset['class year'] = dataset['class year'].apply(lambda x: self.add_laplace_noise(x, sensitivity, epsilon))
            parameters = [mechanism, epsilon, delta]
            return private_dataset, parameters

        except Exception as e:
            logging.error("Error during privatizing class year: %s", e)
            raise

# Export the function for external use
def privatizing_dataset(dataset: pd.DataFrame, config: dict) -> (pd.DataFrame, list): # type: ignore
    privatizer = DataPrivatizer(config)
    return privatizer.privatizing_dataset(dataset)
