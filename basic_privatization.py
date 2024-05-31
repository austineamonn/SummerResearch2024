import numpy as np
import pandas as pd
import logging
from config import load_config

# Load configuration
config = load_config()

# Set up logging
logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])

class DataPrivatizer:
    def __init__(self, config):
        self.gender_mapping = {"Male": "G1", "Female": "G2"}
        self.international_mapping = {"International": "S1", "Domestic": "S2"}
        self.config = config
        self.mechanism = self.config["privacy"]["basic_mechanism"]
        self.sensitivity = self.config["privacy"]["sensitivity"]
        self.epsilon = self.config["privacy"]["epsilon"]
        self.delta = self.config["privacy"]["delta"]
        self.parameters = [self.mechanism, self.sensitivity,self.epsilon,self.delta]

    def privatizing_dataset(self, dataset: pd.DataFrame) -> (pd.DataFrame, list): # type: ignore
        """
        Main function to privatize the dataset.
        """
        logging.info("Starting dataset privatization...")

        private_dataset = dataset.copy()

        try:
            # Add noise to numerical features
            numerical_columns = ['previous courses count', 'unique subjects in courses', 'subjects diversity', 'activities involvement count', 'gpa']
            for col in numerical_columns:
                private_dataset = self.add_noise(private_dataset, col)

            # Privatizing features
            private_dataset['race_ethnicity'] = private_dataset['race_ethnicity'].apply(self.aggregate_race_ethnicity)

            private_dataset['gender'] = private_dataset['gender'].map(self.gender_mapping)

            private_dataset['international'] = private_dataset['international'].map(self.international_mapping)

            private_dataset, parameters = self.privatizing_class_year(dataset, private_dataset)

            logging.info("Dataset privatization complete.")
            return private_dataset, parameters

        except Exception as e:
            logging.error("Error during dataset privatization: %s", e)
            raise

    def aggregate_race_ethnicity(self, race: str) -> str:
        """
        Aggregates race/ethnicity into broader categories.
        """
        return "Non-Minority" if race in ["European American or white"] else "Minority"

    def add_random_noise(self, value: int, noise_level: int) -> int:
        """
        Adds random noise to a value.
        """
        return value + np.random.randint(-noise_level, noise_level)

    def add_gaussian_noise(self, value: int, sensitivity: float, epsilon: float, delta: float) -> float:
        """
        Adds Gaussian noise to a value.
        """
        scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        noise = np.random.normal(0, scale, 1)[0]
        return value + noise

    def add_laplace_noise(self, value: int, sensitivity: float, epsilon: float) -> float:
        """
        Adds Laplace noise to a value.
        """
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale, 1)[0]
        return value + noise
    
    def add_exponential_noise(self, value: int, scale: float) -> float:
        """
        Adds Exponential noise to a value.
        """
        noise = np.random.exponential(scale, 1)[0]
        return value + noise

    def add_gamma_noise(self, value: int, shape: float, scale: float) -> float:
        """
        Adds Gamma noise to a value.
        """
        noise = np.random.gamma(shape, scale, 1)[0]
        return value + noise

    def add_uniform_noise(self, value: int, low: float, high: float) -> float:
        """
        Adds Uniform noise to a value.
        """
        noise = np.random.uniform(low, high, 1)[0]
        return value + noise

    def privatizing_class_year(self, dataset: pd.DataFrame, private_dataset: pd.DataFrame) -> (pd.DataFrame, list): # type: ignore
        """
        Adds noise to the class year based on the chosen mechanism.
        """

        logging.debug("Privatizing class year with mechanism: %s", self.mechanism)
        try:
            if self.mechanism == 'Random':
                private_dataset['class year'] = dataset['class year'].apply(lambda x: self.add_random_noise(x, noise_level=1))
            elif self.mechanism == 'Gaussian':
                private_dataset['class year'] = dataset['class year'].apply(lambda x: self.add_gaussian_noise(x, self.sensitivity, self.epsilon, self.delta))
            elif self.mechanism == 'Laplace':
                private_dataset['class year'] = dataset['class year'].apply(lambda x: self.add_laplace_noise(x, self.sensitivity, self.epsilon))
            elif self.mechanism == 'Exponential':
                private_dataset['class year'] = dataset['class year'].apply(lambda x: self.add_exponential_noise(x, scale=1/self.epsilon))
            elif self.mechanism == 'Gamma':
                private_dataset['class year'] = dataset['class year'].apply(lambda x: self.add_gamma_noise(x, shape=2, scale=1))
            elif self.mechanism == 'Uniform':
                private_dataset['class year'] = dataset['class year'].apply(lambda x: self.add_uniform_noise(x, low=-1, high=1))
            return private_dataset

        except Exception as e:
            logging.error("Error during privatizing class year: %s", e)
            raise

    def add_noise(self, dataframe, column_name, noise_level=0.01):
        """
        Add noise to numerical features for privacy.
        """
        noise = np.random.normal(0, noise_level, dataframe[column_name].shape)
        dataframe[column_name] += noise
        return dataframe

# Export the function for external use
def privatizing_dataset(dataset: pd.DataFrame, config: dict) -> (pd.DataFrame, list): # type: ignore
    privatizer = DataPrivatizer(config)
    return privatizer.privatizing_dataset(dataset)
