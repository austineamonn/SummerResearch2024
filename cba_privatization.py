import numpy as np
import pandas as pd
import logging
from config import load_config

# Load configuration
config = load_config()

# Set up logging
logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])

class CBAPrivatizer:
    def __init__(self, config):
        self.config = config
        self.secrets = self.config["privacy"]["secrets"]
        self.epsilon = self.config["privacy"]["epsilon"]
        self.coupling_strength = self.config["privacy"]["cba"]["coupling_strength"]
        self.parameters = [self.secrets, self.epsilon, self.coupling_strength]

    def privatize_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        logging.info("Starting CBA privatization...")

        private_dataset = dataset.copy()

        # Apply noise based on secrets
        for secret in self.secrets:
            private_dataset[secret] = private_dataset[secret].apply(self.add_cba_noise)

        logging.info("CBA privatization complete.")
        return private_dataset

    def add_cba_noise(self, value):
        """
        Adds noise to a value based on Coupled Behavior Analysis framework.
        The noise is adjusted based on the coupling strength.
        """
        scale = 1.0 / self.epsilon
        coupled_noise = np.random.laplace(0, scale) * self.coupling_strength
        independent_noise = np.random.laplace(0, scale) * (1 - self.coupling_strength)
        noise = coupled_noise + independent_noise
        return value + noise
