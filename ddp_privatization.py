import numpy as np
import pandas as pd
import logging
from config import load_config

# Load configuration
config = load_config()

# Set up logging
logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])

class DDPPrivatizer:
    def __init__(self, config):
        self.config = config
        self.secrets = self.config["privacy"]["secrets"]
        self.epsilon = self.config["privacy"]["epsilon"]
        self.correlation_coefficient = ["privacy"]["ddp"]["correlation_coefficient"]
        self.parameters = [self.secrets, self.epsilon, self.correlation_coefficient]

    def privatize_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        logging.info("Starting DDP privatization...")

        private_dataset = dataset.copy()

        # Apply noise based on secrets
        for secret in self.secrets:
            private_dataset[secret] = private_dataset[secret].apply(self.add_ddp_noise)

        logging.info("DDP privatization complete.")
        return private_dataset

    def add_ddp_noise(self, value):
        """
        Adds noise to a value based on Dependent Differential Privacy framework.
        The noise is adjusted based on the correlation coefficient.
        """
        scale = 1.0 / self.epsilon
        correlated_noise = np.random.laplace(0, scale) * self.correlation_coefficient
        independent_noise = np.random.laplace(0, scale) * (1 - self.correlation_coefficient)
        noise = correlated_noise + independent_noise
        return value + noise