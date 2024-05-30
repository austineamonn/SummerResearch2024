import numpy as np
import pandas as pd
import logging
from config import load_config

# Load configuration
config = load_config()

# Set up logging
logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])

class PufferfishPrivatizer:
    def __init__(self, secrets, discriminative_pairs, epsilon):
        self.secrets = secrets
        self.discriminative_pairs = discriminative_pairs
        self.epsilon = epsilon
        self.parameters = [self.secrets, self.discriminative_pairs, self.epsilon]

    def privatize_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        logging.info("Starting Pufferfish privatization...")

        private_dataset = dataset.copy()

        # Apply noise based on secrets and discriminative pairs
        for secret in self.secrets:
            for discriminative_pair in self.discriminative_pairs:
                private_dataset[secret] = private_dataset[secret].apply(
                    lambda x: self.add_pufferfish_noise(x, discriminative_pair)
                )

        logging.info("Pufferfish privatization complete.")
        return private_dataset

    def add_pufferfish_noise(self, value, discriminative_pair):
        """
        Adds noise to a value based on the Pufferfish framework.
        The noise is influenced by the discriminative pair.
        """
        secret_1, secret_2 = discriminative_pair

        # Adjust noise scale based on the discriminative pair
        if secret_1 in self.secrets:
            scale = 1.0 / (self.epsilon * 2)
        elif secret_2 in self.secrets:
            scale = 1.0 / (self.epsilon * 1.5)
        else:
            scale = 1.0 / self.epsilon

        # Add Laplace noise with the calculated scale
        noise = np.random.laplace(0, scale)
        return value + noise