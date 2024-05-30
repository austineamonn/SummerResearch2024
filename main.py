import logging
from data_generation import generate_synthetic_dataset
from basic_privatization import DataPrivatizer
from privacy_metrics import calculate_privacy_metrics
from config import load_config
from pufferfish_privatization import PufferfishPrivatizer
from ddp_privatization import DDPPrivatizer

# Load configuration
config = load_config()

# Set up logging
logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])

def main():
    """
    First the function generates synthetic data.
    Second the function privatizes the dataset using one of several methods
    Third the function calculates the level of privacy of the privatized dataset through various privacy metrics.
    """

    logging.info("Loading data and generating synthetic dataset...")

    try:
        synthetic_dataset = generate_synthetic_dataset(config["synthetic_data"]["num_samples"])
        logging.info("Synthetic dataset generated with %d samples.", len(synthetic_dataset))
        logging.info("First few rows of the synthetic dataset:\n%s", synthetic_dataset.head())
        synthetic_dataset.to_csv('Dataset.csv', index=False)
        logging.info("Synthetic dataset saved to Dataset.csv.")

        # Choose privacy mechanism
        logging.info("Privatizing the dataset...")
        mechanism = config["privacy"]["mechanism"]
        if mechanism == "Pufferfish":
            # Extract Pufferfish-specific configuration
            pufferfish_config = config["privacy"]
            secrets = pufferfish_config["secrets"]
            discriminative_pairs = pufferfish_config["discriminative_pairs"]
            epsilon = pufferfish_config["epsilon"]

            # Create PufferfishPrivatizer instance
            privatizer = PufferfishPrivatizer(secrets, discriminative_pairs, epsilon)

            # Privatize dataset using Pufferfish mechanism
            private_dataset = privatizer.privatize_dataset(synthetic_dataset)
            logging.info("Pufferfish privatization applied.")
        elif mechanism == "DDP":
            # Extract DDP-specific configuration
            ddp_config = config["privacy"]["ddp"]
            secrets = config["privacy"]["secrets"]
            epsilon = config["privacy"]["epsilon"]
            correlation_coefficient = ddp_config["correlation_coefficient"]

            # Create DDPPrivatizer instance
            privatizer = DDPPrivatizer(secrets, epsilon, correlation_coefficient)

            # Privatize dataset using DDP mechanism
            private_dataset = privatizer.privatize_dataset(synthetic_dataset)
            logging.info("DDP privatization applied.")
        else:
            # Privatize dataset using basic privatization
            privatizer = DataPrivatizer(config)
            private_dataset = privatizer.privatizing_dataset(synthetic_dataset, config)
            logging.info("Basic privatization applied")

        # Save the privatized dataset to a new CSV file
        private_dataset.to_csv('Privatized_Dataset.csv', index=False)
        logging.info("Privatized dataset saved to Privatized_Dataset.csv.")

        logging.info("Calculating privacy metrics...")
        parameters = privatizer.parameters
        privacy_metrics = calculate_privacy_metrics(synthetic_dataset, private_dataset, parameters)
        logging.info("Privacy metrics calculated: %s", privacy_metrics)
        
    except Exception as e:
        logging.error("Error in main execution: %s", e)

if __name__ == "__main__":
    main()
