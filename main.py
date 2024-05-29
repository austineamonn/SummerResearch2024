import logging
from data_generation import generate_synthetic_dataset
from basic_privatization import privatizing_dataset
from privacy_metrics import calculate_privacy_metrics
from config import load_config

# Load configuration
config = load_config()

# Set up logging
logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])

def main():
    """
    First the function generates synthetic data. Second the function privatizes the dataset.
    Third the function calculates the level of privacy of the privatized dataset through various privacy metrics.
    """

    logging.info("Loading data and generating synthetic dataset...")

    try:
        num_samples = config["synthetic_data"]["num_samples"]
        synthetic_dataset = generate_synthetic_dataset(num_samples=num_samples)
        logging.info("Synthetic dataset generated.")
        logging.info("First few rows of the synthetic dataset:\n%s", synthetic_dataset.head())
        synthetic_dataset.to_csv('Dataset.csv', index=False)
        logging.info("Synthetic dataset saved to Dataset.csv.")
        
        logging.info("Privatizing the dataset...")
        private_dataset, parameters = privatizing_dataset(synthetic_dataset, config)
        logging.info("Dataset privatization complete.")
        private_dataset.to_csv('Privatized_Dataset.csv', index=False)
        logging.info("Privatized dataset saved to Privatized_Dataset.csv.")
        
        logging.info("Calculating privacy metrics...")
        privacy_metrics = calculate_privacy_metrics(synthetic_dataset, private_dataset, parameters)
        logging.info("Privacy metrics calculated: %s", privacy_metrics)
        
    except Exception as e:
        logging.error("Error in main execution: %s", e)

if __name__ == "__main__":
    main()
