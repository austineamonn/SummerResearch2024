import logging
from data_generation import generate_synthetic_dataset
from basic_privatization import DataPrivatizer
from privacy_metrics import calculate_privacy_metrics
from config import load_config
from pufferfish_privatization import PufferfishPrivatizer
from ddp_privatization import DDPPrivatizer
from cba_privatization import CBAPrivatizer
from preprocessing import PreProcessing
from neural_network import NeuralNetwork

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
        # Build the synthetic dataset and save it to a CSV file
        synthetic_dataset = generate_synthetic_dataset(config["synthetic_data"]["num_samples"])
        logging.info("Synthetic dataset generated with %d samples.", len(synthetic_dataset))
        logging.info("First few rows of the synthetic dataset:\n%s", synthetic_dataset.head())
        synthetic_dataset.to_csv('Dataset.csv', index=False)
        logging.info("Synthetic dataset saved to Dataset.csv.")

        # Choose privacy mechanism
        logging.info("Privatizing the dataset...")
        mechanism = config["privacy"]["mechanism"]
        if mechanism == "Pufferfish":
            # Create PufferfishPrivatizer instance
            privatizer = PufferfishPrivatizer(config)

            # Privatize dataset using Pufferfish mechanism
            private_dataset = privatizer.privatize_dataset(synthetic_dataset)
            logging.info("Pufferfish privatization applied.")
        elif mechanism == "DDP":
            # Create DDPPrivatizer instance
            privatizer = DDPPrivatizer(config)

            # Privatize dataset using DDP mechanism
            private_dataset = privatizer.privatize_dataset(synthetic_dataset)
            logging.info("DDP privatization applied.")
        elif mechanism == "CBA":
            # Create CBAPrivatizer instance
            privatizer = CBAPrivatizer(config)

            # Privatize dataset using CBA mechanism
            private_dataset = privatizer.privatize_dataset(synthetic_dataset)
            logging.info("CBA privatization applied.")
        else:
            # Privatize dataset using basic privatization
            privatizer = DataPrivatizer(config)
            private_dataset = privatizer.privatizing_dataset(synthetic_dataset, config)
            logging.info("Basic privatization applied")

        # Save the privatized dataset to a new CSV file
        private_dataset.to_csv('Privatized_Dataset.csv', index=False)
        logging.info("Privatized dataset saved to Privatized_Dataset.csv.")

        # Calculate the privacy level of the privatized dataset
        logging.info("Calculating privacy metrics...")
        parameters = privatizer.parameters
        privacy_metrics = calculate_privacy_metrics(synthetic_dataset, private_dataset, parameters)
        logging.info("Privacy metrics calculated: %s", privacy_metrics)

        # Drop the columns that should not impact recommendations for future topics
        columns_to_drop = ["first_name", "last_name", "race_ethnicity", "gender", "international", "socioeconomic status"]
        cleaned_dataset = private_dataset.drop(columns=columns_to_drop)
        logging.info("Removed senstitive information from privatized dataset")

        # Preprocess the cleaned dataset
        processor = PreProcessing(config)
        cleaned_dataset = processor.preprocessor(cleaned_dataset)
        logging.info("Preprocessing of cleaned dataset completed")

        # Save the cleaned dataset to a CSV
        cleaned_dataset.to_csv('Cleaned_Dataset.csv', index=False)
        logging.info("Cleaned dataset saved to Cleaned_Dataset.csv.")

        network = NeuralNetwork(config)
        loss, accuracy = network.neural_network(cleaned_dataset)
        logging.info("Neural network loss %s", loss)
        logging.info("Neural network accuracy %s", accuracy)
        
    except Exception as e:
        logging.error("Error in main execution: %s", e)

if __name__ == "__main__":
    main()
