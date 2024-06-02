import logging
from data_generation import DataGenerator
from privacy_metrics import calculate_privacy_metrics
from config import load_config
from preprocessing import PreProcessing
from neural_network import NeuralNetwork
from privatization import Privatizer
import pandas as pd

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
        if 'Generate Dataset' in config["running_model"]["parts_to_run"]:
            # Build the synthetic dataset and save it to a CSV file
            dataset_builder = DataGenerator()
            logging.debug("DataGenerator instance created")

            synthetic_dataset = dataset_builder.generate_synthetic_dataset(config["synthetic_data"]["num_samples"])
            logging.info("Synthetic dataset generated with %d samples.", len(synthetic_dataset))
            logging.info("First few rows of the synthetic dataset:\n%s", synthetic_dataset.head())
            synthetic_dataset.to_csv('Dataset.csv', index=False)
            logging.info("Synthetic dataset saved to Dataset.csv")
        else:
            synthetic_dataset = pd.read_csv(config["running_model"]["data path"])
            logging.debug("New synthetic dataset not generated")
        
        if'Privatize Dataset' in config["running_model"]["parts_to_run"]:
            # Choose privacy mechanism
            logging.info("Privatizing the dataset...")
            mechanism = config["privacy"]["mechanism"]

            # Privatizing the dataset
            privatizer = Privatizer(config)
            private_dataset = privatizer.privatize_dataset(synthetic_dataset)
            logging.info("Privatization completed using %s", mechanism)

            # Save the privatized dataset to a new CSV file
            private_dataset.to_csv('Privatized_Dataset.csv', index=False)
            logging.info("Privatized dataset saved to Privatized_Dataset.csv")
        else:
            private_dataset = pd.read_csv(config["running_model"]["privatized data path"])
            logging.debug("New privatized dataset not generated")

        if 'Calculate Privacy Metrics' in config["running_model"]["parts_to_run"]:
            # Calculate the privacy level of the privatized dataset
            logging.info("Calculating privacy metrics...")
            parameters = privatizer.parameters
            privacy_metrics = calculate_privacy_metrics(synthetic_dataset, private_dataset, parameters)
            logging.info("Privacy metrics calculated: %s", privacy_metrics)
        else:
            logging.debug("Privacy metrics not run")

        if 'Clean Privatized Dataset' in config["running_model"]["parts_to_run"]:
            # Drop the columns that should not impact recommendations for future topics
            columns_to_drop = config["preprocessing"]["remove_columns"]
            cleaned_dataset = private_dataset.drop(columns=columns_to_drop)
            logging.info("Removed senstitive information from privatized dataset")

            # Preprocess the cleaned dataset
            processor = PreProcessing(config)
            cleaned_dataset = processor.preprocessor(cleaned_dataset)
            logging.info("Preprocessing of cleaned dataset completed")

            # Save the cleaned dataset to a CSV
            cleaned_dataset.to_csv('Cleaned_Dataset.csv', index=False)
            logging.info("Cleaned dataset saved to Cleaned_Dataset.csv")
        else:
            cleaned_dataset = pd.read_csv(config["running_model"]["cleaned data path"])
            logging.debug("New cleaned dataset not generated")
        
        if 'Run Neural Network' in config["running_model"]["parts_to_run"]:
            # Run a neural network on the cleaned dataset
            network = NeuralNetwork(config)
            loss, accuracy = network.neural_network(cleaned_dataset)
            logging.info("Neural network loss %s", loss)
            logging.info("Neural network accuracy %s", accuracy)
        else:
            logging.debug("Neural network not run")

        if 'Simulate Data Attack' in config["running_model"]["parts_to_run"]:
            # Run a simulated data attack on the privatized dataset
            logging.info("Simulated data attacks are still under construction")
        else:
            logging.debug("Simulated data attacks not run")
               
    except Exception as e:
        logging.error("Error in main execution: %s", e)

if __name__ == "__main__":
    main()
