import logging
from data_generation import generate_synthetic_dataset
from basic_privatization import privatizing_dataset
from privacy_metrics import calculate_privacy_metrics

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to generate, privatize datasets and calculate privacy metrics.
    """
    logging.info("Generating synthetic dataset...")
    synthetic_dataset = generate_synthetic_dataset()
    logging.info("Synthetic dataset generated.")

    # Print the first few rows of the original dataset
    logging.info("First few rows of the synthetic dataset:\n%s", synthetic_dataset.head())

    # Save the original dataset to CSV
    synthetic_dataset.to_csv('Dataset.csv', index=False)
    logging.info("Synthetic dataset saved to Dataset.csv.")

    # Privatize the dataset
    logging.info("Privatizing the dataset...")
    private_dataset, parameters = privatizing_dataset(synthetic_dataset)
    logging.info("Dataset privatized.")

    # Save the privatized dataset as a csv
    private_dataset.to_csv('Privatized_Dataset.csv', index=False)
    logging.info("Privatized dataset saved to Privatized_Dataset.csv.")

    # Calculate privacy metrics
    logging.info("Calculating privacy metrics...")
    privacy_metrics = calculate_privacy_metrics(synthetic_dataset, private_dataset, parameters)
    logging.info("Privacy metrics calculated: %s", privacy_metrics)

if __name__ == "__main__":
    main()
