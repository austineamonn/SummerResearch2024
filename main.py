import logging
import pandas as pd
from config import load_config
from datafiles_for_data_construction.data import Data

# Load configuration
config = load_config()

# Set up logging
logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])

# Only import classes that will be used
if 'Generate Dataset' in config["running_model"]["parts_to_run"]:
    processing_unit = config["running_model"]["processing_unit"]
    # Pick GPU vs CPU
    if processing_unit == 'CPU':
        from data_generation.data_generation_CPU import DataGenerator
    elif processing_unit == 'GPU':
        from data_generation.data_generation_GPU import DataGenerator
    else:
        raise ValueError("Did not choose a proper processing unit. Pick CPU or GPU")
if 'Analyze Dataset' in config["running_model"]["parts_to_run"]:
    from data_generation.data_analysis import DataAnalysis
if 'Preprocess Dataset' in config["running_model"]["parts_to_run"] or 'Create RNNs' in config["running_model"]["parts_to_run"]:
    from data_preprocessing.preprocessing import PreProcessing
if 'Privatize Dataset' in config["running_model"]["parts_to_run"]:
    from SummerResearch2024.data_privatization.privatization import Privatizer
if 'Calculate Privacy Metrics' in config["running_model"]["parts_to_run"]:
    from data_privatization.privacy_metrics import PrivacyMetrics
if 'Run Neural Network' in config["running_model"]["parts_to_run"]:
    from neural_network.neural_network import NeuralNetwork

def main():
    """
    First the function generates synthetic data.
    Second the function privatizes the dataset using one of several methods
    Third the function calculates the level of privacy of the privatized dataset through various privacy metrics.
    """

    logging.info("Loading data and running model...")

    try:
        # Access the Datafiles
        data = Data()
        
        if 'Generate Dataset' in config["running_model"]["parts_to_run"]:
            # Build the synthetic dataset and save it to a CSV file
            dataset_builder = DataGenerator(config, data)
            logging.debug("DataGenerator instance created")

            synthetic_dataset = dataset_builder.generate_synthetic_dataset(config["synthetic_data"]["num_samples"])
            logging.info("Synthetic dataset generated with %d samples.", len(synthetic_dataset))
            logging.info("First few rows of the synthetic dataset:\n%s", synthetic_dataset.head())
            synthetic_dataset.to_csv(config["running_model"]["data path"], index=False)
            logging.info("Synthetic dataset saved to Dataset.csv")
        else:
            synthetic_dataset = pd.read_csv(config["running_model"]["data path"])
            logging.debug("New synthetic dataset not generated")

        if 'Analyze Dataset' in config["running_model"]["parts_to_run"]:
            logging.info("Analyzing the dataset...")

            # Analyze the dataset
            analyzer = DataAnalysis(config, synthetic_dataset)
            analyzer.analyze_data()
            logging.info("Synthetic dataset analyzed and results are saved to data_analysis_graphs")
        else:
            logging.debug("Synthetic dataset not analyzed")  

        if 'Preprocess Dataset' in config["running_model"]["parts_to_run"] or 'Create RNNs' in config["running_model"]["parts_to_run"]:
            # Initialize the preprocessor class
            preprocesser = PreProcessing(config, data)

            if 'Preprocess Dataset' in config["running_model"]["parts_to_run"]:
                logging.info("Preprocessing the dataset...")

                # Preprocessing the dataset
                preprocessed_dataset = preprocesser.preprocess_dataset(synthetic_dataset)
                logging.info("Preprocessing completed")

                # Save the preprocessed dataset to a new CSV file
                preprocessed_dataset.to_csv(config["running_model"]["preprocessed data path"], index=False)
                logging.info("Preprocessed dataset saved to Preprocessed_Dataset.csv")
            else:
                preprocessed_dataset = pd.read_csv(config["running_model"]["preprocessed data path"])            
                logging.debug("New preprocessed dataset not generated")  

            if 'Create RNNs'  in config["running_model"]["parts_to_run"]:
                logging.info("Creating the RNNs...")

                # Create the RNNs
                preprocesser.create_RNN_models(preprocessed_dataset)
                logging.info("RNNs created and the results are saved in RNN_models")
        else:
                preprocessed_dataset = pd.read_csv(config["running_model"]["preprocessed data path"])            
                logging.debug("New preprocessed dataset not generated")  
        
        if 'Privatize Dataset' in config["running_model"]["parts_to_run"]:
            # Choose privacy style
            logging.info("Privatizing the dataset...")
            style = config["privacy"]["style"]

            # Privatizing the dataset
            privatizer = Privatizer(config, data, config["privacy"]["style"], config["privacy"]["list_len"])
            private_dataset = privatizer.privatize_dataset(preprocessed_dataset)
            logging.info("Privatization completed using %s", style)

            # Save the privatized dataset to a new CSV file
            private_dataset.to_csv(config["running_model"]["privatized data path"], index=False)
            logging.info("Privatized dataset saved to Privatized_Dataset.csv")

            # Calculate dimensionality of the problem
            dimensionality = private_dataset.shape[1]
            logging.info("The Dimensionality of the problem is %s", dimensionality)
        else:
            private_dataset = pd.read_csv(config["running_model"]["privatized data path"])
            logging.debug("New privatized dataset not generated")

        if 'Calculate Privacy Metrics' in config["running_model"]["parts_to_run"]:
            # Calculate the privacy level of the privatized dataset
            logging.info("Calculating privacy metrics...")
            metrics_privatizer = PrivacyMetrics(config)
            privacy_metrics = metrics_privatizer.calculate_privacy_metrics(preprocessed_dataset, private_dataset)
            logging.info("Privacy metrics calculated: %s", privacy_metrics)
        else:
            logging.debug("Privacy metrics not run")
        
        if 'Run Neural Network' in config["running_model"]["parts_to_run"]:
            # Run a neural network on the cleaned dataset
            network = NeuralNetwork(config)
            X_train, y_train, X_test, y_test = network.test_train_split(private_dataset)
            
            # Tune hyperparameters of the Neural Network
            if 'Tune Neural Network' in config["running_model"]["parts_to_run"]:
                model = network.tune_hyperparameters(X_train, y_train, X_test, y_test)
            else:
                model = network.build_neural_network(X_train, y_train, X_test, y_test)
                logging.debug("Neural network tuning not run")
            loss, accuracy = network.evaluate_model(model, X_test, y_test)
            logging.info("Neural network loss %s", loss)
            logging.info("Neural network accuracy %s", accuracy)

            # Test the Neural Network
            if 'Test Neural Network' in config["running_model"]["parts_to_run"]:
                ave_loss, ave_accuracy = network.cross_validate(private_dataset)
                logging.info("Neural network average loss from cross validation: %s", ave_loss)
                logging.info("Neural network average accuracy from cross validation: %s", ave_accuracy)
                feature_importance = network.get_feature_importance(model, X_test, y_test)
                feature_importance.to_csv('Feature_Importance.csv', index=False)
                logging.info("Neural network features ranked by importance: %s and saved to Feature_Importance.csv", feature_importance.head())
            else:
                logging.debug("Neural network testing not run")
        else:
            logging.debug("Neural network not run")
               
    except Exception as e:
        logging.error("Error in main execution: %s", e)

if __name__ == "__main__":
    main()
