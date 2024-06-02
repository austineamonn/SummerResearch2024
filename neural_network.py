from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import logging
from config import load_config
from dictionary import Data
import pandas as pd

# Load configuration
config = load_config()

# Set up logging
logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])

class NeuralNetwork:
    def __init__(self, config):
        self.config = config
        data = Data(config)
        combined_data = data.get_data()
        self.future_topics = combined_data['future_topics']
        logging.debug("Neural Network class initialized")

    def neural_network(self, df):
        # Ensure the target is defined using existing columns
        existing_transformed_columns = [col for col in self.future_topics if col in df.columns]
        logging.debug("Here are the transformed columns that exist in the dataset: %s", existing_transformed_columns)

        # Log the columns of the DataFrame before dropping any columns
        logging.debug("DataFrame columns before dropping: %s", df.columns.tolist())

        # Define target first
        y = df[existing_transformed_columns]
        
        # Drop columns one by one with a check to ensure they exist
        for col in existing_transformed_columns:
            logging.debug("Examining the %s column", col)
            if col in df.columns:
                df = df.drop(columns=col)
            else:
                logging.debug("%s was not in the columns", col)

        # Log the columns of the DataFrame after dropping
        logging.debug("DataFrame columns after dropping: %s", df.columns.tolist())

        # Define features
        X = df
        logging.debug("Features and targets defined")

        # Convert y to a numpy array
        y = y.values
        logging.debug("Target converted to a numpy array")

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info("Test-train split completed")

        # Print shapes and types for debugging
        logging.debug(f'X_train shape: {X_train.shape}, X_train type: {type(X_train)}, X_train dtype: {X_train.dtypes}')
        logging.debug(f'y_train shape: {y_train.shape}, y_train type: {type(y_train)}, y_train dtype: {y_train.dtype}')


        # Define the neural network model
        model = Sequential([
            Dense(128, input_dim=X_train.shape[1], activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(min(y_train.shape[1], len(self.future_topics)), activation='sigmoid')  # Multi-label classification
        ])
        logging.debug("Neural network defined")

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        logging.debug("Model compiled")

        # Train the model
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
        logging.info("Model trained")

        # Evaluate the model
        loss, accuracy = model.evaluate(X_test, y_test)
        logging.debug("Model evaluated")
        
        # Return performance metrics
        return loss, accuracy
