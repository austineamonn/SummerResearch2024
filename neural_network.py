import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import logging
from config import load_config
from dictionary import get_combined_data

# Load configuration
config = load_config()

# Set up logging
logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])

class NeuralNetwork:
    def __init__(self, config):
        self.config = config
        combined_data = get_combined_data()
        self.future_topics = combined_data['future_topics']

    def neural_network(self, df):
        # Drop 'previous courses' column if it's of type 'object'
        if df['previous courses'].dtype == 'object':
            df = df.drop(columns=['previous courses'])
        logging.debug("Previous courses dropped")

        # Define features and target
        X = df.drop(columns=self.future_topics)
        y = df[self.future_topics]
        logging.debug("Features defined")

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
            Dense(len(self.future_topics), activation='sigmoid')  # Multi-label classification
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
