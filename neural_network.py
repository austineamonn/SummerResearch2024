import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import KFold, train_test_split
from sklearn.inspection import permutation_importance
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from config import load_config
from dictionary import Data

# Load configuration
config = load_config()

# Set up logging
logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])

class NeuralNetwork:
    def __init__(self, config):
        self.config = config
        data = Data(config)
        self.optimizer = config["neural_network"]["optimizer"]
        self.optimizer_params = config["neural_network"]["optimizer_params"]
        self.loss = config["neural_network"]["loss"]
        self.metrics = config["neural_network"]["metrics"]
        combined_data = data.get_data()
        self.future_topics = [col for col in combined_data['future_topics']]
        #self.future_topics = [col.lower() for col in combined_data['future_topics']]  # Convert to lowercase
        """ self.target_columns = []
        # Note that these test train splits are only for the main model
        # not the crossvalidation model
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.model = None
        self.df = df

        # Convert DataFrame columns to lowercase
        self.df.columns = self.df.columns.str.lower()"""

    def test_train_split(self, df):
        # Ensure the target is defined using existing columns
        target_columns = [col for col in self.future_topics if col in self.df.columns]
        logging.debug("Here are the transformed columns that exist in the dataset: %s", target_columns)

        # Log the columns of the DataFrame before dropping any columns
        logging.debug("DataFrame columns before dropping: %s", df.columns.tolist())

        # Define target first
        y = df[target_columns]
        
        # Drop columns one by one with a check to ensure they exist
        for col in target_columns:
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

        # Return the test - train split
        return X_train, y_train, X_test, y_test

    def build_neural_network(self, X_train, y_train, X_test, y_test):
        # Define the neural network model
        input_shape = (X_train.shape[1],)
        model = Sequential([
            tf.keras.Input(shape=input_shape),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(min(y_train.shape[1], len(self.future_topics)), activation='sigmoid')  # Multi-label classification
        ])
        logging.debug("Neural network defined")

        # Ensure creating a new optimizer each time
        if isinstance(self.optimizer, str):
            if self.optimizer_params is not None:
                if self.optimizer == 'adam':
                    optimizer = tf.keras.optimizers.Adam(**self.optimizer_params)
                elif self.optimizer == 'sgd':
                    optimizer = tf.keras.optimizers.SGD(**self.optimizer_params)
                else:
                    raise ValueError(f"Unsupported optimizer: {self.optimizer}")
            else:
                if self.optimizer == 'adam':
                    optimizer = tf.keras.optimizers.Adam()
                elif self.optimizer == 'sgd':
                    optimizer = tf.keras.optimizers.SGD()
                else:
                    raise ValueError(f"Unsupported optimizer: {self.optimizer}")
        elif isinstance(self.optimizer, tf.keras.optimizers.Optimizer):
            optimizer = self.optimizer.__class__.from_config(self.optimizer.get_config())
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")
        
        # Compile the model
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        logging.debug("Model compiled")
        
        # Ensure the target data has the correct shape
        if y_train.shape[1] != model.output_shape[1]:
            raise ValueError(f"Shape mismatch: target data shape {y_train.shape} does not match model output shape {model.output_shape}")

        # Train the model
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
        logging.info("Model trained")
        
        # Return fitted model
        return model

    def evaluate_model(self, model, X_test, y_test, verbose=0):
        if model is None:
            raise ValueError("Model is not defined. Call build_neural_network() first.")
        # Evaluate the model
        loss, accuracy = model.evaluate(X_test, y_test, verbose)
        logging.debug("Model evaluated")
        
        # Return performance metrics
        return loss, accuracy

    def get_feature_importance(self, model, X_test, y_test):
        # Check if model is trained
        if model is None:
            raise ValueError("Model is not defined. Call build_neural_network() first.")
        
        # Perform permutation importance
        perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring='r2')
        
        feature_importance = pd.DataFrame({
            "feature": X_test.columns,
            "importance_mean": perm_importance.importances_mean,
            "importance_std": perm_importance.importances_std
        })
        
        logging.debug("Feature importance computed")
        return feature_importance
    #don't forget to change it to cross_validate_model!
    def cross_validate(self, df, n_splits=5):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        val_losses = []
        val_accuracies = []

        for fold, (train_index, val_index) in enumerate(kf.split(df)):
            X_train, X_val = df.iloc[train_index].copy(), df.iloc[val_index].copy()

            # Log columns to ensure consistency
            logging.debug(f'Fold {fold} - Columns in X_train before processing: {X_train.columns}')
            logging.debug(f'Fold {fold} - Columns in X_val before processing: {X_val.columns}')

            # Determine valid target columns as in test_train_split
            valid_target_columns = [col for col in self.future_topics if col in X_train.columns]

            logging.debug(f'Fold {fold} - Valid target columns: {valid_target_columns}')

            if not valid_target_columns:
                logging.error(f"Fold {fold} - No valid target columns found in the DataFrame.")
                continue

            y_train = X_train[valid_target_columns]
            y_val = X_val[valid_target_columns]

            for col in valid_target_columns:
                if col in X_train.columns:
                    X_train = X_train.drop(columns=col)
                if col in X_val.columns:
                    X_val = X_val.drop(columns=col)

            logging.debug(f'Fold {fold} - Columns in X_train after dropping targets: {X_train.columns}')
            logging.debug(f'Fold {fold} - Columns in X_val after dropping targets: {X_val.columns}')

            # Convert y_train and y_val to numpy arrays
            y_train = y_train.values
            y_val = y_val.values

            if y_train.size == 0 or y_val.size == 0:
                logging.error(f"Fold {fold} - Target arrays are empty. Ensure target columns are correctly identified.")
                continue

            logging.debug(f'Fold {fold} - X_train shape: {X_train.shape}')
            logging.debug(f'Fold {fold} - y_train shape: {y_train.shape}')
            logging.debug(f'Fold {fold} - X_val shape: {X_val.shape}')
            logging.debug(f'Fold {fold} - y_val shape: {y_val.shape}')

            # Create a new neural network and optimizer for each fold
            try:
                model = self.build_neural_network(X_train, y_train, X_val, y_val)
            except Exception as e:
                logging.error(f"Fold {fold} - Error building neural network: {e}")
                continue
            
            # Evaluate the model's loss and accuracy
            try:
                loss, accuracy = self.evaluate_model(model, X_val, y_val, 0)
            except Exception as e:
                logging.error(f"Fold {fold} - Error evaluating model: {e}")
                continue

            val_losses.append(loss)
            val_accuracies.append(accuracy)
            logging.debug(f'Fold {fold} completed - loss: {loss}, accuracy: {accuracy}')
        
        if not val_losses or not val_accuracies:
            raise ValueError("Cross-validation failed. No valid target columns found in the DataFrame.")
        
        avg_loss = np.mean(val_losses)
        avg_accuracy = np.mean(val_accuracies)
        
        logging.info(f'Cross-validation results - Avg loss: {avg_loss}, Avg accuracy: {avg_accuracy}')
        
        return avg_loss, avg_accuracy



    
# Load the dataset
file_path = '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/Cleaned_Dataset.csv'
df = pd.read_csv(file_path)

# Print the first few rows and columns
df.head(), df.columns.tolist()

# Create an instance of the NeuralNetwork class
nn = NeuralNetwork(config)

# Perform cross-validation
avg_loss, avg_accuracy = nn.cross_validate(df)
avg_loss, avg_accuracy