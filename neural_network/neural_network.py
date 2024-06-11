import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.utils import resample
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from kerastuner.tuners import RandomSearch
from config import load_config
from datafiles_for_data_construction.data import Data

# Load configuration
config = load_config()

# Set up logging
logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])

class NeuralNetwork:
    def __init__(self, config):
        self.config = config
        data = Data()
        self.future_topics = data.future_topics()["future_topics"]
        self.optimizer = config["neural_network"]["optimizer"]
        self.optimizer_params = config["neural_network"]["optimizer_params"]
        self.loss = config["neural_network"]["loss"]
        self.metrics = config["neural_network"]["metrics"]
        self.best_hps = None

    def test_train_split(self, df, n_components=100):
        # Ensure the target is defined using existing columns
        target_columns = [col for col in self.future_topics if col in df.columns]
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

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logging.debug("Features standardized")

        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        logging.debug("PCA applied to features")

        # Convert y to a numpy array
        y = y.values
        logging.debug("Target converted to a numpy array")

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
        logging.info("Test-train split completed")

        # Print shapes and types for debugging
        logging.debug(f'X_train shape: {X_train.shape}, X_train type: {type(X_train)}')
        logging.debug(f'y_train shape: {y_train.shape}, y_train type: {type(y_train)}')
        logging.debug(f'X_test shape: {X_test.shape}, X_test type: {type(X_test)}')
        logging.debug(f'y_test shape: {y_test.shape}, y_test type: {type(y_test)}')

        # Return the test - train split
        return X_train, y_train, X_test, y_test

    def build_neural_network(self, X_train, y_train, X_test, y_test, hp=None):
        # Define the neural network model
        input_shape = (X_train.shape[1],)
        
        # Use hyperparameters if provided, otherwise use fixed values
        units1 = hp.Int('units1', min_value=128, max_value=2048, step=64) if hp else 512
        dropout1 = hp.Float('dropout1', min_value=0.0, max_value=0.7, step=0.05) if hp else 0.3
        units2 = hp.Int('units2', min_value=128, max_value=2048, step=64) if hp else 256
        dropout2 = hp.Float('dropout2', min_value=0.0, max_value=0.7, step=0.05) if hp else 0.3
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 5e-4]) if hp else 1e-3

        model = Sequential([
            tf.keras.Input(shape=input_shape),
            Dense(units1, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(dropout1),
            Dense(units2, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(dropout2),
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

        # Use hyperparameter-tuned learning rate if available
        if learning_rate is not None:
            optimizer.learning_rate = learning_rate

        # Compile the model
        model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)
        
        # Callbacks for early stopping and model checkpointing (saving the best performing model)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')

        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                            epochs=config["neural_network"]["epochs"], 
                            batch_size=config["neural_network"]["batch_size"], 
                            callbacks=[early_stopping, model_checkpoint])
        logging.info("Model training complete")

        return model, history
    
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
                model = self.build_neural_network(X_train, y_train, X_val, y_val, self.best_hps)
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

    def tune_hyperparameters(self, x_train, y_train, x_test, y_test):
        tuner = RandomSearch(
            lambda hp: self.build_neural_network(x_train, y_train, x_test, y_test, hp),
            objective='val_loss',
            max_trials=20,  # Increased from 5 to 20
            executions_per_trial=3,
            directory=self.config["running_model"]["directory"],
            project_name='Hyperparameter_Tuning_of_NN'
        )

        tuner.search_space_summary()

        tuner.search(x_train, y_train, epochs=20, validation_split=0.2)  # Increased epochs from 5 to 20

        tuner.results_summary()

        self.best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        model = self.build_neural_network(x_train, y_train, x_test, y_test, self.best_hps)

        logging.info(f"The hyperparameter search is complete. The optimal parameters are: units1={self.best_hps.get('units1')}, dropout1={self.best_hps.get('dropout1')}, units2={self.best_hps.get('units2')}, dropout2={self.best_hps.get('dropout2')}, and learning_rate={self.best_hps.get('learning_rate')}.")

        return model
