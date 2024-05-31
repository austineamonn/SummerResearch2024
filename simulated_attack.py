import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from config import load_config

# Load configuration
config = load_config()

# Set up logging
logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])

class SimulatedAttacks:
    def __init__(self, file_path):
        self.file_path = file_path
        self.anonymized_df = pd.read_csv(file_path)
        self.external_data = self.anonymized_df.sample(frac=0.1, random_state=1)  # 10% sample as external data

    def re_identification_attack(self, quasi_identifiers):
        """
        Perform re-identification attack using quasi-identifiers.
        """
        logging.info("Starting re-identification attack...")
        matches = pd.merge(self.anonymized_df, self.external_data, on=quasi_identifiers, how='inner')
        re_identification_rate = len(matches) / len(self.external_data) * 100
        logging.info("Re-identification rate: %.2f%%", re_identification_rate)
        return re_identification_rate

    def membership_inference_attack(self, quasi_identifiers, target_column='student class year'):
        """
        Simulate membership inference attack.
        """
        logging.info("Starting membership inference attack...")

        # Prepare the dataset for a simple classification task
        X = self.anonymized_df.drop(columns=[target_column])
        y = self.anonymized_df[target_column]

        # Convert categorical features to numerical (one-hot encoding)
        X = pd.get_dummies(X)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        # Train a simple classifier
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        logging.info("Model accuracy: %.2f%%", accuracy * 100)

        # Perform membership inference attack
        inferred_memberships, threshold = self._calculate_memberships(model, X_train, X_test)
        inferred_membership_rate = np.mean(inferred_memberships) * 100

        logging.info("Inferred membership rate: %.2f%%", inferred_membership_rate)
        logging.info("Confidence threshold: %.2f", threshold)
        return inferred_membership_rate

    def _calculate_memberships(self, model, X_train, X_test, threshold_percentile=90):
        """
        Helper function to calculate memberships based on model's confidence scores.
        """
        train_scores = model.predict_proba(X_train)
        test_scores = model.predict_proba(X_test)
        
        train_confidences = np.max(train_scores, axis=1)
        threshold = np.percentile(train_confidences, threshold_percentile)
        
        test_confidences = np.max(test_scores, axis=1)
        inferred_memberships = (test_confidences >= threshold).astype(int)
        
        return inferred_memberships, threshold

# Usage example
if __name__ == "__main__":
    attack_simulator = SimulatedAttacks('/path/to/Privatized_Dataset.csv')
    
    # Define quasi-identifiers
    quasi_identifiers = ['race_ethnicity', 'gender', 'international']
    
    # Perform re-identification attack
    re_identification_rate = attack_simulator.re_identification_attack(quasi_identifiers)
    
    # Perform membership inference attack
    inferred_membership_rate = attack_simulator.membership_inference_attack(quasi_identifiers)
