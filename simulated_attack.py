import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the anonymized dataset
anonymized_df = pd.read_csv('/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/Privatized_Dataset.csv')

# Assume the adversary has access to a subset of the original data
# Create a subset of the original data as external data
external_data = anonymized_df.sample(frac=0.1, random_state=1)  # 10% sample as external data

# Re-identification Attack
def re_identification_attack(anonymized_df, external_data, quasi_identifiers):
    # Match records based on quasi-identifiers
    matches = pd.merge(anonymized_df, external_data, on=quasi_identifiers, how='inner')
    return matches

# Define quasi-identifiers
quasi_identifiers = ['race_ethnicity', 'gender', 'international']

# Perform re-identification attack
matches = re_identification_attack(anonymized_df, external_data, quasi_identifiers)
re_identification_rate = len(matches) / len(external_data) * 100

print(f"Re-identification rate: {re_identification_rate:.2f}%")

# Membership Inference Attack

# Prepare the dataset for a simple classification task
# For simplicity, we'll use 'student class year' as the target and other fields as features
X = anonymized_df.drop(columns=['student class year'])
y = anonymized_df['student class year']

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
print(f"Model accuracy: {accuracy:.2f}%")

# Simulate membership inference attack
# We'll use the model's confidence scores to infer membership status
def membership_inference_attack(model, X_train, X_test, threshold_percentile=90):
    # Get the model's confidence scores
    train_scores = model.predict_proba(X_train)
    test_scores = model.predict_proba(X_test)
    
    # Calculate the confidence threshold based on the training data
    train_confidences = np.max(train_scores, axis=1)
    threshold = np.percentile(train_confidences, threshold_percentile)
    
    # Apply the threshold to the test set
    test_confidences = np.max(test_scores, axis=1)
    inferred_memberships = (test_confidences >= threshold).astype(int)
    
    return inferred_memberships, threshold

# Perform membership inference attack
inferred_memberships, threshold = membership_inference_attack(model, X_train, X_test)
inferred_membership_rate = np.mean(inferred_memberships) * 100

print(f"Inferred membership rate: {inferred_membership_rate:.2f}%")
print(f"Confidence threshold: {threshold:.2f}")
