import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import shap
import os
import sys
import matplotlib.pyplot as plt

# Add the SummerResearch2024 directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import load_config

# Load configuration
config = load_config()

# Load the dataset
file_path = config["running_model"]["preprocessed data path"]
data = pd.read_csv(file_path)

# Separate features and targets
features = data.drop(columns=['career aspirations', 'future topics'])
target_career = data['career aspirations']
target_future = data['future topics']
feature_names = features.columns

# Impute missing values with the mean strategy
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

# Split the imputed data into training and testing sets for both targets
X_train_career, X_test_career, y_train_career, y_test_career = train_test_split(features_imputed, target_career, test_size=0.2, random_state=42)
X_train_future, X_test_future, y_train_future, y_test_future = train_test_split(features_imputed, target_future, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor for 'career aspirations'
model_career = RandomForestRegressor(random_state=42)
model_career.fit(X_train_career, y_train_career)

# Initialize and train the Random Forest Regressor for 'future topics'
model_future = RandomForestRegressor(random_state=42)
model_future.fit(X_train_future, y_train_future)

# Initialize SHAP explainer for both models
explainer_career = shap.Explainer(model_career, X_train_career)
shap_values_career = explainer_career(X_test_career)

explainer_future = shap.Explainer(model_future, X_train_future)
shap_values_future = explainer_future(X_test_future)

# Function to save SHAP summary plots with feature names and titles
def save_shap_summary_plot(shap_values, features, feature_names, target, filename):
    plt.figure(figsize=(10, 8))  # Increase the figure size
    shap.summary_plot(shap_values, features, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f'SHAP Summary Plot for {target}')
    plt.subplots_adjust(left=0.3, right=0.9, top=0.9)  # Adjust the layout to prevent cutting off feature names and title
    plt.savefig(filename)
    plt.close()

# Save the SHAP summary plots to PNG files with feature names and titles
save_shap_summary_plot(shap_values_career, X_test_career, feature_names, 'Career Aspirations', 'shap_summary_career_aspirations.png')
save_shap_summary_plot(shap_values_future, X_test_future, feature_names, 'Future Topics', 'shap_summary_future_topics.png')
