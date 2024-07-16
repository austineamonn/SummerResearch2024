#from SummerResearch2024.association_mining.apriori import combine_lists

import numpy as np
import shap
import pandas as pd

# Load the SHAP values with allow_pickle=True
file_path = '/mnt/data/shap_values.npy'
shap_values = np.load(file_path, allow_pickle=True)

# Extract SHAP values from the Explanation objects
extracted_shap_values = np.array([[sv.values for sv in row] for row in shap_values])

# Calculate the mean absolute SHAP values for each feature
mean_abs_shap_values = np.mean(np.abs(extracted_shap_values), axis=0)

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({
    'Feature': [f'Feature {i+1}' for i in range(mean_abs_shap_values.shape[0])],
    'Importance': mean_abs_shap_values
})

# Sort the features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importance DataFrame
#tools.display_dataframe_to_user(name="Feature Importance", dataframe=feature_importance_df)

# Display the DataFrame
print(feature_importance_df)
