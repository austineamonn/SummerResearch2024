from data_generation import generate_synthetic_dataset

# Generate the dataset
synthetic_dataset = generate_synthetic_dataset()

# Print the first few rows of the dataset
print(synthetic_dataset.head())

# Save to CSV
synthetic_dataset.to_csv('Dataset.csv', index=False)