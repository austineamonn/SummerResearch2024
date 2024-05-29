from data_generation import generate_synthetic_dataset
from basic_privatization import privatizing_dataset
from privacy_metrics import calculate_privacy_metrics

# Generate the dataset
synthetic_dataset = generate_synthetic_dataset()

# Print the first few rows of the original dataset
print(synthetic_dataset.head())

# Save the original dataset to CSV
synthetic_dataset.to_csv('Dataset.csv', index=False)

# Privatize the dataset
private_dataset, parameters = privatizing_dataset(synthetic_dataset)

# Save the privatized dataset as a csv
private_dataset.to_csv('Privatized_Dataset.csv', index=False)

# Calculate privacy metrics
privacy_metrics = calculate_privacy_metrics(synthetic_dataset, private_dataset,parameters)
print(privacy_metrics)
