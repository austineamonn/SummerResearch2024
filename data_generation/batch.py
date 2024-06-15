import pandas as pd
import time
import sys
import os
from SummerResearch2024.data_generation.data_generation import DataGenerator

# Add the SummerResearch2024 directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import load_config

# Load configuration
config = load_config()

def process_batch(num_samples, generator):
    # Generate synthetic data for the batch
    df = generator.generate_synthetic_dataset(num_samples)
    return df

def main():
    total_samples = 100000  # Total number of samples to generate
    batch_size = 1000
    processed_results = []
    generator = DataGenerator(config)

    for i in range(0, total_samples, batch_size):
        num_samples = min(batch_size, total_samples - i)
        processed_batch = process_batch(num_samples, generator)
        processed_results.append(processed_batch)
        print(f"Processed batch {i // batch_size + 1}/{(total_samples // batch_size) + 1}")
        time.sleep(0.1)  # Add a short delay to avoid overloading the system

    # Combine all the processed batches into a single DataFrame
    final_df = pd.concat(processed_results, ignore_index=True)

    # Output the final DataFrame
    final_df.to_csv(config["running_model"]["data path 2"], index=False)

if __name__ == "__main__":
    main()
