import numpy as np
import pandas as pd
import logging
import sys
import os

# Add the SummerResearch2024 directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class Privatizer:
    def __init__(self, config, style=None):
        # Set up logging
        logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])

        # Privatized - Utility split
        self.Xp = config["privacy"]["Xp_list"]
        self.X = config["privacy"]["X_list"]
        self.Xu = config["privacy"]["Xu_list"]
        self.numerical_cols = config["privacy"]["numerical_columns"]

        # Normalization Parameters
        self.normalize_type = config["privacy"]["normalize_type"]

        # Privatization Parameters
        self.sensitivity = config["privacy"]["basic differential privacy"]["sensitivity"]
        self.epsilon = config["privacy"]["basic differential privacy"]["epsilon"]
        self.low = config["privacy"]["uniform"]["low"]
        self.high = config["privacy"]["uniform"]["high"]
        self.shuffle_ratio = config["privacy"]["shuffle"]["shuffle_ratio"]

        # Privatization Method
        if style:
            self.style = style
        else:
            self.style = config["privacy"]["style"]
        
    def normalize_features(self, df, col, type='Zscore'):
        """
        Input: dataset, column to normalize, normalization type
        Output: dataset with the column normalized
        """
        if type == 'min_max':
            # Normalizing column using Min-Max normalization
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        elif type == 'Zscore':
            # Normalizing column using Z-score normalization
            df[col] = (df[col] - df[col].mean()) / df[col].std()
        else:
            logging.error("No normalization type chosen")
        return df
    
    def modulus(self, col, mod):
        """
        Input: columns to apply modulus to, modulus
        Output: changed colums
        """

        for i in range(len(col)):
            while col[i] > mod:
                col[i] = col[i] - mod

        return col

    def basic_differential_privacy(self, data, sensitivity, epsilon=1):
        """
        Input: data column(s) to add noise to, sensitivity (mean or sum), epsilon
        Output: data column(s) with noise addition
        """
        # Add noise based on sensitivity and epsilon
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale, size=data.shape)
        noised_data = data + noise

        # Use modulus to keep data insude
        noised_data = noised_data % data.max()
        #noised_data = self.modulus(noised_data, data.max())

        logging.debug("Laplacian noise added")
        return noised_data
    
    def calculate_sensitivity(self, df, col, type):
        """
        Input: dataset, column to measure sensitivity, type of sensitivity (mean vs sum)
        Output: sensitivity of the column
        """
        # Check if the column contains numerical data
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' contains non-numerical data.")
    
        if type == 'mean':
            # Calculate the sensitivity for the mean query
            # better for average distribution of labels
            range_value = df[col].max() - df[col].min()
            sensitivity = range_value / len(df[col])
            logging.debug("Mean sensitivity calculated for %s", col)
        elif type == 'sum':
            # Calculate the sensitivity for the sum query
            # reflects the count of labels
            max_value = df[col].max()
            sensitivity = max_value  # For sum, sensitivity is the maximum value an individual data point can take
            logging.debug("Sum sensitivity calculated for %s", col)
        else:
            logging.error("No sensitivity type chosen for %s", col)
        
        return sensitivity
    
    def add_uniform_noise(self, data, low=-0.5, high=0.5):
        """
        Input: data column(s) to add noise to, lowest noise loss, highest noise addition
        Output: data column(s) with noise addition
        """
        # Add noised
        noise = np.random.uniform(low, high, size=data.shape)
        noised_data = data + noise

        # Use modulus to keep data insude
        noised_data = noised_data % data.max()
        #noised_data = self.modulus(noised_data, data.max())
        return noised_data

    def random_shuffle(self, df, cols, num_shuffle):
        """
        Input: dataset, name of the column(s) to shuffle, the number or elements to shuffle
        Output: shuffled dataset
        """
        # Randomly pick the indices to shuffle
        indices = np.random.choice(df.index, num_shuffle, replace=False)

        # Get a copy of the DataFrame subset to shuffle
        sub_df = df.loc[indices, cols].copy()

        # Shuffle the DataFrame subset rows
        shuffled_sub_df = sub_df.sample(frac=1).reset_index(drop=True)

        # Assign the shuffled rows back to the original DataFrame positions
        df.loc[indices, cols] = shuffled_sub_df.values

        return df
    
    def privatization_style(self, df, col, type):
        """
        Input: dataset, column(s) to be privatized, privatization type
        Output: dataset with column(s) privatized
        """

        if type == 'basic differential privacy':
            sensitivity = self.calculate_sensitivity(df, col, self.sensitivity)
            df[col] = self.basic_differential_privacy(df[col], sensitivity, self.epsilon)
        elif type == 'uniform':
            df[col] = self.add_uniform_noise(df[col], self.low, self.high)
        elif type == 'shuffle':
            # The number to shuffle depends on the number of rows
            # and the ratio of rows shuffled
            num_shuffle = round(df.shape[0] * self.shuffle_ratio)
            df = self.random_shuffle(df, col, num_shuffle)
        elif type == 'full shuffle':
            # Completely shuffles all values
            num_shuffle = df.shape[0]
            df = self.random_shuffle(df, col, num_shuffle)

        return df

    def privatize_dataset(self, df):
        """
        Input: preprocessed dataset
        Output: privatized dataset
        """

        df_X = df.loc[:, self.X]

        # Privatize the X columns
        if self.style == 'shuffle':
            # Shuffle the X columns
            df = self.privatization_style(df, self.X, self.style)
        else:
            for col in df_X:
                df = self.privatization_style(df, col, self.style)
                logging.debug(f"{col} was privatized using {self.style}")

        return df


# Main execution
if __name__ == "__main__":
    # Import necessary dependencies
    from config import load_config

    # Load configuration and data
    config = load_config()

    # Import synthetic dataset and preprocess it
    df = pd.read_csv(config["running_model"]["preprocessed with RNN data path"])

    # Basic Differential Privacy
    privatizer = Privatizer(config, 'basic differential privacy')
    bdp_df = privatizer.privatize_dataset(df)
    bdp_df.to_csv(config["running_model"]["basic differential privacy privatized data path"], index=False)

    logging.info("Basic Differential Privacy Run")

    # Uniform Noise Addition
    privatizer = Privatizer(config, 'uniform')
    un_df = privatizer.privatize_dataset(df)
    un_df.to_csv(config["running_model"]["uniform noise privatized data path"], index=False)

    logging.info("Uniform Noise Privacy Run")

    # Random Shuffling
    privatizer = Privatizer(config, 'shuffle')
    rs_df = privatizer.privatize_dataset(df)
    rs_df.to_csv(config["running_model"]["random shuffling privatized data path"], index=False)

    logging.info("Random Shuffling Privacy Run")

    # Complete Shuffling
    privatizer = Privatizer(config, 'full shuffle')
    fs_df = privatizer.privatize_dataset(df)
    fs_df.to_csv(config["running_model"]["complete shuffling privatized data path"], index=False)

    logging.info("Random Shuffling Privacy Run")
