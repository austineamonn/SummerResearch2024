import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import ast
import logging

from config import load_config

# Load configuration
config = load_config()

# Set up logging
logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])

class Privatizer:
    def __init__(self, config):
        self.config = config

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
        self.p = config["privacy"]["randomized"]["p"]
        self.shuffle_ratio = config["privacy"]["shuffle"]["shuffle_ratio"]

        # Privatization Method and its associated parameters
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
        noised_data = self.modulus(noised_data, data.max())

        logging.debug("Laplacian noise added")
        return noised_data
    
    def calculate_sensitivity(self, df, col, type):
        """
        Input: dataset, column to measure sensitivity, type of sensitivity (mean vs sum)
        Output: sensitivity of the column
        """
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
        noised_data = self.modulus(noised_data, data.max())
        return noised_data
    
    def add_randomized_response(self, data, p=0.1):
        """
        Input: data column(s) to randomly flip, probability of flipping a value
        Output: data column(s) with random flips
        """
        flip = np.random.choice([0, 1], size=data.shape, p=[1 - p, p])
        noised_data = np.abs(data - flip)  # Flipping the values
        return noised_data

    def random_shuffle(self, df, cols, num_shuffle):
        """
        Input: dataset, column(s) to shuffle, the number or elements to shuffle
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
        elif type == 'randomized':
            df[col] = self.add_randomized_response(df[col], self.p)
        elif type == 'shuffle':
            # The number to shuffle depends on the number of rows
            # and the ratio of rows shuffled
            num_shuffle = round(df.shape[0] * self.shuffle_ratio)
            df = self.random_shuffle(df, col, num_shuffle)

        return df

    def privatize_dataset(self, df):
        """
        Input: preprocessed dataset
        Output: privatized dataset
        """
        # Find the utility columns
        suffixes = ['_career aspirations', '_future topics']

        # Find columns that end with any of the specified suffixes
        utility_cols = [col for col in df.columns if col.endswith(tuple(suffixes))]

        df_X = df.drop(columns=utility_cols)

        # Privatize the X columns
        if self.style == 'shuffle':
            df = self.privatization_style(df, utility_cols, self.style)
        else:
            for col in df_X:
                # Numerical columns
                if col in self.numerical_cols:
                    logging.debug(f"{col} is numerical")
                    # Privatize the column
                    # note that since numerical columns cannot use randomized
                    # response the type gets set to basic differential privacy instead
                    if self.style == 'randomized':
                        df = self.privatization_style(df, col, 'basic differential privacy')
                    else:
                        df = self.privatization_style(df, col, self.style)
                    logging.debug(f"{col} was privatized using {self.style}")
                # Nonnumerical columns
                else:
                    logging.debug(f"{col} is not numerical")
                    # Privatize the column
                    df = self.privatization_style(df, col, self.style)
                    logging.debug(f"{col} was privatized using {self.style}")

        return df
 