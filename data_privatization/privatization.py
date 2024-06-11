import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
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
        self.sensitivity = config["privacy"]["laplace"]["sensitivity"]
        self.epsilon = config["privacy"]["laplace"]["epsilon"]
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
    
    def add_laplacian_noise(self, data, type, sensitivity, epsilon=1):
        """
        Input: data column(s) to add noise to, sensitivity (mean or sum), epsilon
        Output: data column(s) with noise addition
        """
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale, size=data.shape)
        noised_data = data + noise
        # Only clip and round nonnumerical data
        if type == 'nonnumerical':
            # Clip values to be within the binary range
            noised_data = np.clip(noised_data, 0, 1)
            # Round values to 0 or 1
            noised_data = noised_data.round()
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
    
    def add_uniform_noise(self, data, type, low=-0.5, high=0.5):
        """
        Input: data column(s) to add noise to, lowest noise loss, highest noise addition
        Output: data column(s) with noise addition
        """
        noise = np.random.uniform(low, high, size=data.shape)
        noised_data = data + noise
        # Only clip and round nonnumerical data
        if type == 'nonnumerical':
            # Clip values to be within the binary range
            noised_data = np.clip(noised_data, 0, 1)
            # Round values to 0 or 1
            noised_data = noised_data.round()
        return noised_data
    
    def add_randomized_response(self, data, p=0.1):
        """
        Input: data column(s) to randomly flip, probability of flipping a value
        Output: data column(s) with random flips
        """
        flip = np.random.choice([0, 1], size=data.shape, p=[1 - p, p])
        noised_data = np.abs(data - flip)  # Flipping the values
        return noised_data

    def random_shuffle(self, df, col, num_shuffle):
        """
        Input: dataset, column to shuffle, the number or elements to shuffle
        Output: shuffled dataset
        """
        # Randomly pick the indices to shuffle
        indices = np.random.choice(df.index, num_shuffle, replace=False)

        # Get the elements to shuffle
        elements_to_shuffle = df.loc[indices, col].values

        # Shuffle the elements
        np.random.shuffle(elements_to_shuffle)

        # Send shuffled elements back to their original position
        df.loc[indices, col] = elements_to_shuffle

        return df

    def transform_target_variable(self, df, column, suffix=''):
        """
        Input: dataset, column to transform, suffix to add to differentiate column names
        Outputs: dataset with binarized column, binary label names
        """

        # Initialize multilabel binarizer
        mlb = MultiLabelBinarizer()

        # If the input is a string convert it to a list
        def parse_list(x):
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                # Return an empty list if this is not possible
                except (ValueError, SyntaxError):
                    return []
            return x

        # Apply parse_list to the columns
        df[column] = df[column].apply(parse_list)
        df[column] = df[column].apply(lambda x: x if isinstance(x, list) else [])

        # Transform the column using the MLB
        binary_labels = mlb.fit_transform(df[column])

        #Create a dataframe with the binarized feature
        new_col_names = [col + suffix for col in mlb.classes_]
        binary_df = pd.DataFrame(binary_labels, columns=new_col_names)

        # Drop the original column and return the altered dataset
        return df.drop(column, axis=1).join(binary_df), new_col_names
    
    def privatization_style(self, df, col, data_type, type):
        """
        Input: dataset, column(s) to be privatized, column datatype
        (numerical or nonnumerical), privatization type
        Output: dataset with column(s) privatized
        """

        if type == 'laplace':
            sensitivity = self.calculate_sensitivity(df, col, self.sensitivity)
            df[col] = self.add_laplacian_noise(df[col], data_type, sensitivity, self.epsilon)
        elif type == 'uniform':
            df[col] = self.add_uniform_noise(df[col], data_type, self.low, self.high)
        elif type == 'randomized':
            df[col] = self.add_randomized_response(df[col], self.p)
        elif type == 'shuffle':
            num_shuffle = round(len(df[col]) * self.shuffle_ratio)
            df = self.random_shuffle(df, col, num_shuffle)

        return df

    def privatize_dataset(self, df):
        """
        Input: dataset
        Output: privatized dataset
        """

        # Drop the Xp columns
        df = df.drop(columns=self.Xp)
        logging.debug(f"{self.Xp} were dropped")

        # Privatize the X columns
        for col in self.X:
            suffix = '_' + col
            # Only privatize the numerical columns
            if col in self.numerical_cols:
                logging.debug(f"{col} is numerical")
                # Normalize the column
                df = self.normalize_features(df, col, self.normalize_type)
                # Privatize the column
                # note that since numerical columns cannot use randomized
                # response the type gets set to laplace instead
                if self.style == 'randomized':
                    df = self.privatization_style(df, col, 'numerical', 'laplace')
                else:
                    df = self.privatization_style(df, col, 'numerical', self.style)
                logging.debug(f"{col} was privatized using {self.style}")
            # Privatize and binarize the nonnumerical columns
            else:
                logging.debug(f"{col} is not numerical")
                if self.style == 'shuffle':
                    # Privatize the column
                    df = self.privatization_style(df, col, 'nonnumerical', self.style)
                    logging.debug(f"{col} was privatized using {self.style}")
                    # Binarize the column
                    output = self.transform_target_variable(df, col, suffix)
                    logging.debug(f"{col} was binarized")
                    df = output[0]
                else:
                    # Binarize the column
                    df, new_cols = self.transform_target_variable(df, col, suffix)
                    logging.debug(f"{col} was binarized")
                    # Privatize the column
                    df = self.privatization_style(df, new_cols, 'nonnumerical', self.style)
                    logging.debug(f"{col} was privatized using {self.style}")

        # Binarize the Xu columns
        for col in self.Xu:
            output = self.transform_target_variable(df, col)
            df = output[0]
            logging.debug(f"{col} was binarized")

        return df
    
    def clean_dataset(self, df):
        """
        Input: dataset
        Output: privatized dataset
        """

        # Drop the Xp columns
        df = df.drop(columns=self.Xp)
        logging.debug(f"{self.Xp} were dropped")

        # Privatize the X columns
        for col in self.X:
            suffix = '_' + col
            # Only privatize the numerical columns
            if col in self.numerical_cols:
                logging.debug(f"{col} is numerical")
                # Normalize the column
                df = self.normalize_features(df, col, self.normalize_type)
            # Privatize and binarize the nonnumerical columns
            else:
                logging.debug(f"{col} is not numerical")
                # Binarize the column
                output = self.transform_target_variable(df, col, suffix)
                logging.debug(f"{col} was binarized")
                df = output[0]

        # Binarize the Xu columns
        for col in self.Xu:
            output = self.transform_target_variable(df, col)
            df = output[0]
            logging.debug(f"{col} was binarized")

        return df