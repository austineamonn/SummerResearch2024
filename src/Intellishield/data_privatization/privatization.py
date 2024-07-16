import numpy as np
import pandas as pd
import logging
import sys
import os
import ast
from ast import literal_eval

# Add the SummerResearch2024 directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class Privatizer:
    def __init__(self, data, config=None, style=None, list_length=False):
        # Set up logging
        if config is not None:
            logging_level = config["logging"]["level"]
            logging.basicConfig(level=logging_level, format=config["logging"]["format"])
        else:
            logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        ## Privatized - Utility split
        if config is not None:
            self.Xp = config["privacy"]["Xp_list"]
            self.X = config["privacy"]["X_list"]
            self.Xu = config["privacy"]["Xu_list"]
            self.numerical_cols = config["privacy"]["numerical_columns"]
        else:
            self.Xp = [
                'first name','last name','ethnoracial group','gender',
                'international status','socioeconomic status'
            ]
            self.X = [
                'learning style', 'gpa', 'student semester' ,'major' ,
                'previous courses','course types','course subjects',
                'subjects of interest', 'extracurricular activities'
            ]
            self.Xu = [
                'career aspirations', 'future topics'
            ]
            self.numerical_cols = [
                "gpa", "student semester"
            ]

        # Privatization Method
        if style:
            self.style = style
        else:
            self.style = config["privacy"]["style"]
        self.list_length = list_length

        # Privatization Parameters
        if config is not None:
            self.epsilon = config["privacy"]["basic differential privacy"]["epsilon"]
            self.shuffle_ratio = config["privacy"]["shuffle"]["shuffle_ratio"]
        else:
            self.epsilon = 0.1
            self.shuffle_ratio = 0.1
        if self.style == "basic differential privacy":
            self.method = 'laplace'
        elif self.style == "uniform":
            self.method = 'uniform'
        else:
            self.method = config["privacy"]["basic differential privacy"]["method"]

        # Max Values (because all the mins are always 0)
        self.learning_style_max = 1
        self.gpa_max = 4.0
        self.semester_max = 14
        self.major_max = len(data.major()['majors_list'])
        self.course_names_max = len(data.course()['course_names_list'])
        self.course_types_max = len(data.course()['course_type_list'])
        self.course_subjects_max = len(data.course()['course_subject'])
        self.subjects_max = len(data.subjects()['subjects_list'])
        self.activities_max = len(data.extracurricular_activities()['complete_activity_list'])

        # Max list length values (because the mins are always 0)
        # Although these are hard coded in to reduce runtime, the 'longest_list_length' function can be used to get these values.
        self.major_max_len = 2
        self.course_names_max_len = 56
        self.course_types_max_len = 20
        self.course_subjects_max_len = 46
        self.subjects_max_len =  100
        self.activities_max_len = 37

    def get_max(self, col):
        # Get the max values depending on the column
        if col == 'learning style':
            max = self.learning_style_max
        elif col == 'gpa':
            max = self.gpa_max
        elif col == 'student semester':
            max = self.semester_max
        elif col == 'major':
            max = self.major_max
        elif col == 'previous courses':
            max = self.course_names_max
        elif col == 'course types':
            max = self.course_types_max
        elif col == 'course subjects':
            max = self.course_subjects_max
        elif col == 'subjects of interest':
            max = self.subjects_max
        elif col == 'extracurricular activities':
            max = self.activities_max
        else:
            logging.error("%s is an unknown column", col)

        return max

    def longest_list_length(self, df):
        """
        Input: pandas dataframe
        Output: dictionary of the maximum length of the list columns
        """
        # Function to find the length of the longest list in columns that have lists
        max_lengths = {}
        for column in df.columns:
            if isinstance(df[column].iloc[0], str) and df[column].iloc[0].startswith('['):
                # Convert string representation of lists to actual lists
                df[column] = df[column].apply(eval)
                max_lengths[column] = df[column].apply(len).max()
        return max_lengths
    
    def get_max_len(self, col):
        # Get the max list length values depending on the column
        if col == 'learning style':
            max = None
        elif col == 'gpa':
            max = None
        elif col == 'student semester':
            max = None
        elif col == 'major':
            max = self.major_max_len
        elif col == 'previous courses':
            max = self.course_names_max_len
        elif col == 'course types':
            max = self.course_types_max_len
        elif col == 'course subjects':
            max = self.course_subjects_max_len
        elif col == 'subjects of interest':
            max = self.subjects_max_len
        elif col == 'extracurricular activities':
            max = self.activities_max_len
        else:
            logging.error("%s is an unknown column", col)

        return max

    def modulus(self, value, mod):
        """
        Input: value to apply modulus to, modulus
        Output: changed value
        """
        if value > mod: # Subtract when value greater than mod
            while value > mod:
                value = value - mod
        elif value < 0: # Add when value under 0
            while value < 0:
                value = value + mod
        value = int(round(value))
        return value # Return final value between 0 and mod
    
    def gpa_modulus(self, value):
        """
        Input: value to apply modulus to
        Output: changed value
        """
        # For GPA 2.0 is subtracted or added because the minimum value is 2.0
        if value > 4.0: # Subtract when value greater than 4.0
            while value > 4.0:
                value = value - 2.0
        elif value < 2.0: # Add when value under 2.0
            while value < 2.0:
                value = value + 2.0
        value = round(value, 2)
        return value # Return final value between 2.0 and 4.0
    
    def adjust_list_length(self, l, target_length, mid_val, max_len):
        # Ensure target length is within the list length range (0 to max_len)
        target_length = self.modulus(target_length, max_len)

        # Adjust length of list
        current_length = len(l)
        if target_length < current_length:
            return l[:target_length]
        elif target_length > current_length:
            # Any extra list elements become the median value of the column
            return l + [mid_val] * (target_length - current_length)
        else:
            return l
    
    def calculate_sensitivity(self, col, max, max_length):
        """
        Input: Name of the column, maximum value of the column
        Output: Sensitivity level of the column
        """
        # Calculate the column length
        col_len = len(col)

        # Calculate the mean sensitivity
        sensitivity = max / col_len # Range is just the max because the minimum is always 0.
        if max_length is not None:
            len_sensitivity = max_length / col_len
        else:
            len_sensitivity = None
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("Sensitivity and list length sensitivity calculated for %s", col)
        return sensitivity, len_sensitivity
    
    def basic_differential_privacy(self, df, epsilon, method='laplace', list_length=False):
        def add_laplace_noise(value, scale, max_val):
            premod = value + np.random.laplace(0, scale)
            if max_val == 4.0: # A max value of 4.0 only occurs for GPA so it is used to pull out GPA which is not an integer like all the other columns
                final_val = self.gpa_modulus(premod)
            else:
                final_val = self.modulus(premod, max_val)
            return final_val

        def add_laplace_noise_to_list(lst, scale, max_val):
            return [add_laplace_noise(x, scale, max_val) for x in lst]
        
        def add_uniform_noise(value, low, high, max_val):
            premod = value + np.random.uniform(low, high)
            if max_val == 4.0: # A max value of 4.0 only occurs for GPA so it is used to pull out GPA which is not an integer like all the other columns
                final_val = self.gpa_modulus(premod)
            else:
                final_val = self.modulus(premod, max_val)
            return final_val
        
        def add_uniform_noise_to_list(lst, low, high, max_val):
            return [add_uniform_noise(x, low, high, max_val) for x in lst]

        noisy_df = df.copy()
        for column in noisy_df.columns:
            max_val = self.get_max(column)
            max_len = self.get_max_len(column)
            sensitivity, len_sensitivity = self.calculate_sensitivity(column, max_val, max_len)
            scale = sensitivity/epsilon
            if noisy_df[column].dtype == 'object':
                if method == 'laplace':
                    if list_length and column != 'learning style':
                        len_scale = len_sensitivity/epsilon
                        target_lengths = np.random.laplace(0, len_scale, size=len(df)).astype(int)
                        mid_val = max_val / 2
                        # Adjust the lists
                        noisy_df[column] = [self.adjust_list_length(lst, length, mid_val, max_len) for lst, length in zip(noisy_df[column], target_lengths)]
                    noisy_df[column] = noisy_df[column].apply(lambda x: add_laplace_noise_to_list(x, scale, max_val))
                if method == 'uniform':
                    # Calculate noise range
                    high = sensitivity / epsilon
                    low = -high
                    if list_length and column != 'learning style':
                        # Calculate noise range
                        len_high = sensitivity / epsilon
                        len_low = -len_high
                        target_lengths = np.random.uniform(len_low, len_high, size=len(df)).astype(int)
                        mid_val = max_val / 2
                        # Adjust the lists
                        noisy_df[column] = [self.adjust_list_length(lst, length, mid_val, max_len) for lst, length in zip(noisy_df[column], target_lengths)]
                    noisy_df[column] = noisy_df[column].apply(lambda x: add_uniform_noise_to_list(x, low, high, max_val))
            else:
                if method == 'laplace':
                    noisy_df[column] = noisy_df[column].apply(lambda x: add_laplace_noise(x, scale, max_val))
                elif method == 'uniform':
                    noisy_df[column] = noisy_df[column].apply(lambda x: add_uniform_noise(x, low, high, max_val))

        return noisy_df

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

    def privatize_dataset(self, df):
        """
        Input: preprocessed dataset
        Output: privatized dataset
        """

        df_X = df.loc[:, self.X]

        # Privatize the X columns
        if self.style == 'basic differential privacy' or self.style == 'uniform':
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                parameter_dict = {"epsilon": self.epsilon, "method": self.method, "list_length": self.list_length}
                logging.debug(f"Calculating basic differential privacy with the following parameters: {parameter_dict}")
            df_X = self.basic_differential_privacy(df_X, self.epsilon, self.method, self.list_length)
        elif self.style == 'shuffle':
            # The number to shuffle depends on the number of rows
            # and the ratio of rows shuffled
            num_shuffle = round(df.shape[0] * self.shuffle_ratio)
            df_X = self.random_shuffle(df, self.X, num_shuffle)
        elif self.style == 'full shuffle':
            # Completely shuffles all values
            num_shuffle = df.shape[0]
            df_X = self.random_shuffle(df, self.X, num_shuffle)

        return df_X

# Main execution
if __name__ == "__main__":
    # Import necessary dependencies
    from datafiles_for_data_construction.data import Data
    from config import load_config

    # Load configuration and data
    data = Data()
    config = load_config()

    # Import preprocessed (but not dimensionality reduced) dataset being sure to read the list columns as lists not as strings
    df = pd.read_csv(config["running_model"]["preprocessed data path"], converters={'learning style': literal_eval, 'major': literal_eval, 'previous courses': literal_eval, 'course types': literal_eval, 'course subjects': literal_eval, 'subjects of interest': literal_eval, 'extracurricular activities': literal_eval, 'career aspirations': literal_eval, 'future topics': literal_eval})
    
    # Basic Differential Privacy
    privatizer = Privatizer(data, config, 'basic differential privacy')
    bdp_df = privatizer.privatize_dataset(df)
    bdp_df.to_csv(config["running_model"]["basic differential privacy privatized data path"], index=False)

    # Basic Differential Privacy with changing list lengths
    privatizer = Privatizer(data, config, 'basic differential privacy', True)
    bdp_df = privatizer.privatize_dataset(df)
    bdp_df.to_csv(config["running_model"]["basic differential privacy LLC privatized data path"], index=False)

    logging.info("Basic Differential Privacy completed")

    # Uniform Noise Addition
    privatizer = Privatizer(data, config, 'uniform')
    un_df = privatizer.privatize_dataset(df)
    un_df.to_csv(config["running_model"]["uniform noise privatized data path"], index=False)

    # Uniform Noise Addition with changing list lengths
    privatizer = Privatizer(data, config, 'uniform', True)
    un_df = privatizer.privatize_dataset(df)
    un_df.to_csv(config["running_model"]["uniform noise LLC privatized data path"], index=False)

    logging.info("Uniform Noise Privacy completed")
    
    # Random Shuffling
    privatizer = Privatizer(data, config, 'shuffle')
    rs_df = privatizer.privatize_dataset(df)
    rs_df.to_csv(config["running_model"]["random shuffling privatized data path"], index=False)

    logging.info("Random Shuffling Privacy completed")

    # Complete Shuffling
    privatizer = Privatizer(data, config, 'full shuffle')
    fs_df = privatizer.privatize_dataset(df)
    fs_df.to_csv(config["running_model"]["complete shuffling privatized data path"], index=False)

    logging.info("Complete Shuffling Privacy completed")
