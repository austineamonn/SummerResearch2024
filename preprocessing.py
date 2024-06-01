import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from config import load_config
import ast

# Load configuration
config = load_config()

# Set up logging
logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])

class PreProcessing:
    def __init__(self, config):
        self.config = config
        self.numerical_columns = self.config["preprocessing"]["numerical_columns"]
        logging.debug("initialization complete.")
    
    def preprocessor(self, df):
        """
        Main function that takes in the privatized dataset and preprocesses it
        """
        # One-Hot Encoding
        df = self.one_hot_encode(df, 'learning_style')
        df = self.one_hot_encode(df, 'previous courses')
        df = self.one_hot_encode(df, 'course type')
        df = self.one_hot_encode(df, 'subjects in courses')
        logging.info("One-hot encoding complete")

        # TF-IDF Vectorize text columns
        df = self.tfidf_vectorize(df, 'subjects of interest')
        df = self.tfidf_vectorize(df, 'career aspirations')
        df = self.tfidf_vectorize(df, 'extracurricular activities')
        logging.info("TF-IFD vectorization complete")

        # Normalize numerical columns
        df = self.normalize_numerical_features(df, self.numerical_columns)
        logging.info("Normalization complete")

        # Transform the 'future topics' column
        df = self.transform_target_variable(df, 'future topics')
        logging.info("Future topics converted into a multi-label binary format")

        # Return the preprocessed dataset
        return df
    
    def list_to_string(self, lst):
        """
        A function to convert a list to a string
        """
        return " ".join(lst)

    def one_hot_encode(self, df, column):
        """
        One-Hot Encodes a specified column in the dataframe.
        
        Parameters:
        df (pd.DataFrame): The input dataframe.
        column (str): The name of the column to one-hot encode.
        
        Returns:
        pd.DataFrame: The dataframe with the specified column one-hot encoded.
        """
        # Strip any leading/trailing spaces from column names
        df.columns = df.columns.str.strip()
        
        # Check if the column exists in the DataFrame
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        # Check if the column elements are lists, in which case convert them to strings
        first_element = df[column].iloc[0]
        if isinstance(first_element, list):
            df[column] = df[column].apply(self.list_to_string)
        elif isinstance(first_element, str):
            # Safely parse the lists from strings if they are string representations of lists
            df[column] = df[column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            df[column] = df[column].apply(self.list_to_string)

        # Encode the data
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        encoded = one_hot_encoder.fit_transform(df[[column]])
        encoded_df = pd.DataFrame(encoded, columns=one_hot_encoder.get_feature_names_out([column]))
        return pd.concat([df.drop(column, axis=1), encoded_df], axis=1)

    def tfidf_vectorize(self, df, column):
        """
        Applies TF-IDF vectorization to a specified column in the dataframe.
        
        Parameters:
        df (pd.DataFrame): The input dataframe.
        column (str): The name of the column to TF-IDF vectorize.
        
        Returns:
        pd.DataFrame: The dataframe with the specified column TF-IDF vectorized.
        """
        # Join list elements into a single string for each row.
        logging.debug("Combine each element in the list of %s", df[column])
        # The string becomes 'none' if the list is empty to ensure proper vectorization.
        df[column] = df[column].apply(lambda x: ' '.join(x) if isinstance(x, list) and x else 'none' if isinstance(x, list) else x)
        
        tfidf_vectorizer = TfidfVectorizer()
        logging.debug("begin vectorization of %s", df[column])
        tfidf_matrix = tfidf_vectorizer.fit_transform(df[column])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
        return pd.concat([df.drop(column, axis=1), tfidf_df], axis=1)

    def normalize_numerical_features(self, df, columns):
        """
        Normalizes or standardizes specified numerical columns in the dataframe.
        
        Parameters:
        df (pd.DataFrame): The input dataframe.
        columns (list): A list of column names to normalize.
        
        Returns:
        pd.DataFrame: The dataframe with the specified columns normalized.
        """
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df[columns])
        scaled_df = pd.DataFrame(scaled_values, columns=columns)
        return pd.concat([df.drop(columns, axis=1), scaled_df], axis=1)
    
    def transform_target_variable(self, df, column):
        """
        Transforms a multi-label target variable into a binary format.
        
        Parameters:
        df (pd.DataFrame): The input dataframe.
        column (str): The name of the column to transform.
        
        Returns:
        pd.DataFrame: The dataframe with the transformed target variable.
        """
        mlb = MultiLabelBinarizer()
        # Check if the column contains string representations of lists
        def parse_list(x):
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except (ValueError, SyntaxError):
                    return []
            return x

        df[column] = df[column].apply(parse_list)
        
        # Ensure all entries are lists
        df[column] = df[column].apply(lambda x: x if isinstance(x, list) else [])
        
        binary_labels = mlb.fit_transform(df[column])
        binary_df = pd.DataFrame(binary_labels, columns=mlb.classes_)
        return df.drop(column, axis=1).join(binary_df)
