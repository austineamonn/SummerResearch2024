import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
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

        # Privatized - Utility split
        self.Xp = config["privacy"]["Xp_list"]
        self.X = config["privacy"]["X_list"]
        self.Xu = config["privacy"]["Xu_list"]
        self.numerical_cols = config["privacy"]["numerical_columns"]

        # PCA
        self.n_components = config["preprocessing"]["n_components"]

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
        
    def PCA(self, df, n_components=100):
        """
        Input: dataframe (columns to run PCA analysis on), number of components
        Output: PCA analyzed dataframe, PCA
        """
        # Save the locations of NaNs
        nan_mask = df.isna()

        # Impute NaNs with the mean for PCA
        imputer = SimpleImputer(strategy='mean')
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        # Standardizing the features
        features = df_imputed.columns
        df = StandardScaler().fit_transform(df_imputed)

        # Convert standardized data back to a DataFrame
        df_standardized = pd.DataFrame(data=df, columns=features)

        # Initialize PCA
        pca = PCA(n_components=n_components)

        # Fit and transform the standardized data
        principal_components = pca.fit_transform(df_standardized)

        # Create a DataFrame with the principal components
        principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)])

        # Restore NaNs to the PCA results where they originally were
        for col in principal_df.columns:
            principal_df[col] = principal_df[col].where(~nan_mask.any(axis=1), np.nan)

        return principal_df, pca
    
    def analyze_PCA(self, pca):
        """
        Input: PCA
        Output: none
        """
        # Explained variance
        explained_variance = pca.explained_variance_ratio_
        logging.debug(f'Explained variance by each component: {explained_variance}')

        # Plotting the explained variance
        plt.figure(figsize=(8, 5))
        plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center', label='individual explained variance')
        plt.step(range(1, len(explained_variance) + 1), explained_variance.cumsum(), where='mid', label='cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.legend(loc='best')
        plt.tight_layout()
        
        # Save the plot as a file
        plt.savefig(self.config["running_model"]["PCA explained variance path"])
        logging.info("PCA explained variance graph saved to 'explained_variance_plot.png'")

    def preprocess_dataset(self, df, analyze_PCA=True):
        """
        Input: dataset, show_PCA (boolean)
        Output: preprocessed dataset
        """
        utility_cols = []

        # Drop the Xp columns
        df = df.drop(columns=self.Xp)
        logging.debug(f"{self.Xp} were dropped")

        for col in self.X + self.Xu:
            suffix = '_' + col

            if col not in self.numerical_cols:
                # Binarize the column
                df, new_cols = self.transform_target_variable(df, col, suffix)
                logging.debug(f"{col} was binarized")
                
                if col in self.Xu:
                    utility_cols.extend(new_cols)
        
        # Standardize the data and run PCA
        PCA_df = df.drop(columns = utility_cols + self.numerical_cols)
        PCA_df, pca = self.PCA(PCA_df, self.n_components)

        # Recombine the principal components with Xu and the numerical columns
        Xu_df = df[utility_cols]
        Xn_df = df[self.numerical_cols]
        df = pd.concat([Xn_df, PCA_df, Xu_df], axis=1)

        # Analyze the PCA
        if analyze_PCA:
            self.analyze_PCA(pca)

        return df