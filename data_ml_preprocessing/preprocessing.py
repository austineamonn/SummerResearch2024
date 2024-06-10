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
        df = self.transform_target_variable(df, 'learning_style', suffix='_learning_style')
        df = self.transform_target_variable(df, 'previous courses', suffix='_previous_courses')
        df = self.transform_target_variable(df, 'course type', suffix='_course_type')
        df = self.transform_target_variable(df, 'subjects in courses', suffix='_subjects_in_courses')
        
        df = self.transform_target_variable(df, 'subjects of interest', suffix='_subjects_of_interest')
        df = self.transform_target_variable(df, 'career aspirations', suffix='_career_aspirations')
        df = self.transform_target_variable(df, 'extracurricular activities', suffix='_extracurricular_activities')
        
        df = self.normalize_numerical_features(df, self.numerical_columns)
        
        df = self.transform_target_variable(df, 'future topics')
        
        return df

    def list_to_string(self, lst):
        return " ".join(lst)

    def one_hot_encode(self, df, column, suffix=''):
        df[column] = df[column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df_expanded = pd.get_dummies(df[column].apply(pd.Series).stack()).sum(level=0)
        df_expanded = df_expanded.add_suffix(suffix)
        return df.drop(column, axis=1).join(df_expanded)

    def tfidf_vectorize(self, df, column, suffix=''):
        df[column] = df[column].apply(lambda x: " ".join(ast.literal_eval(x)) if isinstance(x, str) else x)
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(df[column])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[col + suffix for col in tfidf_vectorizer.get_feature_names_out()])
        return pd.concat([df.drop(column, axis=1), tfidf_df], axis=1)

    def normalize_numerical_features(self, df, columns, suffix='_normalized'):
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df[columns])
        scaled_df = pd.DataFrame(scaled_values, columns=[col + suffix for col in columns])
        return pd.concat([df.drop(columns, axis=1), scaled_df], axis=1)

    def transform_target_variable(self, df, column, suffix=''):
        mlb = MultiLabelBinarizer()
        def parse_list(x):
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except (ValueError, SyntaxError):
                    return []
            return x

        df[column] = df[column].apply(parse_list)
        df[column] = df[column].apply(lambda x: x if isinstance(x, list) else [])
        binary_labels = mlb.fit_transform(df[column])
        binary_df = pd.DataFrame(binary_labels, columns=[col + suffix for col in mlb.classes_])
        return df.drop(column, axis=1).join(binary_df)
