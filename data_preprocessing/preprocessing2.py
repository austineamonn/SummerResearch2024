import logging
import numpy as np
import pandas as pd
import sys
import os
import ast

# Add the SummerResearch2024 directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import modules from SummerResearch2024 directory
from config import load_config
from datafiles_for_data_construction.data import Data

# Load configuration
config = load_config()

# Set up logging
logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])

class PreProcessing:
    def __init__(self, config, data) -> None:
        self.config = config

        # Privatized - Utility split
        self.Xp = config["privacy"]["Xp_list"]
        self.X = config["privacy"]["X_list"]
        self.Xu = config["privacy"]["Xu_list"]
        self.numerical_cols = config["privacy"]["numerical_columns"]

        # Data
        self.data = data

    def stringlist_to_binarylist(self, stringlist, col):
        binarylist = []

        for rowlist in col:
            # Ensure rowlist is a list of strings
            if isinstance(rowlist, str):
                if isinstance(rowlist, str):
                    try:
                        rowlist = ast.literal_eval(rowlist)
                        if not isinstance(rowlist, list):
                            raise ValueError
                    except (ValueError, SyntaxError):
                        # Handle the case where the string cannot be converted to a list
                        rowlist = rowlist.strip("[]").replace("'", "").split(", ")

            logging.debug(f"Processing row: {rowlist}")
            # Create a list of zeroes the length of the stringlist
            zeros_list = np.zeros(len(stringlist), dtype=int).tolist()

            # Iterate through each item in the list
            for item in rowlist:
                item = item.strip()
                logging.debug(f"Processing item: {item}")
                if item in stringlist:
                    item_spot = stringlist.index(item)
                    zeros_list[item_spot] = 1
                else:
                    logging.debug(f"Error: '{item}' is not in list")

            # Add the list to the binary list
            binarylist.append(zeros_list)

        return binarylist

    def string_list_to_numberedlist(self, stringlist, col):
        numberedlist = []
        unknown_item_list = []

        for rowlist in col:
            # Ensure rowlist is a list of strings
            if isinstance(rowlist, str):
                try:
                    rowlist = ast.literal_eval(rowlist)
                    if not isinstance(rowlist, list):
                        raise ValueError
                except (ValueError, SyntaxError):
                    # Handle the case where the string cannot be converted to a list
                    rowlist = rowlist.strip("[]").replace("'", "").split(", ")

            logging.debug(f"Processing row: {rowlist}")
            # Initialize a new list
            newlist = []
            for item in rowlist:
                item = item.strip()
                logging.debug(f"Processing item: {item}")
                if item in stringlist:
                    # For each item in the list add its index to the new list
                    newlist.append(stringlist.index(item))
                else:
                    unknown_item_list.append(item)
                    logging.debug(f"Error: '{item}' is not in list")
            
            # Add the list to the numbered list
            numberedlist.append(newlist)

            unknown_item_list = list(set(unknown_item_list))

        return numberedlist, unknown_item_list
    
    def preprocess_dataset(self, df):
        """
        Input: dataset
        Output: preprocessed dataset
        """
        list_data = {}

        # Drop the Xp columns
        df = df.drop(columns=self.Xp)
        logging.debug(f"{self.Xp} were dropped")

        # Iterate through the columns
        for col in self.X + self.Xu:
            if col not in self.numerical_cols:
                # X
                if col == 'learning style':
                    stringlist = self.data.learning_style()['learning_style_list']
                    df[col] = self.stringlist_to_binarylist(stringlist, df[col])
                elif col == 'major':
                    stringlist = self.data.major()['majors_list']
                    df[col], unknown_list = self.string_list_to_numberedlist(stringlist, df[col])
                    print('majors: %s', unknown_list)
                elif col == 'previous courses':
                    stringlist = self.data.course()['course_names_list']
                    df[col], unknown_list = self.string_list_to_numberedlist(stringlist, df[col])
                    print('previous courses: %s', unknown_list)
                elif col == 'course types':
                    stringlist = self.data.course()['course_type_list']
                    df[col], unknown_list = self.string_list_to_numberedlist(stringlist, df[col])
                    print('course types: %s', unknown_list)
                elif col == 'course subjects':
                    stringlist = self.data.course()['course_subject']
                    df[col], unknown_list = self.string_list_to_numberedlist(stringlist, df[col])
                    print('course subjects: %s', unknown_list)
                elif col == 'subjects of interest':
                    stringlist = self.data.subjects()['subjects_list']
                    df[col], unknown_list = self.string_list_to_numberedlist(stringlist, df[col])
                    print('subjects of interest: %s', unknown_list)
                elif col == 'extracurricular activities':
                    stringlist = self.data.extracurricular_activities()['complete_activity_list']
                    df[col], unknown_list = self.string_list_to_numberedlist(stringlist, df[col])
                    print('extracurricular activities: %s', unknown_list)
                # Xu
                elif col == 'career aspirations':
                    stringlist = self.data.careers()['careers_list']
                    df[col], unknown_list = self.string_list_to_numberedlist(stringlist, df[col])
                    print('career aspirations: %s', unknown_list)
                elif col == 'future topics':
                    stringlist = self.data.future_topics()['future_topics']
                    df[col], unknown_list = self.string_list_to_numberedlist(stringlist, df[col])
                    print('future topics: %s', unknown_list)
                else:
                    logging.error(f"{col} is not a known column name")

        return df

df = pd.read_csv(config["running_model"]["data path"])
data = Data()
preprocessor = PreProcessing(config, data)

df = preprocessor.preprocess_dataset(df)
df.to_csv(config["running_model"]["preprocessed data path"], index=False)
