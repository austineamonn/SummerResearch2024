import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CSVLoader:
    def __init__(self):
        self.course_year_mapping = {}
        self.course_names = []
        self.first_names = []
        self.last_names = []
        self.course_numbers = []

    def load_course_catalog(self, file_path):
        logging.info("Loading course catalog from %s", file_path)
        course_catalog = pd.read_csv(file_path)
        self.course_names = course_catalog['Name'].tolist()
        self.course_numbers = course_catalog['Number'].tolist()
        self.course_names = [name.replace('&', 'and') for name in self.course_names if pd.notna(name)]

    def load_first_names(self, file_path):
        logging.info("Loading first names from %s", file_path)
        first_names_list = pd.read_csv(file_path)
        self.first_names = first_names_list['Name'].tolist()

    def load_last_names(self, file_path):
        logging.info("Loading last names from %s", file_path)
        last_names_list = pd.read_csv(file_path)
        self.last_names = last_names_list['name'].tolist()


# Initialize and load data
course_loader = CSVLoader()
course_loader.load_course_catalog('/Users/austinnicolas/Documents/SummerREU2024/course-catalog.csv')

fn_loader = CSVLoader()
fn_loader.load_first_names('/Users/austinnicolas/Documents/SummerREU2024/SSA_Names_DB.csv')

ln_loader = CSVLoader()
ln_loader.load_last_names('/Users/austinnicolas/Documents/SummerREU2024/Modified_Common_Surnames_Census_2000.csv')
