import pandas as pd
import logging
from config import load_config

# Load configuration
config = load_config()

# Set up logging
logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])

class CSVLoader:
    def __init__(self):
        self.first_names = []
        self.last_names = []
        self.course_tuples = []

    def load_course_catalog(self, file_path):
        logging.debug("Loading course catalog from %s", file_path)
        course_catalog = pd.read_csv(file_path)

        # Create tuples of a courses name, number, type, and subject. Also replace '&amp' with 'and' in the
        # names. Then cut out duplicate classes
        self.course_tuples = list(
            course_catalog[['Name', 'Number', 'Type', 'Subject']]
            .assign(Name=course_catalog['Name'].str.replace('&amp', 'and'))
            .itertuples(index=False, name=None)
        )
        self.course_tuples = list(set(self.course_tuples))

    def load_first_names(self, file_path):
        logging.debug("Loading first names from %s", file_path)
        first_names_list = pd.read_csv(file_path)
        self.first_names = first_names_list['Name'].tolist()

    def load_last_names(self, file_path):
        logging.debug("Loading last names from %s", file_path)
        last_names_list = pd.read_csv(file_path)
        self.last_names = last_names_list['name'].tolist()


# Initialize and load data
course_loader = CSVLoader()
course_loader.load_course_catalog('/Users/austinnicolas/Documents/SummerREU2024/course-catalog.csv')

fn_loader = CSVLoader()
fn_loader.load_first_names('/Users/austinnicolas/Documents/SummerREU2024/SSA_Names_DB.csv')

ln_loader = CSVLoader()
ln_loader.load_last_names('/Users/austinnicolas/Documents/SummerREU2024/Modified_Common_Surnames_Census_2000.csv')

# Add in logging to confirm all CSVs have been loaded
logging.info("All the CSVs have been loaded.")
