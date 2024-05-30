import pandas as pd
import numpy as np
import random
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CSVLoader:
    def __init__(self):
        self.course_year_mapping = {}
        self.course_names = []
        self.demographics_distribution = {
            'race_ethnicity': {
                'European American or white': 53.4,
                'Latino/a/x American': 20.6,
                'African American or Black': 13.1,
                'Asian American': 7.6,
                'Multiracial': 4.3,
                'American Indian or Alaska Native': 0.7,
                'Pacific Islander': 0.3
            },
            'gender': {
                'Female': 58.36,
                'Male': 41.64
            },
            'international': {
                'Domestic': 94.08,
                'International': 5.92
            },
            'socioeconomic': {
                'Low': 20,
                'Middle': 60,
                'High': 20
            },
            'learning_style': {
                'Visual': 27.27,
                'Auditory': 23.56,
                'Read/Write': 21.16,
                'Kinesthetic': 28.01
            }
        }

    def load_course_catalog(self, file_path):
        logging.info("Loading course catalog from %s", file_path)
        course_catalog = pd.read_csv(file_path)
        self.course_names = course_catalog['Name'].tolist()
        course_numbers = course_catalog['Number'].tolist()
        self.course_names = [name.replace('&', 'and') for name in self.course_names if pd.notna(name)]
        self.course_year_mapping = {course: self.map_course_to_year(number) for course, number in zip(self.course_names, course_numbers)}

    def map_course_to_year(self, course_number):
        if pd.isna(course_number):
            return (0, 0)
        match = re.match(r'(\d{3})', str(course_number))
        if match:
            course_num = int(match.group(1))
            if 100 <= course_num < 200:
                return (1, 2)  # First year: first and second semester
            elif 200 <= course_num < 300:
                return (3, 4)  # Second year: first and second semester
            elif 300 <= course_num < 400:
                return (5, 6)  # Third year: first and second semester
            elif 400 <= course_num < 500:
                return (7, 8)  # Fourth year: first and second semester
            else:
                return (9, 10)  # Fifth and Sixth year: first and second semester
        return (0, 0)

    def assign_demographic(self, demographic_type, bias):
        demographic_dist = self.demographics_distribution[demographic_type]
        demographics = list(demographic_dist.keys())
        if bias == 'real':
            probabilities = [demographic_dist[demo] / 100 for demo in demographics]
            result = np.random.choice(demographics, p=probabilities)
        else:
            result = np.random.choice(demographics)
        return result


# Initialize and load data
loader = CSVLoader()
loader.load_course_catalog('/Users/austinnicolas/Documents/SummerREU2024/course-catalog.csv')

# Export for other modules
course_year_mapping = loader.course_year_mapping
course_names = loader.course_names
assign_demographic = loader.assign_demographic
