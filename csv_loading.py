import pandas as pd
import numpy as np
import random
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CSVLoader:
    def __init__(self, base_path):
        self.base_path = base_path
        self.course_year_mapping = {}
        self.course_names = []
        self.related_career_aspirations = []
        self.extracurricular_activities_list = []
        self.course_subject_mapping = {}
        self.related_topics = []
        self.extracurricular_to_subject = {}
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
            }
        }

    def load_course_catalog(self, file_path):
        logging.info("Loading course catalog from %s", file_path)
        course_catalog = pd.read_csv(file_path)
        self.course_names = course_catalog['Name'].tolist()
        course_numbers = course_catalog['Number'].tolist()
        self.course_names = [name.replace('&', 'and') for name in self.course_names if pd.notna(name)]
        self.course_year_mapping = {course: self.map_course_to_year(number) for course, number in zip(self.course_names, course_numbers)}

    def load_mapping_from_csv(self, filename):
        logging.info("Loading mapping from %s", filename)
        mapping = {}
        df = pd.read_csv(filename)
        for _, row in df.iterrows():
            activity = row.iloc[0]
            subjects = row.iloc[1:].dropna().tolist()
            mapping[activity] = subjects
        return mapping

    def load_extracurricular_to_subject(self, filename):
        logging.info("Loading extracurricular to subject mapping from %s", filename)
        df = pd.read_csv(filename)
        for _, row in df.iterrows():
            activity = row.iloc[0]
            subjects = tuple(row.iloc[1].split(',')) if pd.notna(row.iloc[1]) else []
            self.extracurricular_to_subject[activity] = subjects

    def load_extracurricular_activities(self, filename):
        logging.info("Loading extracurricular activities from %s", filename)
        df = pd.read_csv(filename)
        self.extracurricular_activities_list = df['Activity'].tolist()

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
loader = CSVLoader(base_path='/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/CSV_builder')
loader.load_extracurricular_activities('/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/CSV_builder/extracurricular_activities.csv')
loader.load_extracurricular_to_subject('/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/CSV_builder/extracurricular_to_subject.csv')
loader.load_course_catalog('/Users/austinnicolas/Documents/SummerREU2024/course-catalog.csv')

# Example demographic assignment
demographic = loader.assign_demographic('race_ethnicity', 'real')
logging.info("Assigned demographic: %s", demographic)

# Export for other modules
course_year_mapping = loader.course_year_mapping
course_names = loader.course_names
related_career_aspirations = loader.related_career_aspirations
extracurricular_activities_list = loader.extracurricular_activities_list
course_subject_mapping = loader.course_subject_mapping
related_topics = loader.related_topics
extracurricular_to_subject = loader.extracurricular_to_subject
assign_demographic = loader.assign_demographic
