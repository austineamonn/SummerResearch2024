import pandas as pd
import numpy as np
import random
import re

# Function to load course catalog
def load_course_catalog(file_path):
    course_catalog = pd.read_csv(file_path)
    course_names = course_catalog['Name'].tolist()
    course_numbers = course_catalog['Number'].tolist()
    course_names = [name.replace('&amp;', 'and') for name in course_names if pd.notna(name)]
    return course_names, course_numbers

# Function to load mappings from CSV
def load_mapping_from_csv(filename):
    mapping = {}
    df = pd.read_csv(filename)
    for index, row in df.iterrows():
        activity = row.iloc[0]
        subjects = row.iloc[1:].dropna().tolist()
        mapping[activity] = subjects
    return mapping

# Function to load extracurricular to subject mapping from CSV
def load_extracurricular_to_subject(filename):
    df = pd.read_csv(filename)
    mapping = {}
    for index, row in df.iterrows():
        activity = row.iloc[0]
        subjects = tuple(row.iloc[1].split(',')) if pd.notna(row.iloc[1]) else []
        mapping[activity] = subjects
    return mapping

# Function to load extracurricular activities from CSV
def load_extracurricular_activities(filename):
    df = pd.read_csv(filename)
    return df['Activity'].tolist()

# Load extracurricular activities
extracurricular_activities_list = load_extracurricular_activities('/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/CSV_builder/extracurricular_activities.csv')

# Load extracurricular to subject mapping
extracurricular_to_subject = load_extracurricular_to_subject('/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/CSV_builder/extracurricular_to_subject.csv')

# Load course catalog
course_names, course_numbers = load_course_catalog('/Users/austinnicolas/Documents/SummerREU2024/course-catalog.csv')

#course catalog is from: https://discovery.cs.illinois.edu/dataset/course-catalog/

# Load mappings from CSV
course_to_subject = load_mapping_from_csv('/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/CSV_builder/course_to_subject.csv')
related_topics = load_mapping_from_csv('/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/CSV_builder/related_topics.csv')
related_career_aspirations = load_mapping_from_csv('/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/CSV_builder/related_career_aspirations.csv')

# Function to map course names to subjects
def map_course_to_subject(course_name):
    for subject, keywords in course_to_subject.items():
        if any(keyword in course_name for keyword in keywords):
            return subject
    return None

# Map each course to its subject
course_subject_mapping = {course: map_course_to_subject(course) for course in course_names}

# Define a function to map courses to class years based on course numbers
def map_course_to_year(course_number):
    if pd.isna(course_number):
        return None
    match = re.match(r'(\d{3})', str(course_number))
    if match:
        course_num = int(match.group(1))
        if 100 <= course_num < 200:
            return 1, 2 # First year: first and second semester
        elif 200 <= course_num < 300:
            return 3, 4 # Second year: first and second semester
        elif 300 <= course_num < 400:
            return 5, 6 # Third year: first and second semester
        elif 400 <= course_num < 500:
            return 7, 8 # Fourth year: first and second semester
        else:
            return 9, 10, 11, 12 # Fifth and Sixth year: first and second semester
    return None

# Map each course to its class year range
course_year_mapping = {course: map_course_to_year(number) for course, number in zip(course_names, course_numbers)}

# Extracted demographic distributions from https://educationdata.org/college-enrollment-statistics
demographics_distribution = {
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
    }
}

# Function to randomly assign a demographic value based on its distribution
def assign_demographic(demographic_type, bias):
    demographic_dist = demographics_distribution[demographic_type]
    demographics = list(demographic_dist.keys())
    if bias == 'real':
        probabilities = [demographic_dist[demo] / 100 for demo in demographics]
        result = np.random.choice(demographics, p=probabilities)
    else:
        result = np.random.choice(demographics)
    return result