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
            return 1, 2
        elif 200 <= course_num < 300:
            return 3, 4
        elif 300 <= course_num < 400:
            return 5, 6
        elif 400 <= course_num < 500:
            return 7, 8
        else:
            return 9, 10
    return None

# Map each course to its class year range
course_year_mapping = {course: map_course_to_year(number) for course, number in zip(course_names, course_numbers)}

# Expanded list of subjects of interest
subjects_of_interest = [
    "Physics", "Mathematics", "Biology", "Chemistry", "History", "Literature", 
    "Computer Science", "Art", "Music", "Economics", "Psychology", "Sociology", 
    "Anthropology", "Political Science", "Philosophy", "Environmental Science", 
    "Geology", "Astronomy", "Engineering", "Medicine", "Law", "Business", 
    "Education", "Communications", "Languages", "Theater", "Dance"
]

# Function to filter courses based on class year
def filter_courses_by_year(courses, class_year):
    filtered_courses = []
    for course in courses:
        year_range = course_year_mapping.get(course)
        if year_range and class_year in range(year_range[0], year_range[1] + 1):
            filtered_courses.append(course)
    return filtered_courses

# Ensure the sequence of previous courses taken follows a logical progression
def generate_previous_courses(student_class_year):
    if student_class_year == 0:
        return []
    else:
        possible_courses = filter_courses_by_year(course_names, student_class_year)
        num_courses = student_class_year * 4 + np.random.randint(-3, 4)
        num_courses = max(0, num_courses)
        previous_courses_taken = random.sample(possible_courses, min(num_courses, len(possible_courses)))
        previous_courses_taken = sorted(previous_courses_taken, key=lambda course: course_year_mapping[course])
        return previous_courses_taken
    
# Allow students to have interdisciplinary subjects of interest
def generate_subjects_of_interest():
    if np.random.rand() > 0.2:
        num_subjects_of_interest = np.random.randint(1, 5)
        return random.sample(subjects_of_interest, num_subjects_of_interest)
    else:
        return []

# Generate career aspirations related to subjects of interest
def generate_career_aspirations(subjects_of_interest_list):
    career_aspirations_list = []
    for subject in subjects_of_interest_list:
        if subject in related_career_aspirations:
            career_aspirations_list.extend(related_career_aspirations[subject])
    career_aspirations_list = list(set(career_aspirations_list))
    if career_aspirations_list:
        career_aspirations_list = random.sample(career_aspirations_list, min(5, len(career_aspirations_list)))
    return career_aspirations_list

# Extracurricular Activities
def generate_extracurriculars():
    extracurricular_activities = random.sample(extracurricular_activities_list, np.random.randint(1, 4))
    return extracurricular_activities

# Function to generate the synthetic dataset
def generate_synthetic_dataset(num_samples=10000, correlation_probability=0.7):
    data = []
    for _ in range(num_samples):
        student_class_year = np.random.choice(range(13), p=[0.2] + [0.065]*11 + [0.085])
        
        previous_courses_taken=generate_previous_courses(student_class_year)

        subjects_of_interest_list=generate_subjects_of_interest()

        career_aspirations_list=generate_career_aspirations(subjects_of_interest_list)

        extracurricular_activities = generate_extracurriculars()
        
        # Generate future topics based on subjects from previous courses taken, career aspirations, subjects of interest, and extracurriculars
        future_topics = []
        correlated_courses = [course for course in previous_courses_taken if np.random.rand() < correlation_probability]
        for course in correlated_courses:
            subject = course_subject_mapping.get(course)
            if subject and subject in related_topics:
                future_topics.extend(related_topics[subject])
        
        correlated_subjects = [subject for subject in subjects_of_interest_list if np.random.rand() < correlation_probability]
        for subject in correlated_subjects:
            if subject in related_topics:
                future_topics.extend(related_topics[subject])
        
        correlated_careers = [career for career in career_aspirations_list if np.random.rand() < correlation_probability]
        for career in correlated_careers:
            for subj, careers in related_career_aspirations.items():
                if career in careers and subj in related_topics:
                    future_topics.extend(related_topics[subj])

        correlated_extracurriculars = [extracurricular for extracurricular in extracurricular_activities_list if np.random.rand() < correlation_probability]
        for extracurricular in correlated_extracurriculars:
            for extracurriculars, subject in extracurricular_to_subject.items():
                if extracurricular in extracurriculars and subject in related_topics:
                    future_topics.extend(related_topics[subject])
        
        # Ensure future topics are distinct and limited to 10
        future_topics = list(set(future_topics))
        if not future_topics and (previous_courses_taken or career_aspirations_list or subjects_of_interest_list):
            future_topics = ["Sorry, you did not provide enough details for the model to make an accurate prediction of possible topics of interest for you."]
        
        future_topics = random.sample(future_topics, min(10, len(future_topics)))
        
        data.append({
            'student class year': student_class_year,
            'previous courses taken': previous_courses_taken,
            'career aspirations': career_aspirations_list,
            'subjects of interest': subjects_of_interest_list,
            'extracurricular activities': extracurricular_activities,
            'future topics': future_topics
        })
    
    return pd.DataFrame(data)