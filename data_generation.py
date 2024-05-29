import pandas as pd
import numpy as np
import random
import re
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import sklearn
from packaging import version

# Import functions and data frames
from csv_loading import course_year_mapping, course_names, related_career_aspirations, extracurricular_activities_list, assign_demographic, course_subject_mapping, related_topics, extracurricular_to_subject

# Expanded list of subjects of interest
subjects_of_interest = [
    "Physics", "Mathematics", "Biology", "Chemistry", "History", "Literature", 
    "Computer Science", "Art", "Music", "Economics", "Psychology", "Sociology", 
    "Anthropology", "Political Science", "Philosophy", "Environmental Science", 
    "Geology", "Astronomy", "Engineering", "Medicine", "Law", "Business", 
    "Education", "Communications", "Languages", "Theater", "Dance"
]

# Simulate grades for previous courses
def generate_grades(previous_courses):
    grades = {}
    for course in previous_courses:
        grades[course] = np.random.choice(['A', 'B', 'C', 'D', 'F'], p=[0.3, 0.3, 0.2, 0.1, 0.1])
    return grades

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
    previous_courses_taken = []
    for year in range(1, student_class_year + 1):
        possible_courses = filter_courses_by_year(course_names, year)
        num_courses = 4 + np.random.randint(-2, 2)  # Adding some randomness
        num_courses = max(0, min(num_courses, len(possible_courses)))  # Ensure we don't exceed available courses
        previous_courses_taken.extend(random.sample(possible_courses, num_courses))
        previous_courses_taken = sorted(previous_courses_taken, key=lambda course: course_year_mapping[course])
    return previous_courses_taken

# Allow students to have interdisciplinary subjects of interest
def generate_subjects_of_interest():
    if np.random.rand() > 0.2:
        num_subjects_of_interest = np.random.randint(1, 5)
        return random.sample(subjects_of_interest, num_subjects_of_interest)
    else:
        return []

# Generate career aspirations based on subjects of interest and previous courses
def generate_career_aspirations(subjects_of_interest_list, previous_courses):
    possible_careers = []

    # Add career aspirations related to subjects of interest
    for subject in subjects_of_interest_list:
        if subject in related_career_aspirations:
            possible_careers.extend(related_career_aspirations[subject])

    # Add career aspirations related to previous courses
    for course in previous_courses:
        if course in course_subject_mapping:
            subject = course_subject_mapping[course]
            possible_careers += related_career_aspirations.get(subject, [])

    # Remove duplicates
    possible_careers = list(set(possible_careers))

    # Determine the number of careers to sample
    num_careers = min(5, len(possible_careers))

    if possible_careers:
        possible_careers = random.sample(possible_careers, num_careers)
    
    return possible_careers

# Generate extracurricular activities with dependencies on subjects of interest
def generate_extracurricular_activities(subjects_of_interest):
    num_activities = np.random.randint(1, 5)
    possible_activities = extracurricular_activities_list.copy()
    
    # Bias towards activities related to subjects of interest
    for subject in subjects_of_interest:
        if subject in extracurricular_to_subject:
            possible_activities += extracurricular_to_subject[subject]
    
    return random.sample(possible_activities, num_activities)

# Generate socioeconomic status (SES)
def generate_socioeconomic_status():
    return np.random.choice(['Low', 'Middle', 'High'], p=[0.3, 0.5, 0.2])

# Function to generate the synthetic dataset
def generate_synthetic_dataset(num_samples=1000, correlation_probability=0.7, bias='real'):
    data = []
    for _ in range(num_samples):
        student_class_year = np.random.choice(range(13), p=[0.2] + [0.065]*11 + [0.085])
        
        previous_courses_taken=generate_previous_courses(student_class_year)

        subjects_of_interest_list=generate_subjects_of_interest()

        career_aspirations_list=generate_career_aspirations(subjects_of_interest_list, previous_courses_taken)

        extracurricular_activities = generate_extracurricular_activities(subjects_of_interest_list)

        grades = generate_grades(previous_courses_taken)

        # Feature Engineering
        previous_courses_count = len(previous_courses_taken)
        unique_subjects_in_courses = len(set(course_subject_mapping[course] for course in previous_courses_taken))
        subjects_diversity = len(subjects_of_interest_list)
        activities_involvement_count = len(extracurricular_activities)
        gpa = np.mean([{'A': 4.0, 'B': 3.0, 'C': 2.0, 'D': 1.0, 'F': 0.0}[grade] for grade in grades.values()])

        # Assign demographics
        race_ethnicity = assign_demographic('race_ethnicity', bias)
        gender = assign_demographic('gender', bias)
        international = assign_demographic('international', bias)
        socioeconomic_status = generate_socioeconomic_status()
        
        # Generate future topics based on subjects from previous courses taken, career aspirations, subjects of interest, and extracurriculars
        future_topics = []

        # Add future topics based on previous courses taken
        correlated_courses = [course for course in previous_courses_taken if np.random.rand() < correlation_probability]
        for course in correlated_courses:
            subject = course_subject_mapping.get(course)
            if subject and subject in related_topics:
                future_topics.extend(related_topics[subject])
        
        # Add future topics based on subjects of interest
        correlated_subjects = [subject for subject in subjects_of_interest_list if np.random.rand() < correlation_probability]
        for subject in correlated_subjects:
            if subject in related_topics:
                future_topics.extend(related_topics[subject])
        
        # Add future topics based on career aspirations
        correlated_careers = [career for career in career_aspirations_list if np.random.rand() < correlation_probability]
        for career in correlated_careers:
            for subj, careers in related_career_aspirations.items():
                if career in careers and subj in related_topics:
                    future_topics.extend(related_topics[subj])                  

        # Add future topics based on extracurricular activities
        correlated_extracurriculars = [extracurricular for extracurricular in extracurricular_activities_list if np.random.rand() < correlation_probability]
        for activity in correlated_extracurriculars:
            if activity in extracurricular_to_subject:
                future_topics.extend(extracurricular_to_subject[activity])
        
        # Ensure future topics are distinct and limited to 10
        future_topics = list(set(future_topics))
        if not future_topics and (previous_courses_taken or career_aspirations_list or subjects_of_interest_list or extracurricular_activities_list):
            # Rare case where user has no previous classes, no career aspirations, no subjects of interest, and no extracurriculars
            future_topics = ["Sorry, you did not provide enough details for the model to make an accurate prediction of possible topics of interest for you."]
        
        future_topics = random.sample(future_topics, min(10, len(future_topics)))
        
        data.append({
            'race_ethnicity': race_ethnicity,
            'gender': gender,
            'international': international,
            'socioeconomic status': socioeconomic_status,
            'student class year': student_class_year,
            'previous courses taken': previous_courses_taken,
            'grades': grades,
            'gpa': gpa,
            'previous courses count': previous_courses_count,
            'unique subjects in courses': unique_subjects_in_courses,
            'subjects of interest': subjects_of_interest_list,
            'subjects diversity': subjects_diversity,
            'career aspirations': career_aspirations_list,
            'extracurricular activities': extracurricular_activities,
            'activities involvement count': activities_involvement_count,
            'future topics': future_topics
        })
    
    df = pd.DataFrame(data)
    
    # Add noise to numerical features
    numerical_columns = ['previous courses count', 'unique subjects in courses', 'subjects diversity', 'activities involvement count', 'gpa']
    for col in numerical_columns:
        df = add_noise(df, col)
    
    # Encode categorical features
    categorical_columns = ['race_ethnicity', 'gender', 'international', 'socioeconomic status']
    df = encode_categorical_features(df, categorical_columns)
    
    # Normalize numerical features
    df = normalize_numerical_features(df, numerical_columns)
    
    return df

def add_noise(dataframe, column_name, noise_level=0.01):
    noise = np.random.normal(0, noise_level, dataframe[column_name].shape)
    dataframe[column_name] += noise
    return dataframe

def encode_categorical_features(df, categorical_columns):
    if version.parse(sklearn.__version__) >= version.parse("1.2.0"):
        encoder = OneHotEncoder(sparse_output=False)
    else:
        encoder = OneHotEncoder(sparse=False)
    encoded_features = encoder.fit_transform(df[categorical_columns])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))
    df = df.drop(columns=categorical_columns).reset_index(drop=True)
    df = pd.concat([df, encoded_df], axis=1)
    return df

def normalize_numerical_features(dataframe, numerical_columns):
    scaler = StandardScaler()
    dataframe[numerical_columns] = scaler.fit_transform(dataframe[numerical_columns])
    return dataframe