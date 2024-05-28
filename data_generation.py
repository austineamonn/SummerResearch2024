import pandas as pd
import numpy as np
import random
import re

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
def generate_synthetic_dataset(num_samples=1000, correlation_probability=0.7, bias='real'):
    data = []
    for _ in range(num_samples):
        student_class_year = np.random.choice(range(13), p=[0.2] + [0.065]*11 + [0.085])
        
        previous_courses_taken=generate_previous_courses(student_class_year)

        subjects_of_interest_list=generate_subjects_of_interest()

        career_aspirations_list=generate_career_aspirations(subjects_of_interest_list)

        extracurricular_activities = generate_extracurriculars()

        # Assign demographics
        race_ethnicity = assign_demographic('race_ethnicity', bias)
        gender = assign_demographic('gender', bias)
        international = assign_demographic('international', bias)
        
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
            'race/ethnicity': race_ethnicity,
            'gender': gender,
            'international': international,
            'student class year': student_class_year,
            'previous courses taken': previous_courses_taken,
            'career aspirations': career_aspirations_list,
            'subjects of interest': subjects_of_interest_list,
            'extracurricular activities': extracurricular_activities,
            'future topics': future_topics
        })
    
    return pd.DataFrame(data)