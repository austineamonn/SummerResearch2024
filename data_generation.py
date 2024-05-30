import pandas as pd
import numpy as np
import random
import logging
import re
from sklearn.preprocessing import StandardScaler
from csv_loading import course_loader, fn_loader, ln_loader
from dictionary import get_combined_data

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataGenerator:
    def __init__(self):
        combined_data = get_combined_data()
        self.subjects_of_interest = combined_data['subjects_of_interest']
        self.course_to_subject = combined_data['course_to_subject']
        self.related_career_aspirations = combined_data['related_career_aspirations']
        self.extracurricular_list = combined_data['extracurricular_list']
        self.related_topics = combined_data['related_topics']
        self.course_type_to_learning_styles = combined_data['course_type_to_learning_styles']
        self.careers = combined_data['careers']
        self.course_names = course_loader.course_names
        self.course_numbers = course_loader.course_numbers
        self.race_ethnicity = combined_data['race_ethnicity']
        self.gender = combined_data['gender']
        self.international = combined_data['international']
        self.socioeconomic = combined_data['socioeconomic']
        self.learning_style = combined_data['learning_style']
        self.first_names_list = fn_loader.first_names
        self.last_names_list = ln_loader.last_names
        self.course_year_mapping = {course: self.map_course_to_year(number) for course, number in zip(self.course_names, self.course_numbers)}

    def generate_grades(self, previous_courses):
        """
        Simulate grades for previous courses.
        """
        grades = {course: np.random.choice(['A', 'B', 'C', 'D', 'F'], p=[0.3, 0.3, 0.2, 0.1, 0.1]) for course in previous_courses}
        return grades

    def filter_courses_by_year(self, courses, class_year):
        """
        Filter courses based on class year.
        """
        return [course for course in courses if class_year in range(*self.course_year_mapping.get(course, (0, 0)))]

    def generate_previous_courses(self, student_class_year, learning_style):
        """
        Generate a list of previous courses taken by a student based on their class year and their learning style.
        """
        previous_courses_taken = []
        for year in range(1, student_class_year + 1):
            possible_courses = self.filter_courses_by_year(self.course_names, year)
            # Filter courses based on learning style
            filtered_courses = [
                course for course in possible_courses
                if any(style in self.course_type_to_learning_styles.get(course, []) for style in learning_style)
            ]
            num_courses = max(0, min(4 + np.random.randint(-2, 2), len(filtered_courses)))  # Adding some randomness
            previous_courses_taken.extend(random.sample(filtered_courses, num_courses))

        return previous_courses_taken
    
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
        demographics = list(demographic_type.keys())
        if bias == 'real':
            probabilities = [demographic_type[demo] / 100 for demo in demographics]
            result = np.random.choice(demographics, p=probabilities)
        else:
            result = np.random.choice(demographics)
        return result

    def generate_career_aspirations(self, subjects):
        """
        Generate a list of career aspirations for a student based on their subjects of interest.
        """
        # Find all possible careers related to the subjects of interest
        possible_careers = []
        for subject in subjects:
            if subject in self.related_career_aspirations:
                possible_careers.extend(self.related_career_aspirations[subject])
        
        # Add a small chance of including a random career from all possible careers
        if np.random.rand() < 0.1:  # 10% chance
            possible_careers.append(np.random.choice(self.careers))
        
        # Remove duplicates from the possible careers list
        possible_careers = list(set(possible_careers))
        
        # Select a random number of careers (between 0 and 4) from the possible careers
        return random.sample(possible_careers, min(np.random.randint(0, 5), len(possible_careers))) if possible_careers else []

    def generate_synthetic_dataset(self, num_samples=1000):
        """
        Generate the synthetic dataset.
        """
        data = []
        for _ in range(num_samples):
            first_name = random.choice(self.first_names_list)
            last_name = random.choice(self.last_names_list)
            student_class_year = np.random.randint(1, 5)
            learning_style = self.assign_demographic(self.learning_style, 'real')
            previous_courses = self.generate_previous_courses(student_class_year, learning_style)
            previous_courses_count = len(previous_courses)
            subjects_in_courses = [self.course_to_subject[course] for course in previous_courses if course in self.course_to_subject]
            unique_subjects_in_courses = len(set(subjects_in_courses))
            subjects_of_interest_list = random.sample(self.subjects_of_interest,  min(np.random.randint(1, 5), len(self.subjects_of_interest)))
            subjects_diversity = len(subjects_of_interest_list)
            career_aspirations_list = self.generate_career_aspirations(subjects_of_interest_list)
            extracurricular_activities = random.sample(self.extracurricular_list, min(np.random.randint(0, 4), len(self.extracurricular_list)))
            activities_involvement_count = len(extracurricular_activities)
            future_topics = random.sample(list(self.related_topics.keys()), min(np.random.randint(1, 5), len(self.related_topics.keys()))) if self.related_topics else []
            gpa = round(np.random.uniform(2.0, 4.0), 2)
            
            data.append({
                'first_name': first_name,
                'last_name': last_name,
                'race_ethnicity': self.assign_demographic(self.race_ethnicity, 'real'),
                'gender': self.assign_demographic(self.gender, 'real'),
                'international': self.assign_demographic(self.international, 'real'),
                'socioeconomic status': self.assign_demographic(self.socioeconomic, 'real'),
                'learning_style': learning_style,
                'gpa': gpa,
                'class year': student_class_year,
                'previous courses': previous_courses,
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
        
        # Normalize numerical features
        numerical_columns = ['previous courses count', 'unique subjects in courses', 'subjects diversity', 'activities involvement count', 'gpa']
        df = self.normalize_numerical_features(df, numerical_columns)
        
        return df

    def normalize_numerical_features(self, dataframe, numerical_columns):
        """
        Normalize numerical features.
        """
        scaler = StandardScaler()
        dataframe[numerical_columns] = scaler.fit_transform(dataframe[numerical_columns])
        return dataframe

# Export the function for external use
def generate_synthetic_dataset(num_samples=1000):
    generator = DataGenerator()
    return generator.generate_synthetic_dataset(num_samples)

# Usage example
if __name__ == "__main__":
    generator = DataGenerator()
    synthetic_dataset = generator.generate_synthetic_dataset(1000)
    logging.info("Synthetic dataset generated with %d samples.", len(synthetic_dataset))
    logging.info("First few rows of the synthetic dataset:\\n%s", synthetic_dataset.head())
