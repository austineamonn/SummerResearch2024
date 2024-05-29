import pandas as pd
import numpy as np
import random
import logging
import sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from packaging import version
from csv_loading import course_year_mapping, course_names, related_career_aspirations, extracurricular_activities_list, assign_demographic, course_subject_mapping, related_topics, extracurricular_to_subject

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataGenerator:
    def __init__(self):
        self.subjects_of_interest = [
            "Physics", "Mathematics", "Biology", "Chemistry", "History", "Literature", 
            "Computer Science", "Art", "Music", "Economics", "Psychology", "Sociology", 
            "Anthropology", "Political Science", "Philosophy", "Environmental Science", 
            "Geology", "Astronomy", "Engineering", "Medicine", "Law", "Business", 
            "Education", "Communications", "Languages", "Theater", "Dance"
        ]

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
        return [course for course in courses if class_year in range(*course_year_mapping.get(course, (0, 0)))]

    def generate_previous_courses(self, student_class_year):
        """
        Generate a list of previous courses taken by a student based on their class year.
        """
        previous_courses_taken = []
        for year in range(1, student_class_year + 1):
            possible_courses = self.filter_courses_by_year(course_names, year)
            num_courses = max(0, min(4 + np.random.randint(-2, 2), len(possible_courses)))  # Adding some randomness
            previous_courses_taken.extend(random.sample(possible_courses, num_courses))
        return previous_courses_taken

    def generate_synthetic_dataset(self, num_samples=1000):
        """
        Generate the synthetic dataset.
        """
        data = []
        for _ in range(num_samples):
            student_class_year = np.random.randint(1, 5)
            previous_courses = self.generate_previous_courses(student_class_year)
            previous_courses_count = len(previous_courses)
            subjects_in_courses = [course_subject_mapping[course] for course in previous_courses if course in course_subject_mapping]
            unique_subjects_in_courses = len(set(subjects_in_courses))
            subjects_of_interest_list = random.sample(self.subjects_of_interest, np.random.randint(1, 5))
            subjects_diversity = len(subjects_of_interest_list)
            career_aspirations_list = random.sample(related_career_aspirations, np.random.randint(1, 3)) if related_career_aspirations else []
            extracurricular_activities = random.sample(extracurricular_activities_list, np.random.randint(0, 4))
            activities_involvement_count = len(extracurricular_activities)
            future_topics = random.sample(related_topics, np.random.randint(1, 5)) if related_topics else []
            gpa = round(np.random.uniform(2.0, 4.0), 2)
            
            data.append({
                'race_ethnicity': assign_demographic('race_ethnicity', 'real'),
                'gender': assign_demographic('gender', 'real'),
                'international': assign_demographic('international', 'real'),
                'socioeconomic status': assign_demographic('socioeconomic', 'real'),
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
        
        # Add noise to numerical features
        numerical_columns = ['previous courses count', 'unique subjects in courses', 'subjects diversity', 'activities involvement count', 'gpa']
        for col in numerical_columns:
            df = self.add_noise(df, col)
        
        # Encode categorical features
        categorical_columns = ['race_ethnicity', 'gender', 'international', 'socioeconomic status']
        df = self.encode_categorical_features(df, categorical_columns)
        
        # Normalize numerical features
        df = self.normalize_numerical_features(df, numerical_columns)
        
        return df

    def add_noise(self, dataframe, column_name, noise_level=0.01):
        """
        Add noise to numerical features for privacy.
        """
        noise = np.random.normal(0, noise_level, dataframe[column_name].shape)
        dataframe[column_name] += noise
        return dataframe

    def encode_categorical_features(self, df, categorical_columns):
        """
        Encode categorical features using one-hot encoding.
        """
        encoder = OneHotEncoder(sparse_output=False) if version.parse(sklearn.__version__) >= version.parse("1.2.0") else OneHotEncoder(sparse=False)
        encoded_features = encoder.fit_transform(df[categorical_columns])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))
        df = df.drop(columns=categorical_columns).reset_index(drop=True)
        df = pd.concat([df, encoded_df], axis=1)
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
