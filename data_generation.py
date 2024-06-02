import pandas as pd
import numpy as np
import random
import logging
import re
from csv_loading import course_loader, fn_loader, ln_loader
from dictionary import Data
from config import load_config

# Load configuration
config = load_config()

# Set up logging
logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])

class DataGenerator:
    def __init__(self):
        data = Data(config)
        combined_data = data.get_data()
        self.subjects_of_interest = combined_data['subjects_of_interest']
        self.course_to_subject = combined_data['course_to_subject']
        self.related_career_aspirations = combined_data['related_career_aspirations']
        self.extracurricular_list = combined_data['extracurricular_list']
        self.related_topics = combined_data['related_topics']
        self.course_type_to_learning_styles = combined_data['course_type_to_learning_styles']
        self.race_ethnicity = combined_data['race_ethnicity']
        self.gender = combined_data['gender']
        self.international = combined_data['international']
        self.socioeconomic = combined_data['socioeconomic']
        self.learning_style = combined_data['learning_style']
        self.careers = combined_data['careers']
        logging.debug("Combined data loaded.")

        self.course_tuples = course_loader.course_tuples
        self.course_year_mapping = {course[0]: self.map_course_to_year(course) for course in self.course_tuples}
        logging.debug("Course information loaded.")

        self.first_names_list = fn_loader.first_names
        self.last_names_list = ln_loader.last_names
        logging.debug("First and last names loaded.")

    def generate_grades(self, previous_courses):
        """
        Simulate grades for previous courses.
        """
        grades = {course: np.random.choice(['A', 'B', 'C', 'D', 'F'], p=[0.3, 0.3, 0.2, 0.1, 0.1]) for course in previous_courses}
        return grades

    def filter_courses_by_year(self, courses, semester):
        """
        Filter courses based on student semester.
        """
        return [course for course in courses if semester in range(*self.course_year_mapping.get(course[0], (0, 0)))]

    def generate_previous_courses(self, semester, learning_styles):
        """
        Generate a list of previous courses taken by a student based on their student semester and their learning style.
        """
        logging.debug(f"Generating previous courses for semester: {semester}, learning_styles: {learning_styles}")
        previous_courses_taken = []

        for year in range(1, semester + 1):
            possible_courses = self.filter_courses_by_year(self.course_tuples, year)
            logging.debug(f"Possible courses for year {year}: {possible_courses}")


            filtered_courses = []
            for course in possible_courses:
                style = course[2]
                course_styles = self.course_type_to_learning_styles[style]
                for lstyle in learning_styles: # Allows for multiple learning styles
                    if lstyle in course_styles:
                        filtered_courses.append(course)

            logging.debug(f"Filtered courses for learning styles {learning_styles}: {filtered_courses}")

            # Select a number of courses to simulate the courses taken
            num_courses = max(0, min(4 + np.random.randint(-2, 2), len(filtered_courses)))  # Adding some randomness
            previous_courses_taken.extend(random.sample(filtered_courses, num_courses))

        logging.debug(f"Previous courses taken: {previous_courses_taken}")
        return previous_courses_taken


    def map_course_to_year(self, course_tuple):
        course_number = course_tuple[1]
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
                return (9, 12)  # Fifth and Sixth year: first and second semester
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
            logging.debug("First name chosen.")

            last_name = random.choice(self.last_names_list)
            logging.debug("Last name chosen.")

            student_semester = np.random.randint(1, 12)
            logging.debug("Student semester chosen chosen.")

            learning_style = [self.assign_demographic(self.learning_style, 'real')]
            if np.random.rand() < 0.1:  # 10% chance to add an extra learning style
                extra_style = self.assign_demographic(self.learning_style, 'real')
                learning_style.append(extra_style)
            logging.debug("Learning style chosen")

            previous_courses = self.generate_previous_courses(student_semester, learning_style)
            logging.debug("Previous courses chosen")

            previous_courses_count = len(previous_courses)
            subjects_in_courses = list(set([course[3] for course in previous_courses]))
            unique_subjects_in_courses = len(subjects_in_courses)
            logging.debug("Previous courses described by count, course subject, and course subject diversity")

            # Pick the subject of interest based on previous courses, with some random extra subjects added in
            subjects_of_interest_list = []
            for course in previous_courses:
                logging.debug("Course being examined: %s", course)
                if course[3] in self.course_to_subject.keys():
                    subjects_of_interest_list.append(self.course_to_subject[course[3]])
                    logging.debug("The couse has been added to the list")
                else:
                    logging.debug("this course subject %s", course[3])
                    logging.debug("was not in the list %s", self.course_to_subject.keys())
            logging.debug("Initial subject list chosen: %s", subjects_of_interest_list)

            if np.random.rand() < 0.3:  # 30% chance to add extra subjects of interest
                extra_subjects = random.sample(self.subjects_of_interest,  min(np.random.randint(1, 3), len(self.subjects_of_interest)))
                subjects_of_interest_list.extend(extra_subjects)

            subjects_of_interest_list = list(set(random.sample(subjects_of_interest_list,  min(np.random.randint(1, 5), len(subjects_of_interest_list)))))

            logging.debug("Subjects of interest chosen")

            subjects_diversity = len(subjects_of_interest_list)
            logging.debug("Subjects of interest described by diversity")

            career_aspirations_list = self.generate_career_aspirations(subjects_of_interest_list)
            logging.debug("Career aspirations chosen")

            extracurricular_activities = random.sample(self.extracurricular_list, min(np.random.randint(0, 4), len(self.extracurricular_list)))
            logging.debug("Extracurriculars chosen")

            activities_involvement_count = len(extracurricular_activities)
            logging.debug("Extracurriculars described by length.")

            future_topics = random.sample(list(self.related_topics.keys()), min(np.random.randint(1, 5), len(self.related_topics.keys()))) if self.related_topics else []
            logging.debug("Future topics chosen.")

            gpa = round(np.random.uniform(2.0, 4.0), 2)
            logging.debug("GPA chosen.")
            
            data.append({
                'first_name': first_name,
                'last_name': last_name,
                'race_ethnicity': self.assign_demographic(self.race_ethnicity, 'real'),
                'gender': self.assign_demographic(self.gender, 'real'),
                'international': self.assign_demographic(self.international, 'real'),
                'socioeconomic status': self.assign_demographic(self.socioeconomic, 'real'),
                'learning_style': learning_style,
                'gpa': gpa,
                'student semester': student_semester,

                # Split the tuples into course name and course type
                'previous courses': [course[0] for course in previous_courses],
                'course type': list(set([course[2] for course in previous_courses])),
                'previous courses count': previous_courses_count,
                'subjects in courses': subjects_in_courses,
                'unique subjects in courses': unique_subjects_in_courses,
                'subjects of interest': subjects_of_interest_list,
                'subjects diversity': subjects_diversity,
                'career aspirations': career_aspirations_list,
                'extracurricular activities': extracurricular_activities,
                'activities involvement count': activities_involvement_count,
                'future topics': future_topics
            })
            logging.debug("Data appended.")

        return pd.DataFrame(data)

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
