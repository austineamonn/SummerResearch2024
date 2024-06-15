import pandas as pd
import numpy as np
import random
import logging
import ast
import sys
import os
from collections import Counter
import multiprocessing as mp

# Add the SummerResearch2024 directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datafiles_for_data_construction.data import Data
from config import load_config

# Load configuration
config = load_config()

# Set up logging
logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])

class DataGenerator:
    def __init__(self, config):
        data = Data()
        self.config = config

        # First Names
        first_names_data = data.first_name()
        self.first_names = first_names_data['first_name']

        # Last Names
        last_names_data = data.last_name()
        self.last_names = last_names_data['last_name']

        # Ethnoracial Group
        ethnoracial_data = data.ethnoracial_group()
        self.ethnoracial_stats = ethnoracial_data['ethnoracial_stats']
        self.ethnoracial_to_activities = ethnoracial_data['ethnoracial_to_activities']
        self.ethnoracial_dist = config["synthetic_data"]["ethnoracial_group"]

        # Gender
        gender_data = data.gender()
        self.gender_stats = gender_data['gender']
        self.gender_to_activities = gender_data['gender_to_activities']
        self.gender_dist = config["synthetic_data"]["gender"]

        # International Student Status
        international_data = data.international_status()
        self.international_stats = international_data['international']
        self.international_to_activities = international_data['student_status_to_activities']
        self.international_dist = config["synthetic_data"]["international_status"]

        # Demographics to Extracurriculars - combine the 3 dictionaries into one
        self.demographics_to_activities = {k: v for d in (self.ethnoracial_to_activities, self.gender_to_activities, self.international_to_activities) for k, v in d.items()}

        # Socioeconomic Status
        SES_data = data.socioeconomics_status()
        self.socioeconomic_stats = SES_data['socioeconomic']
        self.socioeconomic_dist = config["synthetic_data"]["socioeconomic_status"]

        # Learning Styles
        LS_data = data.learning_style()
        self.learning_style_stats = LS_data['learning_style']
        self.learning_style_dist = config["synthetic_data"]["learning_style"]

        # Majors
        major_data = data.major()
        self.major_tuples = major_data['major_tuples']
        self.major_list = major_data['majors_list']
        self.major_to_activities = major_data['major_to_activities']
        self.major_to_course_subject = major_data['major_to_course_subject']
        self.majors_to_future_topics = major_data['majors_to_future_topics']

        # Courses
        course_data = data.course()
        self.course_tuples = course_data['course_tuples']
        self.course_subject_list = course_data['course_subject']
        self.course_subject_to_unabbreviated_subject = course_data['course_subject_to_unabbreviated_subject']
        self.course_subject_to_major = course_data['course_subject_to_major']
        self.course_type_to_learning_styles = course_data['course_type_to_learning_styles']

        # Subjects of Interest
        subject_data = data.subjects()
        self.subjects_list = subject_data['subjects_list']
        self.subjects_to_future_topics = subject_data['subjects_to_future_topics']
        self.subjects_to_careers = subject_data['subjects_to_careers']
        self.subjects_to_activtities = subject_data['subjects_to_activtities']

        # Careers
        career_data = data.careers()
        self.careers_list = career_data['careers_list']
        self.career_to_activities = career_data['career_to_activities']
        self.careers_to_subjects = career_data['careers_to_subjects']
        self.careers_to_future_topics = career_data['careers_to_future_topics']

        # Extracurricular Activities
        activities_data = data.extracurricular_activities()
        self.activity_list = activities_data['activity_list']
        self.career_based_activities = activities_data['career_based']
        self.identity_based_activities = activities_data['identity_based']
        self.identity_org_for_all = activities_data['identity_org_for_all']
        self.academic_and_honors_actitivities = activities_data['academic_and_honors']
        self.service_and_philanthropy_activities = activities_data['service_and_philanthropy']
        self.arts_and_culture_activities = activities_data['arts_and_culture']
        self.sports_and_recreation_actitivities = activities_data['sports_and_recreation']
        self.special_interest_activities = activities_data['special_interest']
        self.political_and_advocacy_activities = activities_data['political_and_advocacy']
        self.religious_and_spiritual_activities = activities_data['religious_and_spiritual']
        self.activities_to_future_topics = activities_data['activities_to_future_topics']
        self.activities_to_subjects = activities_data['activities_to_subjects']
        self.activities_to_careers = activities_data['activities_to_careers']
        
        # Future Topics
        future_topics_data = data.future_topics()
        self.future_topics_list = future_topics_data['future_topics']

        logging.debug("Data loaded.")

    def most_common_class_subject(self, previous_courses=None):
        if previous_courses is None:
            previous_courses = []

        # Extract the 'Type' elements from the tuples
        types = [course[2] for course in previous_courses if course[2] != "None"]

        # Count the occurrences of each 'Type'
        type_counter = Counter(types)

        # Find the most common 'Type', leave blank if no previous courses
        if type_counter:
            most_common_type, most_common_count = type_counter.most_common(1)[0]
        else:
            most_common_type = None
            most_common_count = 0

        return most_common_type, most_common_count

    def filter_course_by_learning_style(self, courses, learning_styles):
        learning_styles_set = set(learning_styles)
        filtered_ls_courses = [
            course for course in courses
            if learning_styles_set.intersection(self.course_type_to_learning_styles[course[2]])
        ]

        logging.debug(f"Filtered courses for learning styles {learning_styles}: {filtered_ls_courses}")
        return filtered_ls_courses

    def filter_course_by_major(self, courses, majors=None):
        if majors is None:
            majors = []

        majors_set = set(majors)
        filtered_mj_courses = [
            course for course in courses
            if majors_set.intersection(self.course_subject_to_major.get(course[3], []))
        ]

        logging.debug(f"Filtered courses for majors {majors}: {filtered_mj_courses}")
        return filtered_mj_courses

    def generate_previous_courses(self, semester, learning_styles, previous_courses=None, majors=None):
        """
        Generate a list of previous courses taken by a student based on their student semester, learning style, and major.
        """
        if previous_courses is None:
            previous_courses = []
        if majors is None:
            majors = []

        logging.debug(f"Generating previous courses for semester: {semester}, learning_styles: {learning_styles}")

        # Convert each course in course_tuples to a tuple to ensure hashability
        course_tuples_hashable = [tuple(course) for course in self.course_tuples]

        # Count the occurrences of each course in total_courses
        course_counter = Counter(course_tuples_hashable)

        # Convert each course in previous_courses to a tuple to ensure hashability
        previous_courses_hashable = [tuple(course) for course in previous_courses]

        # Remove the courses from total courses if the student has already taken them
        for course in previous_courses_hashable:
            if course in course_counter:
                del course_counter[course]

        logging.debug("Courses already taken were removed")

        def filter_courses(courses, level):
            return [course for course in courses if course[1] == level]

        # Convert course_counter keys to a list for further processing
        total_courses_list = list(course_counter.keys())

        # Most classes for the student should come from those at their level
        # Higher courses are for some students in their second semester of X year who can
        # take X+1 year courses
        # Lower courses are the breadth courses that students will take
        possible_courses = []
        possible_courses_lower = []
        possible_courses_higher = []

        if semester == 0:
            return possible_courses
        elif semester == 1 or semester == 2:
            possible_courses = filter_courses(total_courses_list, 1)
            if semester == 2:
                possible_courses_higher = filter_courses(total_courses_list, 2)
        elif semester == 3 or semester == 4:
            possible_courses = filter_courses(total_courses_list, 2)
            possible_courses_lower = filter_courses(total_courses_list, 1)
            if semester == 4:
                possible_courses_higher = filter_courses(total_courses_list, 3)
        elif semester == 5 or semester == 6:
            possible_courses = filter_courses(total_courses_list, 3)
            possible_courses_lower = filter_courses(total_courses_list, 1) + filter_courses(total_courses_list, 2)
            if semester == 6:
                possible_courses_higher = filter_courses(total_courses_list, 4)
        elif semester == 7 or semester == 8:
            possible_courses = filter_courses(total_courses_list, 4)
            possible_courses_lower = filter_courses(total_courses_list, 1) + filter_courses(total_courses_list, 2) + filter_courses(total_courses_list, 3)
            if semester == 8:
                possible_courses_higher = filter_courses(total_courses_list, 5)
        elif semester == 9 or semester == 10:
            possible_courses = filter_courses(total_courses_list, 5)
            possible_courses_lower = filter_courses(total_courses_list, 1) + filter_courses(total_courses_list, 2) + filter_courses(total_courses_list, 3)
            if semester == 10:
                possible_courses_higher = filter_courses(total_courses_list, 6)
        elif semester == 11 or semester == 12:
            possible_courses = filter_courses(total_courses_list, 6)
            possible_courses_lower = filter_courses(total_courses_list, 1) + filter_courses(total_courses_list, 2) + filter_courses(total_courses_list, 3)
            if semester == 12:
                possible_courses_higher = filter_courses(total_courses_list, 7)
        elif semester > 12:
            possible_courses = filter_courses(total_courses_list, 7)
            possible_courses_lower = filter_courses(total_courses_list, 1) + filter_courses(total_courses_list, 2) + filter_courses(total_courses_list, 3)

        # Filter the courses by learning style and majors
        possible_courses_ls = self.filter_course_by_learning_style(possible_courses, learning_styles)
        possible_courses_mj = self.filter_course_by_major(possible_courses, majors)

        possible_courses_higher_ls, possible_courses_higher_mj = [], []
        possible_courses_lower_ls, possible_courses_lower_mj = [], []

        if semester > 1:
            possible_courses_higher_ls = self.filter_course_by_learning_style(possible_courses_higher, learning_styles)
            possible_courses_higher_mj = self.filter_course_by_major(possible_courses_higher, majors)
            if semester > 2:
                possible_courses_lower_ls = self.filter_course_by_learning_style(possible_courses_lower, learning_styles)
                possible_courses_lower_mj = self.filter_course_by_major(possible_courses_lower, majors)

        # Pick 4 of the possible courses to take, mostly from the main group but some from the 'higher' and 'lower' groups
        classes_left = random.randint(2, 4)
        possible_courses = self.pick_classes_with_frequency(possible_courses, possible_courses_ls, possible_courses_mj, course_counter, classes_left)
        classes_left = 4 - len(possible_courses)
        possible_courses.extend(self.pick_classes_with_frequency(possible_courses_lower, possible_courses_lower_ls, possible_courses_lower_mj, course_counter, classes_left))
        classes_left = 4 - len(possible_courses)
        possible_courses.extend(self.pick_classes_with_frequency(possible_courses_higher, possible_courses_higher_ls, possible_courses_higher_mj, course_counter, classes_left))
        classes_left = 4 - len(possible_courses)

        # Add in classes to get to 4 most of the time
        while classes_left > 0:
            possible_courses.extend(self.pick_classes_with_frequency(possible_courses, possible_courses_ls, possible_courses_mj, course_counter, classes_left))
            classes_left = 4 - len(possible_courses)
            # 10% chance that students just take 3 classes
            if classes_left == 1 and np.random.rand() < 0.1:
                break

        # Adding the courses for the semester
        logging.debug(f"Courses for semester {semester}: {possible_courses}")
        previous_courses.extend(possible_courses)

        logging.debug(f"Previous courses taken: {previous_courses}")
        return previous_courses

    def filter_courses_by_number(self, courses, number):
        # Ensure courses is a list
        if not isinstance(courses, list):
            raise ValueError("Expected 'courses' to be a list")

        # Define the prefix for filtering
        prefix = str(number)

        # Filter and sort courses that match the specified prefix
        filtered_courses = sorted(
            (course for course in courses if str(course[1]).startswith(prefix)),
            key=lambda x: str(x[1])
        )

        logging.debug(f"Filtered courses {filtered_courses} for the number {number}")
        return filtered_courses
    
    def pick_classes(self, courses, course_ls, course_mj, sample_size=4):
        # Combine lists
        combined_list = courses

        # Return an empty list immediately if combined_list is empty
        if not combined_list:
            return []
        
        # Create Weights
        weights = []
        weight_ls = 2
        weight_mj = 5

        for item in combined_list:
            if item in course_mj: # Highest weight for major courses
                weights.append(weight_mj)
            elif item in course_ls:
                weights.append(weight_ls)  # Higher weight for learning style courses
            else:
                weights.append(1)  # Lower weight for all other courses

        # Sample items from the combined list with the specified weights
        sampled_items = self.weighted_sample_without_replacement(combined_list, weights, sample_size)
        
        return sampled_items

    def pick_classes_with_frequency(self, courses, filtered_ls_courses, filtered_mj_courses, course_counter, classes_left):
        weighted_courses = []
        for course in courses:
            if course in filtered_ls_courses and course in filtered_mj_courses:
                weighted_courses.extend([course] * course_counter[course])

        if not weighted_courses:
            return []

        selected_courses = random.sample(weighted_courses, min(classes_left, len(weighted_courses)))
        return selected_courses

    def weighted_sample_without_replacement(self, combined_list, weights, sample_size):
        if not combined_list or not weights:
            raise ValueError("Combined list and weights must not be empty")

        if len(combined_list) != len(weights):
            raise ValueError("Combined list and weights must have the same length")

        if sample_size > len(combined_list):
            raise ValueError("Sample size cannot be larger than the combined list")

        # Ensure weights are normalized to sum to 1
        weights_sum = np.sum(weights)
        if weights_sum == 0:
            raise ValueError("Sum of weights must not be zero")
            
        normalized_weights = np.array(weights) / weights_sum

        # Get the indices of the sampled elements
        sampled_indices = np.random.choice(len(combined_list), size=sample_size, replace=False, p=normalized_weights)

        # Retrieve the sampled items from the list
        sampled_items = [combined_list[i] for i in sampled_indices]

        return sampled_items

    def assign_demographic(self, demographic_type, bias='Uniform'):
        demographics = list(demographic_type.keys())
        
        if not demographics:
            raise ValueError("Demographic type must not be empty")

        if bias == 'real': # "real" uses the real statistical distributions
            probabilities = [demographic_type[demo] / 100 for demo in demographics]
            result = np.random.choice(demographics, p=probabilities)
        elif bias == 'Uniform': # "Uniform" uses uniform distributions
            result = np.random.choice(demographics)
        else:
            raise ValueError(f"Unexpected bias value: {bias}")

        return result

    def generate_learning_style(self):
        learning_style = [self.assign_demographic(self.learning_style_stats, self.learning_style_dist)]
        
        if np.random.rand() < 0.1:  # 10% chance to add an extra learning style
            extra_style = self.assign_demographic(self.learning_style_stats, self.learning_style_dist)
            if extra_style not in learning_style:
                learning_style.append(extra_style)
            else:
                logging.debug("Double up of learning style")

        return learning_style
    
    def generate_gpa(self, semester):
        gpa = None if semester == 0 else round(np.random.uniform(2.0, 4.0), 2)
        
        if gpa is None:
            logging.debug("GPA left empty because student semester is zero")
        else:
            logging.debug("GPA chosen: %.2f", gpa)
        
        return gpa

    def choose_major(self, semester, gender):
        # Weigh majors by gender
        major_weights = [
            (1 - float(major[2]) / 173) + (1 - float(major[1]) / 100 if gender == 'Male' else float(major[1]) / 100)
            for major in self.major_tuples
        ]
        
        if semester <= 4:
            if np.random.rand() < 0.3:  # Only 30% of underclassmen students have a major / intended major
                major = [random.choices(self.major_list, major_weights, k=1)[0]]
            else:
                major = []
                logging.debug("Major left empty because student has no major / intended major")
        else:
            major = [random.choices(self.major_list, major_weights, k=1)[0]]

        if len(major) == 1 and np.random.rand() < 0.3:  # Only 30% of students have a second major / intended major
            extra_major = random.choices(self.major_list, major_weights, k=1)[0]
            if extra_major not in major:
                major.append(extra_major)
            else:
                logging.debug("Double up of major")

        return major

    def generate_career_aspirations(self, careers=None, subjects=None, majors=None, activities=None):
        """
        Generate a list of career aspirations for a student based on their subjects of interest.
        """
        if careers is None:
            careers = []
        if subjects is None:
            subjects = []
        if majors is None:
            majors = []
        if activities is None:
            activities = []

        possible_careers = set()

        # Find all possible careers related to the subjects of interest
        for subject in subjects:
            if subject in self.subjects_to_careers:
                possible_careers.update(self.subjects_to_careers[subject])

        # Find top 5 careers most major(s) get after college
        for major in majors:
            for major_tuple in self.major_tuples:
                if major in major_tuple:
                    # Convert the string representation of the list to an actual list
                    careers_list = ast.literal_eval(major_tuple[4])
                    possible_careers.update(careers_list)

        # Find all possible careers related to extracurricular activities
        for activity in activities:
            if activity in self.activities_to_careers:
                possible_careers.update(self.activities_to_careers[activity])

        # Add a small chance of including a random career from all possible careers
        if np.random.rand() < 0.1:  # 10% chance
            possible_careers.add(np.random.choice(self.careers_list))

        possible_careers = list(possible_careers)

        # Select a random number of careers (between 0 and 4) from the possible careers
        selected_careers = random.sample(possible_careers, min(np.random.randint(0, 5), len(possible_careers)))

        # Add the new career aspirations to the list
        careers.extend(selected_careers)

        return careers

    def generate_extracurricular_activities(self, subjects=None, activities=None, majors=None, careers=None):
        """
        Inputs: subjects of interest, current activities list
        Output: new activity list
        """
        if subjects is None:
            subjects = []
        if activities is None:
            activities = []
        if majors is None:
            majors = []
        if careers is None:
            careers = []

        possible_activities = set()

        # Find extracurriculars related to subjects of interest
        for subject in subjects:
            value = self.subjects_to_activtities.get(subject)
            if value:
                logging.debug(f"Key {subject} exists in the dictionary with value {value}")
                if np.random.rand() < 0.3:  # 30% chance a person joins a club based on a subject of interest
                    possible_activities.update(value)
                    logging.debug(f"{value} added because of {subject}")

        # Find extracurriculars related to majors
        for major in majors:
            if major in self.major_to_activities:
                possible_activities.update(self.major_to_activities[major])

        # Find extracurriculars related to careers
        for career in careers:
            if career in self.career_to_activities:
                possible_activities.update(self.career_to_activities[career])

        # 10% chance a person joins a random club
        if np.random.rand() < 0.1:
            extra_club = random.choice(self.activity_list)
            possible_activities.add(extra_club)

        # Convert set to list before sampling
        possible_activities_list = list(possible_activities)

        # Randomly add 0-5 of the possible activities
        new_activities = random.sample(possible_activities_list, min(np.random.randint(0, 5), len(possible_activities_list)))
        activities.extend(new_activities)

        return activities
    
    def generate_identity_org(self, identities):
        """
        Inputs: current identities
        Output: activity list
        """
        possible_activities = set()

        for identity in identities:
            value = self.demographics_to_activities.get(identity)
            if value:
                logging.debug(f"Key {identity} exists in the dictionary with value {value}")
                if np.random.rand() < 0.2:  # 20% chance a person joins a club based on an identity
                    possible_activities.update(value)
                    logging.debug(f"{value} added because of {identity}")

        # Identity based clubs students can join regardless of identity
        if np.random.rand() < 0.05:
            extra_club = random.choice(self.identity_org_for_all)
            possible_activities.add(extra_club)
        
        # Convert set to list before sampling
        possible_activities_list = list(possible_activities)

        # Take a random sample (0, 3) of possible activities
        activities = random.sample(possible_activities_list, min(np.random.randint(0, 3), len(possible_activities_list)))

        return activities

    def generate_subjects_of_interest(self, previous_courses=None, subjects=None, majors=None, top_subject=None, careers=None, activities=None):
        if previous_courses is None:
            previous_courses = []
        if subjects is None:
            subjects = []
        if majors is None:
            majors = []
        if careers is None:
            careers = []
        if activities is None:
            activities = []

        possible_subjects = set()

        # Add in subjects of interest based on previous courses taken
        for course in previous_courses:
            logging.debug("Course being examined: %s", course)
            subject_abbr = course[3]
            if subject_abbr in self.course_subject_to_unabbreviated_subject:
                unabbreviated_subject = self.course_subject_to_unabbreviated_subject[subject_abbr]
                course_level = course[1]
                if 100 <= course_level < 200 and np.random.rand() < 0.3:
                    possible_subjects.add(unabbreviated_subject)
                    logging.debug("%s has been added to the list", unabbreviated_subject)
                elif 200 <= course_level < 300 and np.random.rand() < 0.6:
                    possible_subjects.add(unabbreviated_subject)
                    logging.debug("%s has been added to the list", unabbreviated_subject)
                elif course_level >= 300:
                    possible_subjects.add(unabbreviated_subject)
                    logging.debug("%s has been added to the list", unabbreviated_subject)
            else:
                logging.debug("this course subject %s was not in the list %s", subject_abbr, self.course_subject_to_unabbreviated_subject.keys())
        logging.debug("Initial subject list chosen: %s", possible_subjects)

        # Add in subjects of interest based on career aspirations
        for career in careers:
            possible_subjects.update(self.careers_to_subjects.get(career, []))
        logging.debug("Added in subjects based on career aspirations")

        # Find subjects related to extracurriculars
        for activity in activities:
            possible_subjects.update(self.activities_to_subjects.get(activity, []))

        # Add random subjects
        if np.random.rand() < 0.3:  # 30% chance to add extra subjects of interest
            extra_subjects = random.sample(self.subjects_list, min(np.random.randint(1, 3), len(self.subjects_list)))
            possible_subjects.update(extra_subjects)

        # Convert set to list before sampling
        possible_subjects_list = list(possible_subjects)

        # Randomly add 0-5 of the possible subjects of interest
        subjects.extend(random.sample(possible_subjects_list, min(np.random.randint(0, 5), len(possible_subjects_list))))

        # Add in all the subjects related to a person's major
        for major in majors:
            course_subject = self.major_to_course_subject.get(major)
            if course_subject:
                unabbreviated_subject = self.course_subject_to_unabbreviated_subject.get(course_subject)
                if unabbreviated_subject:
                    subjects.append(unabbreviated_subject)
                    logging.debug(f"{unabbreviated_subject} added because of {major}")

        # Add in the subjects related to the most common subject in courses taken
        if top_subject and top_subject in self.course_subject_to_unabbreviated_subject:
            subjects.append(self.course_subject_to_unabbreviated_subject[top_subject])

        return subjects
    
    def create_weighted_list(self, input_list):
        # Return empty lists immediately if input_list is empty
        if not input_list:
            return [], []

        # Use Counter to count occurrences of each element in the input list
        element_counts = Counter(input_list)

        # Create a list of unique elements and a list of corresponding weights
        unique_elements = list(element_counts.keys())
        weights = list(element_counts.values())

        return unique_elements, weights
    
    def generate_future_topics(self, subjects=None, subject_weights=None, activities=None, activity_weights=None, careers=None, career_weights=None, majors=None, major_weights=None):
        if subjects is None:
            subjects = []
        if subject_weights is None:
            subject_weights = []
        if activities is None:
            activities = []
        if activity_weights is None:
            activity_weights = []
        if careers is None:
            careers = []
        if career_weights is None:
            career_weights = []
        if majors is None:
            majors = []
        if major_weights is None:
            major_weights = []

        # Filter out elements with None weights and create combined lists and weights
        combined_list = []
        combined_weights = []
        
        for lst, wts in [(subjects, subject_weights), (activities, activity_weights), (careers, career_weights), (majors, major_weights)]:
            for element, weight in zip(lst, wts):
                if weight is not None:
                    combined_list.append(element)
                    combined_weights.append(weight)
        
        # Combine the list elements and their corresponding weights into a list of tuples
        combined = list(zip(combined_list, combined_weights))
        
        # Sort the combined list based on weights in descending order
        combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)
        
        future_topics = set()
        n = 1

        while len(future_topics) < 5 and n <= len(combined_sorted):
            top_n_elements = combined_sorted[:n]
            possible_topics = set()

            for element, weight in top_n_elements:
                # Add future topics for subjects of interest
                possible_topics.update(self.subjects_to_future_topics.get(element, []))
                
                # Add future topics for extracurriculars 20% of the time
                if element in self.activities_to_future_topics and np.random.rand() < 0.2:
                    possible_topics.update(self.activities_to_future_topics[element])
                
                # Add future topics for career aspirations
                possible_topics.update(self.careers_to_future_topics.get(element, []))
                
                # Add future topics for majors
                possible_topics.update(self.majors_to_future_topics.get(element, []))

            future_topics.update(possible_topics)
            n += 1
        
        # Ensure no more than 5 future topics are recommended
        future_topics = list(future_topics)
        future_topics = random.sample(future_topics, min(5, len(future_topics)))

        # Give 5 random future topics if the list is empty
        if not future_topics:
            future_topics = random.sample(self.future_topics_list, 5)

        return future_topics
    """
    def generate_single_sample(self):
        logging.debug("Generating single sample")
        first_name = random.choice(self.first_names)
        last_name = random.choice(self.last_names)
        ethnoracial_group = self.assign_demographic(self.ethnoracial_stats, self.ethnoracial_dist)
        gender = self.assign_demographic(self.gender_stats, self.gender_dist)
        international_status = self.assign_demographic(self.international_stats, self.international_dist)
        socioeconomic_status = self.assign_demographic(self.socioeconomic_stats, self.socioeconomic_dist)
        identities = [ethnoracial_group, gender, international_status]
        learning_style = self.generate_learning_style()
        student_semester = np.random.randint(0, 15)
        gpa = self.generate_gpa(student_semester)
        logging.debug(f"GPA chosen: {gpa}")
        major = self.choose_major(student_semester, gender)
        extracurricular_activities = self.generate_identity_org(identities)
        previous_courses = []
        subjects = []
        career_aspirations_list = []

        for semester in range(student_semester + 1):
            previous_courses = self.generate_previous_courses(semester, learning_style, previous_courses, major)
            num_courses = len(previous_courses)
            top_subject, class_count = self.most_common_class_subject(previous_courses)
            subjects = self.generate_subjects_of_interest(previous_courses, subjects, major, top_subject, career_aspirations_list)
            career_aspirations_list = self.generate_career_aspirations(career_aspirations_list, subjects, major, extracurricular_activities)
            extracurricular_activities = self.generate_extracurricular_activities(subjects, extracurricular_activities, major, career_aspirations_list)
            
            if semester >= 8 and num_courses >= 32 and class_count >= 8:
                student_semester = semester
                break

        course_names = [course[0] for course in previous_courses]
        course_type = list(set(course[2] for course in previous_courses))
        course_subject = list(set(course[3] for course in previous_courses))
        
        subjects, subject_weights = self.create_weighted_list(subjects)
        extracurricular_activities, activity_weights = self.create_weighted_list(extracurricular_activities)
        career_aspirations_list, career_weights = self.create_weighted_list(career_aspirations_list)
        
        major_weights = [student_semester * gpa if student_semester != 0 else 0] * len(major)
        
        future_topics = self.generate_future_topics(subjects, subject_weights, extracurricular_activities, activity_weights, career_aspirations_list, career_weights, major, major_weights)
        logging.debug("Single sample generated")
        return {
            'first name': first_name,
            'last name': last_name,
            'ethnoracial group': ethnoracial_group,
            'gender': gender,
            'international status': international_status,
            'socioeconomic status': socioeconomic_status,
            'learning style': learning_style,
            'gpa': gpa,
            'student semester': student_semester,
            'major': major,
            'previous courses': course_names,
            'course types': course_type,
            'course subjects': course_subject,
            'subjects of interest': subjects,
            'extracurricular activities': extracurricular_activities,
            'career aspirations': career_aspirations_list,
            'future topics': future_topics
        }"""

    def generate_single_sample(self):
        logging.debug("Generating single sample")
        gpa = None
        student_semester = random.randint(0, 8)  # Random semester for testing
        if student_semester != 0:
            gpa = round(random.uniform(2.0, 4.0), 2)
            logging.debug(f"GPA chosen: {gpa}")
        else:
            logging.debug("GPA left empty because student semester is zero")

        # Mocking the rest of the data generation for simplicity
        ethnoracial_group = "Latino/a/x American"
        gender = "Male"
        international_status = "International"
        socioeconomic_status = "Lower-middle income"
        learning_style = ['Read/Write']
        major = ['Film Video And Photographic Arts'] if student_semester > 1 else []
        previous_courses = []
        course_types = []
        course_subjects = []
        subjects_of_interest = ['Media and Cinema Studies']
        extracurricular_activities = ['Media Club', 'Anime Club', 'Photography Club', 'Film Society']
        career_aspirations = []
        future_topics = ['Art', 'Communications', 'Film Studies', 'Cultural Studies', 'Journalism']

        logging.debug("Single sample generated")
        return {
            'first name': 'Hana',
            'last name': 'Nennig',
            'ethnoracial group': ethnoracial_group,
            'gender': gender,
            'international status': international_status,
            'socioeconomic status': socioeconomic_status,
            'learning style': learning_style,
            'gpa': gpa,
            'student semester': student_semester,
            'major': major,
            'previous courses': previous_courses,
            'course types': course_types,
            'course subjects': course_subjects,
            'subjects of interest': subjects_of_interest,
            'extracurricular activities': extracurricular_activities,
            'career aspirations': career_aspirations,
            'future topics': future_topics
        }

    def generate_synthetic_dataset(self, num_samples=1000):
        """Generate synthetic dataset without multiprocessing."""
        logging.debug("Starting synthetic dataset generation")
        data = [self.generate_single_sample() for _ in range(num_samples)]
        logging.debug("Synthetic dataset generation completed")
        return pd.DataFrame(data)

# Main execution
if __name__ == "__main__":
    logging.debug("Script started")
    data_generator = DataGenerator(config)
    synthetic_data = data_generator.generate_synthetic_dataset(num_samples=100)
    logging.debug("Dataset generated")
    print(synthetic_data.head())


"""
Improvements:
Caching: from functools import lru_cache
Cut out Logging - really slows down code
Sets over lists - for checking membership it is much faster
List comprehension - much faster
Counter from Collections - used to count occurences
"""