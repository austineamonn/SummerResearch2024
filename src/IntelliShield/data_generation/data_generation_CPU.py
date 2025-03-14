import pandas as pd
import numpy as np
import random
import logging
import ast
from collections import Counter
import json
from IntelliShield.data_generation.data import Data

class DataGenerator:
    def __init__(self, data: Data, logger = None, dist_dict: dict = None):
        # Set up logging
        self.logger = logger

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
        if dist_dict is not None:
            self.ethnoracial_dist = dist_dict["ethnoracial_group"]
        else:
            self.ethnoracial_dist = "real"

        # Gender
        gender_data = data.gender()
        self.gender_stats = gender_data['gender']
        self.gender_to_activities = gender_data['gender_to_activities']
        if dist_dict is not None:
            self.gender_dist = dist_dict["gender"]
        else:
            self.gender_dist = "real"

        # International Student Status
        international_data = data.international_status()
        self.international_stats = international_data['international']
        self.international_to_activities = international_data['student_status_to_activities']
        if dist_dict is not None:
            self.international_dist = dist_dict["international_status"]
        else:
            self.international_dist = "real"

        # Demographics to Extracurriculars - combine the 3 dictionaries into one
        self.demographics_to_activities = {k: v for d in (self.ethnoracial_to_activities, self.gender_to_activities, self.international_to_activities) for k, v in d.items()}

        # Socioeconomic Status
        SES_data = data.socioeconomics_status()
        self.socioeconomic_stats = SES_data['socioeconomic']
        if dist_dict is not None:
            self.socioeconomic_dist = dist_dict["socioeconomic_status"]
        else:
            self.socioeconomic_dist = "real"

        # Learning Styles
        LS_data = data.learning_style()
        self.learning_style_stats = LS_data['learning_style']
        if dist_dict is not None:
            self.learning_style_dist = dist_dict["learning_style"]
        else:
            self.learning_style_dist = "real"

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

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("Data loaded.")

    def most_common_class_subject(self, previous_courses=[]):
        """
        Determine the most common class subject from the list of previous courses.
        :param previous_courses: List of previous courses taken by the student.
        :return: The most common subject among the previous courses.
        """
        types = [course[3] for course in previous_courses if course[3] != "None"]
        type_counter = Counter(types)
        if len(previous_courses) > 0:
            most_common_type, most_common_count = type_counter.most_common(1)[0]
        else:
            most_common_type = None
            most_common_count = 0
        return most_common_type, most_common_count

    def filter_course_by_learning_style(self, courses, learning_styles):
        """
        Filter courses by learning styles.
        :param courses: List of all courses.
        :param learning_styles: Learning styles to filter by.
        :return: List of courses matching the learning styles.
        """
        filtered_ls_courses = []
        for course in courses:
            styles = course[2]  # Assuming course[2] is now a list of styles
            if not isinstance(styles, list):
                styles = [styles]  # Convert to list if it's a single element

            # Collect all learning styles for the given course styles
            course_styles = set()
            for style in styles:
                course_styles.update(self.course_type_to_learning_styles.get(style, []))

            # Check if any of the learning styles match
            for lstyle in learning_styles:
                if lstyle in course_styles:
                    filtered_ls_courses.append(course)
                    break  # Avoid adding the same course multiple times

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"Filtered courses for learning styles {learning_styles}: {filtered_ls_courses}")

        return filtered_ls_courses

    def filter_course_by_major(self, courses, majors=[]):
        filtered_mj_courses = []
        for course in courses:
            subject = course[3]
            course_subjects = self.course_subject_to_major.get(subject, [])
            for major in majors:
                if major in course_subjects:
                    filtered_mj_courses.append(course)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"Filtered courses for majors {majors}: {filtered_mj_courses}")
        return filtered_mj_courses

    def generate_previous_courses(self, semester, learning_styles, previous_courses=[], majors=[]):
        """
        Generate a list of previous courses based on the student's semester, learning styles, and majors.
        :param semester: The semester number of the student.
        :param learning_styles: The learning styles of the student.
        :param majors: The majors of the student.
        :return: List of previous courses taken by the student.
        """
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"Generating previous courses for semester: {semester}, learning_styles: {learning_styles}")
        # Initialize the course lists
        possible_courses = []
        possible_courses_lower = []
        possible_courses_higher = []
        total_courses = self.course_tuples

        for course in total_courses:
            if course in previous_courses:
                total_courses = [x for x in total_courses if x != course]

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("Courses already taken were removed")
        if semester == 0:
            return possible_courses
        elif semester == 1 or semester == 2:
            possible_courses = self.filter_courses_by_number(total_courses, 1)
            if semester == 2:
                possible_courses_higher = self.filter_courses_by_number(total_courses, 2)
        elif semester == 3 or semester == 4:
            possible_courses = self.filter_courses_by_number(total_courses, 2)
            possible_courses_lower = self.filter_courses_by_number(total_courses, 1)
            if semester == 4:
                possible_courses_higher = self.filter_courses_by_number(total_courses, 3)
        elif semester == 5 or semester == 6:
            possible_courses = self.filter_courses_by_number(total_courses, 3)
            possible_courses_lower = self.filter_courses_by_number(total_courses, 1)
            possible_courses_lower.extend(self.filter_courses_by_number(total_courses, 2))
            if semester == 6:
                possible_courses_higher = self.filter_courses_by_number(total_courses, 4)
        elif semester == 7 or semester == 8:
            possible_courses = self.filter_courses_by_number(total_courses, 4)
            possible_courses_lower = self.filter_courses_by_number(total_courses, 1)
            possible_courses_lower.extend(self.filter_courses_by_number(total_courses, 2))
            possible_courses_lower.extend(self.filter_courses_by_number(total_courses, 3))
            if semester == 8:
                possible_courses_higher = self.filter_courses_by_number(total_courses, 5)
        elif semester == 9 or semester == 10:
            possible_courses = self.filter_courses_by_number(total_courses, 5)
            possible_courses_lower = self.filter_courses_by_number(total_courses, 1)
            possible_courses_lower.extend(self.filter_courses_by_number(total_courses, 2))
            possible_courses_lower.extend(self.filter_courses_by_number(total_courses, 3))
            if semester == 10:
                possible_courses_higher = self.filter_courses_by_number(total_courses, 6)
        elif semester == 11 or semester == 12:
            possible_courses = self.filter_courses_by_number(total_courses, 6)
            possible_courses_lower = self.filter_courses_by_number(total_courses, 1)
            possible_courses_lower.extend(self.filter_courses_by_number(total_courses, 2))
            possible_courses_lower.extend(self.filter_courses_by_number(total_courses, 3))
            if semester == 12:
                possible_courses_higher = self.filter_courses_by_number(total_courses, 7)
        elif semester > 12:
            possible_courses = self.filter_courses_by_number(total_courses, 7)
            possible_courses_lower = self.filter_courses_by_number(total_courses, 1)
            possible_courses_lower.extend(self.filter_courses_by_number(total_courses, 2))
            possible_courses_lower.extend(self.filter_courses_by_number(total_courses, 3))
        possible_courses_ls = self.filter_course_by_learning_style(possible_courses, learning_styles)
        possible_courses_mj = self.filter_course_by_major(possible_courses, majors)
        if semester > 1:
            possible_courses_higher_ls = self.filter_course_by_learning_style(possible_courses_higher, learning_styles)
            possible_courses_higher_mj = self.filter_course_by_major(possible_courses_higher, majors)
            if semester > 2:
                possible_courses_lower_ls = self.filter_course_by_learning_style(possible_courses_lower, learning_styles)
                possible_courses_lower_mj = self.filter_course_by_major(possible_courses_lower, majors)
            else:
                possible_courses_lower_ls = []
                possible_courses_lower_mj = []
        else:
            possible_courses_higher_ls = []
            possible_courses_higher_mj = []
            possible_courses_lower_ls = []
            possible_courses_lower_mj = []
        classes_left = random.randint(2, 4)
        final_courses = self.pick_classes(possible_courses, possible_courses_ls, possible_courses_mj, classes_left)
        classes_left = 4 - len(final_courses)
        final_courses.extend(self.pick_classes(possible_courses_lower, possible_courses_lower_ls, possible_courses_lower_mj, classes_left))
        classes_left = 4 - len(final_courses)
        final_courses.extend(self.pick_classes(possible_courses_higher, possible_courses_higher_ls, possible_courses_higher_mj, classes_left))
        classes_left = 4 - len(final_courses)
        while classes_left > 0:
            for course in possible_courses:
                if course in final_courses:
                    possible_courses = [x for x in possible_courses if x != course]
            final_courses.extend(self.pick_classes(possible_courses, possible_courses_ls, possible_courses_mj, classes_left))
            classes_left = 4 - len(final_courses)
            if classes_left == 1 and np.random.rand() < 0.1:
                break
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"Courses for semester {semester}: {final_courses}")
        previous_courses.extend(final_courses)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"Previous courses taken: {previous_courses}")
        return previous_courses

    def filter_courses_by_number(self, courses, number):
        """
        Filter courses by a specific course number.
        :param courses: List of all courses.
        :param number: Course number to filter by.
        :return: List of courses matching the course number.
        """
        if not isinstance(courses, list):
            raise ValueError("Expected 'courses' to be a list")
        prefix = str(number)
        filtered_courses = [
            course for course in courses 
            if str(course[1]).startswith(prefix)
        ]
        filtered_courses.sort(key=lambda x: str(x[1]))
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"Filtered courses {filtered_courses} for the number {number}")
        return filtered_courses

    def pick_classes(self, courses, course_ls, course_mj, classes_left=4):
        """
        Pick classes from the filtered lists based on remaining classes needed.
        :param courses: List of all courses.
        :param courses_ls: List of courses filtered by learning style.
        :param courses_mj: List of courses filtered by major.
        :return: List of selected courses.
        """
        combined_list = [tuple(course) if isinstance(course, list) else course for course in courses]
        if not combined_list:
            return []
        
        weights = []
        weight_ls = 2
        weight_mj = 5
        
        for item in combined_list:
            base_weight = 1
            if item in course_mj:
                base_weight = weight_mj
            elif item in course_ls:
                base_weight = weight_ls
            
            # Increase the weight by the number of times the class appears in the courses list
            additional_weight = item[4]
            total_weight = base_weight + additional_weight
            weights.append(total_weight)
        
        # Create a list of tuples (item, weight)
        weighted_courses = list(zip(combined_list, weights))
        
        # Normalize weights for unique selection without replacement
        total_weight = sum(weight for item, weight in weighted_courses)
        normalized_weights = [weight / total_weight for item, weight in weighted_courses]
        
        k = min(classes_left, len(combined_list))
        
        # Use random.choices with unique selection constraint
        sampled_items = []
        seen_courses = set()
        for _ in range(k):
            if not weighted_courses:
                break

            selected = random.choices(weighted_courses, weights=normalized_weights, k=1)[0]
            selected_course_name = selected[0][0]  # Extract course name
            if selected_course_name in seen_courses:
                continue  # Skip if this course name has already been selected
            
            sampled_items.append(selected[0])
            seen_courses.add(selected_course_name)

            # Remove the selected item from the list (as well as any other courses with the same name) and update weights
            weighted_courses = [(item, weight) for item, weight in weighted_courses if item[0] != selected_course_name]
            total_weight = sum(weight for item, weight in weighted_courses)
            normalized_weights = [weight / total_weight for item, weight in weighted_courses]

        return sampled_items

    def assign_demographic(self, demographic_type, bias='Uniform'):
        """
        Assign demographic attributes to students based on the specified type and bias.
        :param demographic_type: Type of demographic attribute to assign (e.g., gender, ethnicity).
        :param bias: The distribution bias to use for assignment.
        :return: The assigned demographic value.
        """
        demographics = list(demographic_type.keys())
        if bias == 'real':
            probabilities = [demographic_type[demo] / 100 for demo in demographics]
            result = np.random.choice(demographics, p=probabilities)
        elif bias == 'uniform':
            result = np.random.choice(demographics)
        return result

    def generate_learning_style(self):
        """
        Generate a learning style for the student.
        :return: A learning style.
        """
        learning_style = [self.assign_demographic(self.learning_style_stats, self.learning_style_dist)]
        if np.random.rand() < 0.1:
            extra_style = self.assign_demographic(self.learning_style_stats, self.learning_style_dist)
            if extra_style not in learning_style:
                learning_style.append(extra_style)
        return learning_style
    
    def generate_gpa(self, semester):
        """
        Generate a GPA for the student based on the semester.
        :param semester: The semester number of the student.
        :return: A GPA value.
        """
        if semester == 0:
            gpa = None
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug("GPA left empty because student semester is zero")
        else:
            gpa = round(np.random.uniform(2.0, 4.0), 2)
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug("GPA chosen: %.2f", gpa)
        return gpa

    def choose_major(self, semester, gender):
        """
        Choose a major for the student based on the semester and gender.
        :param semester: The semester number of the student.
        :param gender: The gender of the student.
        :return: A major.
        """
        # Weigh majors by gender and popularity
        major_weights = [
            (1 - float(major[2]) / 173) + (1 - float(major[1]) / 100 if gender == 'Male' else float(major[1]) / 100)
            for major in self.major_tuples
        ]

        major = []
        if semester <= 4:
            if np.random.rand() < 0.3:
                major.append(random.choices(self.major_list, major_weights, k=1)[0])
            elif logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug("Major left empty because student has no major / intended major")
        else:
            major.append(random.choices(self.major_list, major_weights, k=1)[0])

        if len(major) == 1 and np.random.rand() < 0.3:
            extra_major = random.choices(self.major_list, major_weights, k=1)[0]
            if extra_major not in major:
                major.append(extra_major)
        
        return major

    def generate_career_aspirations(self, careers=[], subjects=[], majors=[], activities=[]):
        """
        Generate career aspirations for the student.
        :param careers: List of possible careers.
        :param subjects: List of subjects of interest.
        :return: A list of career aspirations.
        """
        
        possible_careers = []
        for subject in subjects:
            if subject in self.subjects_to_careers:
                possible_careers.extend(self.subjects_to_careers[subject])
        for major in majors:
            for major_tuple in self.major_tuples:
                if major in major_tuple:
                    careers_list = ast.literal_eval(major_tuple[4])
                    possible_careers.extend(careers_list)
        for activity in activities:
            if activity in self.activities_to_careers:
                possible_careers.extend(self.activities_to_careers[activity])
        if np.random.rand() < 0.1:
            possible_careers.append(np.random.choice(self.careers_list))
        possible_careers = random.sample(possible_careers, min(np.random.randint(0, 5), len(possible_careers)))
        careers.extend(possible_careers)
        return careers

    def generate_extracurricular_activities(self, subjects=[], activities=[], majors=[], careers=[]):
        """
        Generate extracurricular activities for the student.
        :param subjects: List of subjects of interest.
        :param activities: List of possible activities.
        :return: A list of extracurricular activities.
        """
        
        possible_activities = []
        for subject in subjects:
            value = self.subjects_to_activtities.get(subject)
            if value is not None:
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(f"Key {subject} exists in the dictionary with value {value}")
                if np.random.rand() < 0.3:
                    possible_activities.extend(value)
                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.debug(f"{value} added because of {subject}")
            elif logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(f"Key {subject} does not exist in the dictionary")
        for major in majors:
            if major in self.major_to_activities:
                possible_activities.extend(self.major_to_activities[major])
        for career in careers:
            if career in self.career_to_activities:
                possible_activities.extend(self.career_to_activities[career])
        if np.random.rand() < 0.1:
            extra_club = random.choice(self.activity_list)
            possible_activities.append(extra_club)
        activities.extend(random.sample(possible_activities, min(np.random.randint(0, 5), len(possible_activities))))
        return activities
    
    def generate_identity_org(self, identities):
        """
        Generate identity organizations for the student.
        :param identities: List of possible identities.
        :return: A list of identity organizations.
        """
        possible_activities = []
        for identity in identities:
            value = self.demographics_to_activities.get(identity)
            if value is not None:
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(f"Key {identity} exists in the dictionary with value {value}")
                if np.random.rand() < 0.2:
                    possible_activities.extend(value)
                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.debug(f"{value} added because of {identity}")
            elif logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(f"Key {identity} does not exist in the dictionary")
        if np.random.rand() < 0.05:
            extra_club = random.choice(self.identity_org_for_all)
            possible_activities.append(extra_club)
        activities = random.sample(possible_activities, min(np.random.randint(0,3), len(possible_activities)))
        return activities
    
    def generate_subjects_of_interest(self, previous_courses=[], subjects=[], majors=[], top_subject=None, careers=[], activities=[]):
        """
        Generate subjects of interest for the student based on previous courses and subject weights.
        :param previous_courses: List of previous courses taken by the student.
        :param subjects: The current list of subjects of interest.
        :return: An updated list of subjects of interest.
        """

        possible_subjects = []
        for course in previous_courses:
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug("Course being examined: %s", course)
            if course[3] in self.course_subject_to_unabbreviated_subject.keys():
                if course[1] < 300:
                    random_value = np.random.rand()
                    if 100 <= course[1] < 200:
                        if random_value < 0.3:
                            possible_subjects.append(self.course_subject_to_unabbreviated_subject[course[3]])
                            if logging.getLogger().isEnabledFor(logging.DEBUG):
                                logging.debug("%s has been added to the list", self.course_subject_to_unabbreviated_subject[course[3]])
                    else:
                        if random_value  < 0.6:
                            possible_subjects.append(self.course_subject_to_unabbreviated_subject[course[3]])
                            if logging.getLogger().isEnabledFor(logging.DEBUG):
                                logging.debug("%s has been added to the list", self.course_subject_to_unabbreviated_subject[course[3]])
                else:
                    possible_subjects.append(self.course_subject_to_unabbreviated_subject[course[3]])
                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.debug("%s has been added to the list", self.course_subject_to_unabbreviated_subject[course[3]])
            elif logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug("this course subject %s was not in the list %s", course[3], self.course_subject_to_unabbreviated_subject.keys())
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("Initial subject list chosen: %s", possible_subjects)
        for career in careers:
            if career in self.careers_to_subjects:
                subjects = self.careers_to_subjects.get(career, None)
                if subjects is not None:
                    possible_subjects.extend(subjects)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("Added in subjects based on career aspirations")
        for activity in activities:
            if activity in self.activities_to_subjects:
                possible_subjects.extend(self.activities_to_subjects[activity])
        if np.random.rand() < 0.3:
            extra_subjects = random.sample(self.subjects_list,  min(np.random.randint(1, 3), len(self.subjects_list)))
            possible_subjects.extend(extra_subjects)
        subjects.extend(random.sample(possible_subjects, min(np.random.randint(0, 5), len(possible_subjects))))
        for major in majors:
            value = self.major_to_course_subject.get(major)
            if value:
                unabbreviated_subject = self.course_subject_to_unabbreviated_subject.get(value)
                if unabbreviated_subject:
                    subjects.extend([unabbreviated_subject])
                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.debug(f"{unabbreviated_subject} added because of {major}")
            elif logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(f"Key {major} does not exist in the dictionary")
        if top_subject is not None:
            if top_subject in self.course_subject_to_unabbreviated_subject:
                subjects.append(self.course_subject_to_unabbreviated_subject[top_subject])
        return subjects
    
    def create_weighted_list(self, input_list):
        """
        Create a weighted list from the input list.
        :param input_list: The input list to create weights for.
        :return: A weighted list.
        """
        if not input_list:
            # Return an empty list immediately if input_list is empty
            return [], []
        # Use Counter to count occurrences of each element in the input list
        element_counts = Counter(input_list)

        # Create a list of unique elements and their corresponding weights
        unique_elements = list(element_counts.keys())
        weights = list(element_counts.values())
        return unique_elements, weights
    
    def generate_future_topics(self, subjects=[], subject_weights=[], activities=[], activity_weights=[], careers=[], career_weights=[], majors=[], major_weights=[]):
        """
        Generate future topics of interest for the student based on subjects of interest
        extracurriculars, and career aspirations as well as their associated weights.
        :param subjects: List of subjects of interest.
        :param subject_weights: Weights for each subject.
        :param activities: List of extracurriculars.
        :param activity_weights: Weights for each extracurricular.
        :param careers: List of career aspirations.
        :param career_weights: Weights for each career aspiration.
        :param majors: List of majors.
        :param major_weights: Weights for each major.
        :return: A list of 5 future topics.
        """
       
        combined_list = []
        combined_weights = []
        for lst, wts in [(subjects, subject_weights), (activities, activity_weights), (careers, career_weights), (majors, major_weights)]:
            for element, weight in zip(lst, wts):
                if weight is not None:
                    combined_list.append(element)
                    combined_weights.append(weight)
        combined = list(zip(combined_list, combined_weights))
        combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)
        n = 1
        future_topics = []
        while len(future_topics) < 5:
            top_n_elements = combined_sorted[:n]
            possible_topics = []
            for element, weight in top_n_elements:
                if element in self.subjects_to_future_topics:
                    possible_topics.extend(self.subjects_to_future_topics[element])
                if element in self.activities_to_future_topics:
                    if np.random.rand() < 0.2:
                        possible_topics.extend(self.activities_to_future_topics[element])
                if element in self.careers_to_future_topics:
                    possible_topics.extend(self.careers_to_future_topics[element])
                if element in self.majors_to_future_topics:
                    possible_topics.extend(self.majors_to_future_topics[element])
            future_topics = list(set(possible_topics))
            n += 1
            if n > len(combined_sorted):
                break
        future_topics = random.sample(future_topics, min(5, len(future_topics)))
        if len(future_topics) == 0:
            future_topics = random.sample(self.future_topics_list, 5)
        return future_topics

    def generate_single_sample(self):
        """
        Generate a single sample of synthetic data.
        :return: Dictionary representing a single data sample.
        """
        try:
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
            major = self.choose_major(student_semester, gender)
            extracurricular_activities = self.generate_identity_org(identities)

            previous_courses = []
            subjects = []
            career_aspirations_list = []

            # Iterate through each semester the student has been at the school.
            # Courses, subjects or interest, career aspirations and extracurriculars
            # all interact with one another.
            for semester in range(student_semester + 1):
                previous_courses = self.generate_previous_courses(semester, learning_style, previous_courses, major)
                num_courses = len(previous_courses)
                top_subject, class_count = self.most_common_class_subject(previous_courses)
                subjects = self.generate_subjects_of_interest(previous_courses, subjects, major, top_subject, career_aspirations_list)
                career_aspirations_list = self.generate_career_aspirations(career_aspirations_list, subjects, major, extracurricular_activities)
                extracurricular_activities = self.generate_extracurricular_activities(subjects, extracurricular_activities, major, career_aspirations_list)

                # If the student has been there for 4 or more semesters and has enough courses,
                # including those in their major then make them graduate by breaking the loop
                if semester >= 8 and num_courses >= 32 and class_count >= 8:
                    student_semester = semester
                    break

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug("Iterated through %s semesters", student_semester)

            # Split the tuples into course name, course type, and course subject
            # Ex: Intro Asian American Studies, Discussion/Recitation, AAS
            course_names = [course[0] for course in previous_courses]
            course_type = list(set([tuple(course[2]) for course in previous_courses]))
            course_subject = list(set([course[3] for course in previous_courses]))

            # Create weighted lists based on how often an element was in a list
            subjects, subject_weights = self.create_weighted_list(subjects)
            extracurricular_activities, activity_weights = self.create_weighted_list(extracurricular_activities)
            career_aspirations_list, career_weights = self.create_weighted_list(career_aspirations_list)

            # Majors are weighted by semester and gpa
            major_weights = [student_semester * gpa if student_semester != 0 else 0] * len(major)

            # Generate the future topics
            future_topics = self.generate_future_topics(subjects, subject_weights, extracurricular_activities, activity_weights, career_aspirations_list, career_weights, major, major_weights)

            # Create the new 'student'
            sample_data = {
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
            }

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug("Single sample generated")
            return sample_data
        except Exception as e:
            logging.error(f"Error generating sample: {e}")
            return None

    def generate_batch_samples(self, batch_size):
        samples = []
        for _ in range(batch_size):
            # Re-seed random number generators
            random.seed()

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(f"Random seed set for batch: {random.randint(0, 1e9)}")
            
            sample = self.generate_single_sample()
            if sample is not None:
                samples.append(sample)
        return samples

    def generate_synthetic_dataset(self, num_samples=1000, batch_size=100):
        """
        Generate a synthetic dataset with the specified number of samples.
        :param num_samples: Number of samples to generate.
        :return: DataFrame containing the generated dataset.
        """
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("Starting synthetic dataset generation")

        # Calculate the number of batches
        num_batches = num_samples // batch_size
        remainder = num_samples % batch_size

        data = []

        # Generate batches
        for batch_num in range(num_batches):
            batch_samples = self.generate_batch_samples(batch_size)
            batch_serialized = [json.dumps(sample) for sample in batch_samples]
            logging.debug(f"Generated batch {batch_num+1} samples: {batch_serialized[:5]}")  # Log first 5 samples for verification
            data.extend([json.loads(sample) for sample in batch_serialized])

        # Handle remainder
        if remainder > 0:
            remainder_samples = self.generate_batch_samples(remainder)
            remainder_serialized = [json.dumps(sample) for sample in remainder_samples]
            logging.debug(f"Generated remainder samples: {remainder_serialized[:5]}")  # Log first 5 samples for verification
            data.extend([json.loads(sample) for sample in remainder_serialized])

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("Synthetic dataset generation completed")
            
        return pd.DataFrame(data)

# Main execution
if __name__ == "__main__":
    import cProfile
    import pstats
    # Import necessary dependencies
    from IntelliShield.logger import setup_logger

    # Load logger and data
    logger = setup_logger('data_generation_CPU_logger', 'data_generation_CPU.log')
    data = Data()

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Script started")
    
    # Initialize data generator class
    data_generator = DataGenerator(data, logger)

    # Get config values
    num_samples = 1000
    batch_size = 100
    rewrite = False
    data_path = 'Dataset.csv'

    profiler = cProfile.Profile()
    profiler.enable()

    synthetic_data = data_generator.generate_synthetic_dataset(num_samples, batch_size)

    profiler.disable()

    # Save the profiling stats to a file
    profile_stats_file = "profile_stats.txt"
    with open(profile_stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats()

    logger.info("Profiling stats saved to %s", profile_stats_file)

    if logger.isEnabledFor(logging.DEBUG):
        logging.debug("Dataset generated")
    print(synthetic_data.head())

    # Rewrite the file or add to it
    if rewrite:
        synthetic_data.to_csv(data_path, index=False)
        logger.info("Synthetic dataset saved to Dataset.csv")
    else:
        synthetic_data.to_csv(data_path, mode='a', header=False, index=False)
        logger.info("New synthetic data added to Dataset.csv")