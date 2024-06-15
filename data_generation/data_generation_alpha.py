import pandas as pd
import numpy as np
import random
import logging
import ast
from collections import Counter
from datafiles_for_data_construction.data import Data
from config import load_config

# Load configuration
config = load_config()

# Set up logging
logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])

class DataGenerator:
    def __init__(self):
        data = Data()

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

        # Gender
        gender_data = data.gender()
        self.gender_stats = gender_data['gender']
        self.gender_to_activities = gender_data['gender_to_activities']

        # International Student Status
        international_data = data.international_status()
        self.international_stats = international_data['international']
        self.international_to_activities = international_data['student_status_to_activities']

        # Demographics to Extracurriculars - combine the 3 dictionaries into one
        self.demographics_to_activities = {k: v for d in (self.ethnoracial_to_activities, self.gender_to_activities, self.international_to_activities) for k, v in d.items()}

        # Socioeconomic Status
        SES_data = data.socioeconomics_status()
        self.socioeconomic_stats = SES_data['socioeconomic']

        # Learning Styles
        LS_data = data.learning_style()
        self.learning_style_stats = LS_data['learning_style']

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

    def most_common_class_subject(self, previous_courses=[]):
        # Extract the 'Type' elements from the tuples
        types = [course[2] for course in previous_courses if course[2] != "None"]

        # Count the occurrences of each 'Type'
        type_counter = Counter(types)

        # Find the most common 'Type', leave blank if no previous courses
        if len(previous_courses) > 0:
            most_common_type, most_common_count = type_counter.most_common(1)[0]
        else:
            most_common_type = None
            most_common_count = 0

        return most_common_type, most_common_count

    def filter_course_by_learning_style(self, courses, learning_styles):
        # Filter possible courses for learning style
        filtered_ls_courses = []
        for course in courses:
            style = course[2]
            course_styles = self.course_type_to_learning_styles[style]
            for lstyle in learning_styles: # Allows for multiple learning styles
                if lstyle in course_styles:
                    filtered_ls_courses.append(course)

        logging.debug(f"Filtered courses for learning styles {learning_styles}: {filtered_ls_courses}")
        return filtered_ls_courses

    def filter_course_by_major(self, courses, majors=[]):
        # Filter possible courses for majors
        filtered_mj_courses = []
        for course in courses:
            subject = course[3]
            course_subjects = self.course_subject_to_major.get(subject, [])
            for major in majors: # Allows for multiple majors
                if major in course_subjects:
                    filtered_mj_courses.append(course)

        logging.debug(f"Filtered courses for majors {majors}: {filtered_mj_courses}")
        return filtered_mj_courses

    def generate_previous_courses(self, semester, learning_styles, previous_courses=[], majors=[]):
        """
        Generate a list of previous courses taken by a student based on their student semester, learning style, and major.
        """
        logging.debug(f"Generating previous courses for semester: {semester}, learning_styles: {learning_styles}")
        possible_courses = []
        possible_courses_lower = []
        possible_courses_higher = []
        total_courses = self.course_tuples

        # Remove the courses from total courses if the student has already taken them.
        for course in total_courses:
            if course in previous_courses:
                total_courses = [x for x in total_courses if x != course]

        logging.debug("Courses already taken were removed")

        # Most classes for the student should come from those at their level
        # Higher courses are for some students in their second semester of X year who can
        # take X+1 year courses
        # Lower courses are the breadth courses that students will take
        if semester == 0:
            # Return an empty list for students who are not yet in college
            return possible_courses
        elif semester == 1 or semester == 2:
            possible_courses = self.filter_courses_by_number(total_courses, 1)
            if semester == 2:
                possible_courses_higher =self.filter_courses_by_number(total_courses, 2)
        elif semester == 3 or semester == 4:
            possible_courses = self.filter_courses_by_number(total_courses, 2)
            possible_courses_lower = self.filter_courses_by_number(total_courses, 1)
            if semester == 4:
                possible_courses_higher =self.filter_courses_by_number(total_courses, 3)
        elif semester == 5 or semester == 6:
            possible_courses = self.filter_courses_by_number(total_courses, 3)
            possible_courses_lower = self.filter_courses_by_number(total_courses, 1)
            possible_courses_lower.extend(self.filter_courses_by_number(total_courses, 2))
            if semester == 6:
                possible_courses_higher =self.filter_courses_by_number(total_courses, 4)
        elif semester == 7 or semester == 8:
            possible_courses = self.filter_courses_by_number(total_courses, 4)
            possible_courses_lower = self.filter_courses_by_number(total_courses, 1)
            possible_courses_lower.extend(self.filter_courses_by_number(total_courses, 2))
            possible_courses_lower.extend(self.filter_courses_by_number(total_courses, 3))
            if semester == 8:
                possible_courses_higher =self.filter_courses_by_number(total_courses, 5)
        elif semester == 9 or semester == 10:
            possible_courses = self.filter_courses_by_number(total_courses, 5)
            possible_courses_lower = self.filter_courses_by_number(total_courses, 1)
            possible_courses_lower.extend(self.filter_courses_by_number(total_courses, 2))
            possible_courses_lower.extend(self.filter_courses_by_number(total_courses, 3))
            if semester == 10:
                possible_courses_higher =self.filter_courses_by_number(total_courses, 6)
        elif semester == 11 or semester == 12:
            possible_courses = self.filter_courses_by_number(total_courses, 6)
            possible_courses_lower = self.filter_courses_by_number(total_courses, 1)
            possible_courses_lower.extend(self.filter_courses_by_number(total_courses, 2))
            possible_courses_lower.extend(self.filter_courses_by_number(total_courses, 3))
            if semester == 12:
                possible_courses_higher =self.filter_courses_by_number(total_courses, 7)
        elif semester > 12:
            possible_courses = self.filter_courses_by_number(total_courses, 7)
            possible_courses_lower = self.filter_courses_by_number(total_courses, 1)
            possible_courses_lower.extend(self.filter_courses_by_number(total_courses, 2))
            possible_courses_lower.extend(self.filter_courses_by_number(total_courses, 3))

        # Filter the courses by learning style and majors
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
                possible_courses_lower_mj =[]
        else:
            possible_courses_higher_ls = []
            possible_courses_higher_mj = []
            possible_courses_lower_ls = []
            possible_courses_lower_mj =[]

        # Pick 4 of the possible courses to take, mostly from the main group
        # but some from the 'higher' and 'lower' groups
        classes_left = random.randint(2, 4)
        possible_courses = self.pick_classes(possible_courses, possible_courses_ls, possible_courses_mj, classes_left)
        classes_left = 4 - len(possible_courses)
        possible_courses.extend(self.pick_classes(possible_courses_lower, possible_courses_lower_ls, possible_courses_lower_mj, classes_left))
        classes_left = 4 - len(possible_courses)
        possible_courses.extend(self.pick_classes(possible_courses_higher, possible_courses_higher_ls, possible_courses_higher_mj, classes_left))
        classes_left = 4 - len(possible_courses)

        # Add in classes to get to 4 most of the time
        while classes_left > 0:
            possible_courses.extend(self.pick_classes(possible_courses, possible_courses_ls, possible_courses_mj, classes_left))
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

        # Filter courses that match the specified prefix
        filtered_courses = [
            course for course in courses 
            if str(course[1]).startswith(prefix)
        ]

        # Sort the filtered courses by 'Number'
        filtered_courses.sort(key=lambda x: str(x[1]))

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
        k = min(sample_size, len(combined_list))
        sampled_items = random.choices(combined_list, weights=weights, k=k)
        
        return sampled_items

    def assign_demographic(self, demographic_type, bias='Uniform'):
        demographics = list(demographic_type.keys())
        # "real" uses the real statistical distributions
        if bias == 'real':
            probabilities = [demographic_type[demo] / 100 for demo in demographics]
            result = np.random.choice(demographics, p=probabilities)
        # "Uniform" uses uniform distributions - Assumes "Uniform" if no bias given
        elif bias == 'Uniform':
            result = np.random.choice(demographics)

        return result

    def generate_learning_style(self):
        learning_style = [self.assign_demographic(self.learning_style_stats, config["synthetic_data"]["learning_style"])]
        if np.random.rand() < 0.1:  # 10% chance to add an extra learning style
            extra_style = self.assign_demographic(self.learning_style_stats, config["synthetic_data"]["learning_style"])
            if extra_style in learning_style:
                logging.debug("Double up of learning style")
            else: # Only add the extra learning style if it is different than the original one
                learning_style.append(extra_style)

        return learning_style
    
    def generate_gpa(self, semester):
        if semester == 0:
            gpa = None
            logging.debug("GPA left empty because student semester is zero")
        else:
            gpa = round(np.random.uniform(2.0, 4.0), 2)
            logging.debug("GPA chosen: %.2f", gpa)

        return gpa

    def choose_major(self, semester, gender):
        # Weigh majors by gender
        major_weights = []
        for major in self.major_tuples:
            ranking = 1-(float(major[2])/173) # There are 173 majors
            # Assume nonmale
            percent = float(major[1])/100
            # If male change the percentage
            if gender == 'Male':
                percent = 1-(percent)
            # Weigh the major by the gender percentage and the popularity ranking
            weight = percent + ranking
            major_weights.append(weight)
        
        if semester <=4:
            if np.random.rand() < 0.3: # Only 30% of underclassmen students have a major / intended major
                major = random.choices(self.major_list, major_weights, k=1)[0]
            else:
                major = []
                logging.debug("Major left empty because student has no major / intended major")
        elif semester > 4:
            major = [random.choices(self.major_list, major_weights, k=1)[0]]
        
        if len(major) == 1:
            if np.random.rand() < 0.3: # Only 30% of students have a second major / intended major
                extra_major = random.choices(self.major_list, major_weights, k=1)[0]
                if extra_major in major:
                    logging.debug("Double up of major")
                else: # Only add the extra major if it is different than the original one
                    major.append(extra_major)

        return major

    def generate_career_aspirations(self, careers=[], subjects=[], majors=[], activities=[]):
        """
        Generate a list of career aspirations for a student based on their subjects of interest.
        """
        possible_careers = []
        # Find all possible careers related to the subjects of interest
        for subject in subjects:
            if subject in self.subjects_to_careers:
                possible_careers.extend(self.subjects_to_careers[subject])

        # Find top 5 careers most major(s) get after college
        for major in majors:
            for major_tuple in self.major_tuples:
                if major in major_tuple:
                    # Convert the string representation of the list to an actual list
                    careers_list = ast.literal_eval(major_tuple[4])
                    possible_careers.extend(careers_list)  # Use extend to add individual careers

        # Find all possible careers related to extracurricular activities
        for activity in activities:
            if activity in self.activities_to_careers:
                possible_careers.extend(self.activities_to_careers[activity])

        # Add a small chance of including a random career from all possible careers
        if np.random.rand() < 0.1:  # 10% chance
            possible_careers.append(np.random.choice(self.careers_list))

        # Select a random number of careers (between 0 and 4) from the possible careers
        possible_careers = random.sample(possible_careers, min(np.random.randint(0, 5), len(possible_careers)))

        # Add the new career aspirations to the list
        careers.extend(possible_careers)

        return careers

    def generate_extracurricular_activities(self, subjects=[], activities=[], majors=[], careers=[]):
        """
        Inputs: subjects of interest, current activities list
        Output: new activity list
        """
        possible_activities = []

        # Find extracurriculars related to subjects of interest
        for subject in subjects:
            value = self.subjects_to_activtities.get(subject)
            if value is not None:
                logging.debug(f"Key {subject} exists in the dictionary with value {value}")
                if np.random.rand() < 0.3: # 30% chance a person joins a club based on a subject of interest
                    possible_activities.extend(value)
                    logging.debug(f"{value} added because of {subject}")
            else:
                logging.debug(f"Key {subject} does not exist in the dictionary")

        # Find extracurriculars related to majors
        for major in majors:
            if major in self.major_to_activities:
                possible_activities.extend(self.major_to_activities[major])

        # Find extracurriculars related to careers
        for career in careers:
            if career in self.career_to_activities:
                possible_activities.extend(self.career_to_activities[career])

        # 10% chance a person joins a random club
        if np.random.rand() < 0.1:
            extra_club = random.choice(self.activity_list)
            possible_activities.append(extra_club)

        # Randomly add 0-5 of the possible activities and ensure no duplicates
        activities.extend(random.sample(possible_activities, min(np.random.randint(0, 5), len(possible_activities))))

        return activities
    
    def generate_identity_org(self, identities):
        """
        Inputs: current identities
        Output: activity list
        """
        possible_activities = []

        for identity in identities:
            value = self.demographics_to_activities.get(identity)
            if value is not None:
                logging.debug(f"Key {identity} exists in the dictionary with value {value}")
                if np.random.rand() < 0.2: # 20% chance a person joins a club based on an identity
                    possible_activities.extend(value)
                    logging.debug(f"{value} added because of {identity}")
            else:
                logging.debug(f"Key {identity} does not exist in the dictionary")

        # Identity based clubs students can join regardless of identity
        if np.random.rand() < 0.05:
            extra_club = random.choice(self.identity_org_for_all)
            possible_activities.append(extra_club)

        # Take a random sample (0, 3) of possible activities
        activities = random.sample(possible_activities, min(np.random.randint(0,3), len(possible_activities)))

        return activities
    
    def generate_subjects_of_interest(self, previous_courses=[], subjects=[], majors=[], top_subject=None, careers=[], activities=[]):
        possible_subjects = []

        # Add in subjects of interest based on previous courses taken
        for course in previous_courses:
            logging.debug("Course being examined: %s", course)
            if course[3] in self.course_subject_to_unabbreviated_subject.keys():
                    # 100 level courses indicate a 30% interest in the subject
                    if 100 <= course[1] < 200:
                        if np.random.rand() < 0.3:
                            possible_subjects.append(self.course_subject_to_unabbreviated_subject[course[3]])
                            logging.debug("%s has been added to the list", self.course_subject_to_unabbreviated_subject[course[3]])
                    # 200 level courses indicate a 60% interest in the subject
                    elif 200 <= course[1] < 300:
                        if np.random.rand() < 0.6:
                            possible_subjects.append(self.course_subject_to_unabbreviated_subject[course[3]])
                            logging.debug("%s has been added to the list", self.course_subject_to_unabbreviated_subject[course[3]])
                    # By the 300 level you are committed to the subject
                    elif 300 <= course[1]:
                        possible_subjects.append(self.course_subject_to_unabbreviated_subject[course[3]])
                        logging.debug("%s has been added to the list", self.course_subject_to_unabbreviated_subject[course[3]])
            else:
                logging.debug("this course subject %s was not in the list %s", course[3], self.course_subject_to_unabbreviated_subject.keys())
        logging.debug("Initial subject list chosen: %s", possible_subjects)

        # Add in subjects of interest based on career aspirations
        for career in careers:
            if career in self.careers_to_subjects:
                subjects = self.careers_to_subjects.get(career, None)
                if subjects is not None:
                    possible_subjects.extend(subjects)
        logging.debug("Added in subjects based on career aspirations")

        # Find subjects related to extracurriculars
        for activity in activities:
            if activity in self.activities_to_subjects:
                possible_subjects.extend(self.activities_to_subjects[activity])

        # Add random subjects
        if np.random.rand() < 0.3:  # 30% chance to add extra subjects of interest
            extra_subjects = random.sample(self.subjects_list,  min(np.random.randint(1, 3), len(self.subjects_list)))
            possible_subjects.extend(extra_subjects)

        # Randomly add 0-5 of the possible subjects of interest
        subjects.extend(random.sample(possible_subjects, min(np.random.randint(0, 5), len(possible_subjects))))
        
        # Add in all the subjects related to a person's major
        for major in majors:
            value = self.major_to_course_subject.get(major)
            if value is not None:
                # Map the subject abbreviation to the related subject (if there is one)
                unabbreviated_subject = self.course_subject_to_unabbreviated_subject.get(value, None)
                if unabbreviated_subject != None:
                    subjects.extend([unabbreviated_subject])
                    logging.debug(f"{unabbreviated_subject} added because of {major}")
            else:
                logging.debug(f"Key {major} does not exist in the dictionary")

        # Add in the subjects related to the most common subject in courses taken
        if top_subject is not None:
            if top_subject in self.course_subject_to_unabbreviated_subject:
                subjects.append(self.course_subject_to_unabbreviated_subject[top_subject])

        return subjects
    
    def create_weighted_list(self, input_list):
        # Return an empty list immediately if input_list is empty
        if not input_list:
            return [], []
        
        # Count occurrences of each element in the input list
        element_counts = {}
        for element in input_list:
            if element in element_counts:
                element_counts[element] += 1
            else:
                element_counts[element] = 1

        # Create a list of unique elements
        unique_elements = list(element_counts.keys())

        # Create a list of corresponding weights
        weights = list(element_counts.values())

        return unique_elements, weights
    
    def generate_future_topics(self, subjects=[], subject_weights=[], activities=[], activity_weights=[], careers=[], career_weights=[], majors=[], major_weights=[]):
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
        
        # Initally set n to 5 and future topics as empty
        n = 1
        future_topics = []

        while len(future_topics) < 5:
            top_n_elements = combined_sorted[:n]
            possible_topics = []

            for element, weight in top_n_elements:
                # Add future topics for subjects of interest
                if element in self.subjects_to_future_topics:
                    possible_topics.extend(self.subjects_to_future_topics[element])

                # Add future topics for extracurriculars 20% of the time
                if element in self.activities_to_future_topics:
                    if np.random.rand() < 0.2:
                        possible_topics.extend(self.activities_to_future_topics[element])

                # Add future topics for career aspirations
                if element in self.careers_to_future_topics:
                    possible_topics.extend(self.careers_to_future_topics[element])

                # Add future topics for majors
                if element in self.majors_to_future_topics:
                    possible_topics.extend(self.majors_to_future_topics[element])

            # Remove duplicates and update future_topics
            future_topics = list(set(possible_topics))

            # Increase n to get more elements in the next iteration
            n += 1

            # If the length of combined_sorted is less than n, break the loop to prevent an infinite loop
            if n > len(combined_sorted):
                break
        
        # Ensure no more than 5 future topics are recommended
        future_topics = random.sample(future_topics, min(5, len(future_topics)))

        # Give 5 random future topics if the list is empty
        if len(future_topics) == 0:
            future_topics = random.sample(self.future_topics_list, 5)

        return future_topics

    def generate_synthetic_dataset(self, num_samples=1000):
        """
        Input: Number of samples in the dataset
        
        Output: Synthetic Dataset
        """
        #Create the inital empty dataset
        data = []

        # Create num_samples many samples in the dataset
        for _ in range(num_samples):

            first_name = random.choice(self.first_names)
            logging.debug("First name chosen: %s", first_name)

            last_name = random.choice(self.last_names)
            logging.debug("Last name chosen: %s", last_name)

            ethnoracial_group = self.assign_demographic(self.ethnoracial_stats, config["synthetic_data"]["ethnoracial_group"])
            logging.debug("Ethnoracial group chosen: %s", ethnoracial_group)

            gender = self.assign_demographic(self.gender_stats, config["synthetic_data"]["gender"])
            logging.debug("Gender chosen: %s", gender)

            international_status = self.assign_demographic(self.international_stats, config["synthetic_data"]["international_status"])
            logging.debug("International student status chosen: %s", international_status)

            socioeconomic_status = self.assign_demographic(self.socioeconomic_stats, config["synthetic_data"]["socioeconomic_status"])
            logging.debug("Socioeconomics status chosen: %s", socioeconomic_status)

            # List of all a student's identities
            identities = [ethnoracial_group, gender, international_status]

            learning_style = self.generate_learning_style()
            logging.debug("Learning style(s) chosen: %s", learning_style)
            
            # Student semester chosen from 0 (Not yet started college) to 14 (Seventh year second semester)
            student_semester = np.random.randint(0, 15)
            logging.debug("Student semester chosen: %d", student_semester)

            gpa = self.generate_gpa(student_semester)

            major = self.choose_major(student_semester, gender)
            logging.debug("Major(s) chosen: %s", major)

            # Generate identity based extracurriculars
            extracurricular_activities = self.generate_identity_org(identities)

            previous_courses = []
            subjects = []
            career_aspirations_list = []
            logging.debug("Lists initialized: previous courses %s, subjects of interest %s, and career aspirations %s", previous_courses, subjects, career_aspirations_list)

            for semester in range(student_semester+1):

                # Generate the courses for this semester of the student
                previous_courses = self.generate_previous_courses(semester, learning_style, previous_courses, major)
                logging.debug("Previous courses chosen: %s", previous_courses)

                num_courses = len(previous_courses)
                logging.debug("The student has taken %s courses", num_courses)

                top_subject, class_count = self.most_common_class_subject(previous_courses)
                logging.debug("Top subject %s from previous courses and the number of classes in that subject %s", top_subject, class_count)

                subjects = self.generate_subjects_of_interest(previous_courses, subjects, major, top_subject, career_aspirations_list)
                logging.debug("Subjects of interest generated: %s", subjects)

                career_aspirations_list = self.generate_career_aspirations(career_aspirations_list, subjects, major, extracurricular_activities)
                logging.debug("Career aspirations chosen: %s", career_aspirations_list)

                extracurricular_activities = self.generate_extracurricular_activities(subjects, extracurricular_activities, major, career_aspirations_list)
                logging.debug("Extracurriculars generated: %s", extracurricular_activities)

                # If the student has been there for 4 or more semesters and has enough courses,
                # including those in their major then make them graduate by breaking the loop
                if semester >=8 & num_courses >= 32 & class_count >=8:
                    student_semester = semester
                    break

            # Split the tuples into course name, course type, and course subject
            # Ex: Intro Asian American Studies, Discussion/Recitation, AAS
            course_names = [course[0] for course in previous_courses]
            course_type = list(set([course[2] for course in previous_courses]))
            course_subject = list(set([course[3] for course in previous_courses]))
            logging.debug("Course tuple split up")

            # Create weighted lists based on how often an element was in a list
            subjects, subject_weights = self.create_weighted_list(subjects)
            extracurricular_activities, activity_weights = self.create_weighted_list(extracurricular_activities)
            career_aspirations_list, career_weights = self.create_weighted_list(career_aspirations_list)
            # Majors are weighted by semester and gpa
            # Older students with higher gpa's have a higher weight
            if student_semester != 0:
                weight = student_semester * gpa
                major_weights = [weight] * len(major)
            else:
                major_weights = [0] * len(major)
            logging.debug("Weighted lists created")

            # Generate the future topics
            future_topics = self.generate_future_topics(subjects, subject_weights, extracurricular_activities, activity_weights, career_aspirations_list, career_weights, major, major_weights)
            logging.debug("Future Topics found: %s", future_topics)

            # Add the new 'student' to the dataset
            data.append({
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
                'career aspirations': career_aspirations_list,
                'extracurricular activities': extracurricular_activities,
                'future topics': future_topics
            })
            logging.debug("Data appended.")

        # Return the synthetic dataset
        return pd.DataFrame(data)