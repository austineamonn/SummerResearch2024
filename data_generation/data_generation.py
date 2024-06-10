import pandas as pd
import numpy as np
import random
import logging
import re
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
        # Subjects of Interest
        self.subjects_of_interest = combined_data['subjects_of_interest']
        self.course_to_subject = combined_data['course_to_subject']
        self.related_topics = combined_data['related_topics']
        self.course_subject_to_major = combined_data['course_subject_to_major']

        # Career Aspirations
        self.careers = combined_data['careers']
        self.related_career_aspirations = combined_data['subject_to_career']
        self.career_to_subject = combined_data['career_to_subject']
        self.careers_to_topics = combined_data['careers_to_topics']

        # Extracurriculars
        self.extracurricular_list = combined_data['extracurricular_list']
        self.activity_to_subject = combined_data['extracurricular_to_subject']
        self.activity_to_future_topics = combined_data['extracurricular_to_topics']
        self.identity_org = combined_data['identity_to_org']
        self.identity_org_for_all = combined_data['identity_organizations_open_to_all']
        
        # Learning Style
        self.learning_style = combined_data['learning_style']
        self.course_type_to_learning_styles = combined_data['course_type_to_learning_styles']
        
        # Demographics
        self.race_ethnicity = combined_data['race_ethnicity']
        self.gender = combined_data['gender']
        self.international = combined_data['international']
        self.socioeconomic = combined_data['socioeconomic']
        
        # Majors
        self.majors = combined_data['majors']
        self.majors_to_subjects = combined_data['majors_to_subjects']
        self.majors_to_careers = combined_data['majors_to_careers']

        logging.debug("Combined data loaded.")

        self.course_tuples = course_loader.course_tuples
        self.course_year_mapping = {course[0]: self.map_course_to_year(course) for course in self.course_tuples}
        logging.debug("Course information loaded.")

        self.first_names_list = fn_loader.first_names
        self.last_names_list = ln_loader.last_names
        logging.debug("First and last names loaded.")

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
            course_subjects = self.course_subject_to_major[subject]
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
        possible_courses_higher_ls = self.filter_course_by_learning_style(possible_courses_higher, learning_styles)
        possible_courses_lower_ls = self.filter_course_by_learning_style(possible_courses_lower, learning_styles)

        possible_courses_mj = self.filter_course_by_major(possible_courses, majors)
        possible_courses_higher_mj = self.filter_course_by_major(possible_courses_higher, majors)
        possible_courses_lower_mj = self.filter_course_by_major(possible_courses_lower, majors)

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

    def generate_career_aspirations(self, subjects=[], majors=[]):
        """
        Generate a list of career aspirations for a student based on their subjects of interest.
        """
        # Find all possible careers related to the subjects of interest
        possible_careers = []
        for subject in subjects:
            if subject in self.related_career_aspirations:
                possible_careers.extend(self.related_career_aspirations[subject])

        # Find all possible careers related to major(s)
        for major in majors:
            if major in self.majors_to_careers:
                possible_careers.extend(self.majors_to_careers[major])
        
        # Add a small chance of including a random career from all possible careers
        if np.random.rand() < 0.1:  # 10% chance
            possible_careers.append(np.random.choice(self.careers))
        
        # Remove duplicates from the possible careers list
        possible_careers = list(set(possible_careers))
        
        # Select a random number of careers (between 0 and 4) from the possible careers
        return random.sample(possible_careers, min(np.random.randint(0, 5), len(possible_careers))) if possible_careers else []

    def generate_extracurricular_activities(self, subjects, activities=[]):
        """
        Inputs: subjects of interest, current activities list
        Output: new activity list
        """
        possible_activities = []

        for subject in subjects:
            value = self.activity_to_subject.get(subject)
            if value is not None:
                logging.debug(f"Key {subject} exists in the dictionary with value {value}")
                if np.random.rand() < 0.3: # 30% chance a person joins a club based on a subject of interest
                    possible_activities.append(value)
                    logging.debug(f"{value} added because of {subject}")
            else:
                logging.debug(f"Key {subject} does not exist in the dictionary")

        # 10% chance a person joins a club regarless of subjects of interest
        if np.random.rand() < 0.1:
            extra_club = random.choice(self.extracurricular_list)
            possible_activities.append(extra_club)

        # Randomly add 0-5 of the possible activities and ensure no duplicates
        activities.extend(random.sample(possible_activities, min(np.random.randint(0, 5), len(possible_activities))))
        activities = list(set(activities))

        logging.debug(f"Extracurriculars are: {activities}")
        return activities
    
    def generate_identity_org(self, identities):
        """
        Inputs: current identities
        Output: activity list
        """
        activities = []

        for identity in identities:
            value = self.identity_org.get(identity)
            if value is not None:
                logging.debug(f"Key {identity} exists in the dictionary with value {value}")
                if np.random.rand() < 0.2: # 20% chance a person joins a club based on an identity
                    activities.append(value)
                    logging.debug(f"{value} added because of {identity}")
            else:
                logging.debug(f"Key {identity} does not exist in the dictionary")

        # Identity based clubs students can join regardless of identity
        if np.random.rand() < 0.05:
            extra_club = random.choice(self.identity_org_for_all)
            activities.append(extra_club)

        # Ensure no duplicates
        activities = list(set(activities))

        logging.debug(f"Extracurriculars are: {activities}")
        return activities
    
    def generate_subjects_of_interest(self, previous_courses=[], subjects=[], majors=[], top_subject=None, careers=[]):
        possible_subjects = []

        # Add in subjects of interest based on previous courses taken
        for course in previous_courses:
            logging.debug("Course being examined: %s", course)
            if course[3] in self.course_to_subject.keys():
                    # 100 level courses indicate a 30% interest in the subject
                    if 100 <= course[1] < 200:
                        if np.random.rand() < 0.3:
                            possible_subjects.append(self.course_to_subject[course[3]])
                            logging.debug("%s has been added to the list", self.course_to_subject[course[3]])
                    # 200 level courses indicate a 60% interest in the subject
                    elif 200 <= course[1] < 300:
                        if np.random.rand() < 0.6:
                            possible_subjects.append(self.course_to_subject[course[3]])
                            logging.debug("%s has been added to the list", self.course_to_subject[course[3]])
                    # By the 300 level you are committed to the subject
                    elif 300 <= course[1]:
                        possible_subjects.append(self.course_to_subject[course[3]])
                        logging.debug("%s has been added to the list", self.course_to_subject[course[3]])
            else:
                logging.debug("this course subject %s was not in the list %s", course[3], self.course_to_subject.keys())
        logging.debug("Initial subject list chosen: %s", possible_subjects)

        # Add in subjects of interest based on career aspirations
        for career in careers:
            if career in self.career_to_subject:
                possible_subjects.extend(self.career_to_subject[career])
        logging.debug("Added in subjects based on career aspirations")

        if np.random.rand() < 0.3:  # 30% chance to add extra subjects of interest
            extra_subjects = random.sample(self.subjects_of_interest,  min(np.random.randint(1, 3), len(self.subjects_of_interest)))
            possible_subjects.extend(extra_subjects)

        # Randomly add 0-5 of the possible subjects of interest
        subjects.extend(random.sample(possible_subjects, min(np.random.randint(0, 5), len(possible_subjects))))
        
        # Add in all the subjects related to a person's major
        for major in majors:
            value = self.majors_to_subjects.get(major)
            if value is not None:
                subjects.extend(value)
                logging.debug(f"{value} added because of {major}")
            else:
                logging.debug(f"Key {major} does not exist in the dictionary")

        # Add in the subjects related to the most common subject in courses taken
        if top_subject is not None:
            if top_subject in self.course_to_subject:
                subjects.append(self.course_to_subject[top_subject])
        
        logging.debug("Final subjects list %s", subjects)
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
    
    def generate_future_topics(self, subjects=[], subject_weights=[], activities=[], activity_weights=[], careers=[], career_weights=[]):
        # Filter out elements with None weights and create combined lists and weights
        combined_list = []
        combined_weights = []
        
        for lst, wts in [(subjects, subject_weights), (activities, activity_weights), (careers, career_weights)]:
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
                if element in self.related_topics:
                    possible_topics.extend(self.related_topics[element])

                # Add future topics for extracurriculars 20% of the time
                if element in self.activity_to_future_topics:
                    if np.random.rand() < 0.2:
                        possible_topics.extend(self.activity_to_future_topics[element])

                # Add future topics for career aspirations
                if element in self.careers_to_topics:
                    possible_topics.extend(self.careers_to_topics[element])

            # Remove duplicates and update future_topics
            future_topics = list(set(possible_topics))

            # Increase n to get more elements in the next iteration
            n += 1

            # If the length of combined_sorted is less than n, break the loop to prevent an infinite loop
            if n > len(combined_sorted):
                break
        
        # Ensure no more than 5 future topics are recommended
        future_topics = random.sample(future_topics, min(5, len(future_topics)))
        logging.info(f"Future topics {future_topics}")
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

            first_name = random.choice(self.first_names_list)
            logging.debug("First name chosen")

            last_name = random.choice(self.last_names_list)
            logging.debug("Last name chosen")

            ethnoracial_group = self.assign_demographic(self.race_ethnicity, config["synthetic_data"]["ethnoracial_group"])
            logging.debug("Ethnoracial group chosen")

            gender = self.assign_demographic(self.gender, config["synthetic_data"]["gender"])
            logging.debug("Gender chosen")

            international_status = self.assign_demographic(self.international, config["synthetic_data"]["international_status"])
            logging.debug("International student status chosen")

            socioeconomic_status = self.assign_demographic(self.socioeconomic, config["synthetic_data"]["socioeconomic_status"])

            # List of all a student's identities
            identities = [ethnoracial_group, gender, international_status]

            learning_style = [self.assign_demographic(self.learning_style, config["synthetic_data"]["learning_style"])]
            if np.random.rand() < 0.1:  # 10% chance to add an extra learning style
                extra_style = self.assign_demographic(self.learning_style, config["synthetic_data"]["learning_style"])
                if extra_style in learning_style:
                    logging.debug("Double up of learning style")
                else: # Only add the extra learning style if it is different than the original one
                    learning_style.append(extra_style)
            logging.debug("Learning style(s) chosen")
            
            # Student semester chosen from 0 (Not yet started college) to 14 (Seventh year second semester)
            student_semester = np.random.randint(0, 15)
            logging.debug("Student semester chosen: %d", student_semester)

            if student_semester == 0:
                gpa = None
                logging.debug("GPA left empty because student semester is zero")
            else:
                gpa = round(np.random.uniform(2.0, 4.0), 2)
                logging.debug("GPA chosen: %.2f", gpa)


            if student_semester <=4:
                if np.random.rand() < 0.3: # Only 30% of underclassmen students have a major / intended major
                    major = [random.choice(self.majors)]
                else:
                    major = []
                    logging.debug("Major left empty because student has no major / intended major")
            elif student_semester > 4:
                major = [random.choice(self.majors)]
            
            if len(major) == 1:
                if np.random.rand() < 0.3: # Only 30% of students have a second major / intended major
                    extra_major = random.choice(self.majors)
                    if extra_major in major:
                        logging.debug("Double up of major")
                    else: # Only add the extra major if it is different than the original one
                        major.append(extra_major)
            logging.debug("Major(s) chosen")

            extracurricular_activities = self.generate_identity_org(identities)

            previous_courses = []
            subjects = []
            career_aspirations_list = []

            for semester in range(student_semester+1):

                # Generate the courses for this semester of the student
                previous_courses = self.generate_previous_courses(semester, learning_style, previous_courses, major)
                logging.debug("Previous courses chosen")

                num_courses = len(previous_courses)
                logging.debug("The student has taken %s courses", num_courses)

                top_subject, class_count = self.most_common_class_subject(previous_courses)
                logging.debug("Top subject from previous courses and the number of classes in that subject found")

                subjects = self.generate_subjects_of_interest(previous_courses, subjects, major, top_subject, career_aspirations_list)
                logging.debug("Subjects of interest generated")

                career_aspirations_list = self.generate_career_aspirations(subjects, major)
                logging.debug("Career aspirations chosen")

                extracurricular_activities = self.generate_extracurricular_activities(subjects, extracurricular_activities)
                logging.debug("Extracurriculars generated")

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

            # Create weighted lists based on how often an element was in a list
            subjects, subject_weights = self.create_weighted_list(subjects)
            extracurricular_activities, activity_weights = self.create_weighted_list(extracurricular_activities)
            career_aspirations_list, career_weights = self.create_weighted_list(career_aspirations_list)

            # Generate the future topics
            future_topics = self.generate_future_topics(subjects, subject_weights, extracurricular_activities, activity_weights, career_aspirations_list, career_weights)

            # Add the new 'student' to the dataset
            data.append({
                'first name': first_name,
                'last name': last_name,
                'ethnoracial group': ethnoracial_group,
                'gender': gender,
                'international status': international_status,
                'socioeconomic status': socioeconomic_status,
                'learning_style': learning_style,
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
