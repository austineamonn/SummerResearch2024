# SummerResearch2024
For the iCompBio REU program Summer of 2024

# General Outline of Summer Research Project:
To build a framework that will take various inputs from college student users, protect their data by privatizing the dataset, and then output recommendations for topics that the student should consider for future study based on the privatized dataset.

# Goal:
Take student input data and build a privatized version. From the privatized version a machine learning model will provide students with topics for future study. Then the students take these topics to advisors, professors, counselors, peers, and others. These people will help the student pick what courses to take the upcoming semester based on the topics given and the courses offered at the student’s school.

# main:
The main file of the framework. Generates a synthetic dataset using 'data_generation', privatizes the dataset using 'basic_privatization', and calculates the privacy metrics using 'privacy_metrics'.

# csv_loading:
Loads the mappings between various elements of the dataset. Creates the data groups and distributions that will be used to build the synthetic dataset.

# data_generation:
Generates the synthetic dataset. The dataset contains the following elements: race/ethnicity, gender, international, student class year, previous courses taken, career aspirations, subjects of interest, extracurricular activities, and future topics.

# basic_privatization:
Generates the privatized dataset based on the synthetic dataset. Privatizes the following elements: race/ethnicity, gender, international, and student class year.

# privacy_metrics:
Calculates the level of data privatization using various metrics.

# Sources:
https://discovery.cs.illinois.edu/dataset/course-catalog/ - Course Catalog and Course Level Dataset

https://educationdata.org/college-enrollment-statistics - College Demographic Statistics
