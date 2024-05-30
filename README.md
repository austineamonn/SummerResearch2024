# SummerResearch2024
For the iCompBio REU program Summer of 2024

# General Outline of Summer Research Project:
To build a framework that will take various inputs from college student users, protect their data by privatizing the dataset, and then output recommendations for topics that the student should consider for future study based on the privatized dataset.

# Goal:
Take student input data and build a privatized version. From the privatized version a machine learning model will provide students with topics for future study. Then the students take these topics to advisors, professors, counselors, peers, and others. These people will help the student pick what courses to take the upcoming semester based on the topics given and the courses offered at the student’s school.

# main:
The main file of the framework. Generates a synthetic dataset using 'data_generation', privatizes the dataset using 'basic_privatization', and calculates the privacy metrics using 'privacy_metrics'.

# config:
Contains the basic configurations for the model.

# csv_loading:
Loads the CSVs and pulls out the relevant information they contain.

# dictionary:
Dictionary that containts demographic information and mappings between various features of the dataset.

# data_generation:
Generates the synthetic dataset. The dataset contains the following elements: race/ethnicity, gender, international, student class year, previous courses taken, career aspirations, subjects of interest, extracurricular activities, and future topics.

# basic_privatization:
Generates the privatized dataset based on the synthetic dataset. Privatizes the following elements: race/ethnicity, gender, international, and student class year.

# privacy_metrics:
Calculates the level of data privatization using various metrics.

# simulated_attack:
Simulates attacks on the dataset. This can run two different types of attack: re-identification and membership inference.

# Sources and Acknowlegments:
https://discovery.cs.illinois.edu/dataset/course-catalog/ - Course Catalog and Course Level Dataset

https://educationdata.org/college-enrollment-statistics - College Demographic Statistics

https://files.eric.ed.gov/fulltext/EJ1192524.pdf - Learning Style Statistics

https://data.world/len/us-first-names-database - First and Last Names Database

Kifer D, Machanavajjhala A. Pufferfish: A framework for mathematical privacy definitions. ACM Trans Database Syst. 2014. https://doi.org/10.1145/2514689 - Inspiration for the pufferfish privatization

Zhao J, Zhang J, Poor HV. Dependent differential privacy for correlated data, 2017;pp. 1–7. https://doi.org/10.1109/GLOCOMW.2017.8269219 - Inspiration for the dependent differential privacy

Python GPT and Chat GPT4o assisted in the programming process
