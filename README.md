# SummerResearch2024
For the iCompBio REU program Summer of 2024. Project Lead: Austin Nicolas. Project Mentor: Dr. Shahnewaz Sakib.

# General Outline of Summer Research Project:
To build a framework that will take various inputs from college student users, protect their data by privatizing the dataset, and build a machine learning model trained on the privatized data that will output recommendations for topics that a student should consider for future study based on their inputs.

# Goal:
Take student input data and build a privatized version. From the privatized version a machine learning model will provide students with topics for future study. Then the students take these topics to advisors, professors, counselors, peers, and others. These people will help the student pick what courses to take the upcoming semester based on the topics given and the courses offered at the student’s school.

# main:
The main file of the framework. Generates a synthetic dataset using 'data_generation', privatizes the dataset using 'privatization', calculates the privacy metrics using 'privacy_metrics', cleans the data for machine learning with 'preprocessing', and trains a neural network on the data using 'neural_network'.

# config:
Contains the basic configurations for the model.

# csv_loading:
Loads the CSVs (Course catalog, first names, and last names) and pulls out the relevant information they contain.

# dictionary:
Dictionary that containts demographic information and mappings between various features of the dataset.

# data_generation:
Generates the synthetic dataset. The dataset contains the following elements: first name, last name, race or ethnicity, gender, international student status,socioeconomic status, learning style, gpa, student semester, previous courses, previous course types, previous courses count, unique subjects in courses, subjects of interest, subjects of interest diversity, career aspirations, extracurricular activities, activities involvement count, and future topics.

# privatization:
Generates the privatized dataset based on the synthetic dataset using anonymization, generalization, and noise addition. You can choose the noise addition from: Random, Laplace, Gaussian, Uniform, Exponential, Gamma, Pufferfish, Dependent Differential Privacy, and Coupled Behavior Analysis.

# privacy_metrics:
Calculates the level of data privatization using various metrics. K-anonymity, L-diversity, Epsilon, Delta, Noise level, Generalization level, Mean comparison, and STD comparison.

# preprocessing:
Prepares privatized dataset to be fed into the machine learning model by generating a cleaned dataset. This includes some encoding, vectorization, and normalization as well as cutting out some elements (first name, last name, race or ethnicity, gender, international student status, and socioeconomic status) that should not play a role in how the model assigns future topics.

# neural_network:
Creates and runs a neural network on the cleaned dataset. The target is 'future topics' and the features are learning style, gpa, student semester, previous courses, previous course type, previous courses count, unique subjects in courses, subjects of interest, subjects of interest diversity, career aspirations, extracurricular activities, and activities involvement count.

# simulated_attack:
Simulates attacks on the dataset. This can run two different types of attack: re-identification and membership inference. This section has not yet been incorporated into main.py as it is still under construction.

# Sources and Acknowlegments:
https://discovery.cs.illinois.edu/dataset/course-catalog/ - Course Catalog and Course Level Dataset

https://educationdata.org/college-enrollment-statistics - College Demographic Statistics

https://files.eric.ed.gov/fulltext/EJ1192524.pdf - Learning Style Statistics

https://data.world/len/us-first-names-database - First and Last Names Database

Kifer D, Machanavajjhala A. Pufferfish: A framework for mathematical privacy definitions. ACM Trans Database Syst. 2014. https://doi.org/10.1145/2514689 - Inspiration for the pufferfish privatization

Zhao J, Zhang J, Poor HV. Dependent differential privacy for correlated data, 2017;pp. 1–7. https://doi.org/10.1109/GLOCOMW.2017.8269219 - Inspiration for the dependent differential privacy

Cao L, Ou Y, Yu P. Coupled behavior analysis with applications. Knowledge Data Eng IEEE Trans. 2012;24:1–1. https://doi.org/10.1109/TKDE.2011.129 - Inspiration for the coupled behavior analysis

Python GPT and Chat GPT4o assisted in the programming process
