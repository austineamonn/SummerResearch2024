# SummerResearch2024
For the iCompBio REU program Summer of 2024. Project Lead: Austin Nicolas. Project Mentor: Dr. Shahnewaz Sakib.

# General Outline of Summer Research Project:
To build a framework that will take various inputs from college student users, protect their data by privatizing the dataset, and build a machine learning model trained on the privatized data that will solve a multilabel classification problem and output recommendations for topics that a student should consider for future study based on their inputs.

# Goal:
Take student input data and build a privatized version. From the privatized version a machine learning model will provide students with topics for future study. Then the students take these topics to advisors, professors, counselors, peers, and others. These people will help the student pick what courses to take the upcoming semester based on the topics given and the courses offered at the student’s school.

# main:
The main file of the framework. Generates a synthetic dataset using 'data_generation', privatizes the dataset using 'privatization', calculates the privacy metrics using 'privacy_metrics', cleans the data for machine learning with 'preprocessing', and trains a neural network on the data using 'neural_network'.

# config:
Contains the basic configurations for the model. Most important is the ability to configure which parts of the model you want to run. The list you can pick from is: Generate Dataset, Privatize Dataset, Calculate Privacy Metrics, Clean Privatized Dataset, Run Neural Network, Test Neural Network.

# dictionary - Currently being replaced by data.py:
Dictionary that containts demographic information, mappings between various features of the dataset, and feature generalizations based on four generalization levels: full, broad, slight, and none.

# data_generation:
Generates the synthetic dataset. The dataset contains the following elements: first name, last name, race or ethnicity, gender, international student status,socioeconomic status, learning style, gpa, student semester, previous courses, previous course types, previous courses count, course subjects, unique subjects in courses, subjects of interest, subjects of interest diversity, career aspirations, extracurricular activities, activities involvement count, and future topics.

# privatization - Under Construction:
Generates the privatized dataset based on the synthetic dataset using anonymization, generalization, and noise addition. You can choose the noise addition from: Random, Laplace, Gaussian, Uniform, Exponential, Gamma, Pufferfish, Dependent Differential Privacy, and Coupled Behavior Analysis.

# privacy_metrics - Under Construction:
Calculates the level of data privatization using various metrics. K-anonymity, L-diversity, Epsilon, Delta, Noise level, Generalization level, Mean comparison, and STD comparison.

# preprocessing - Under Construction:
Prepares privatized dataset to be fed into the machine learning model by generating a cleaned dataset. This includes some encoding, vectorization, and normalization as well as cutting out some elements (first name, last name, race or ethnicity, gender, international student status, and socioeconomic status) that should not play a role in how the model assigns future topics. The privatized dataset has 3459 feature columns and 164 target columns. ​

# neural_network - Under Construction:
Creates and runs a neural network on the cleaned dataset. PCA is done to reduce the dimensionality of the problem. The target is 'future topics' and the features are learning style, gpa, student semester, previous courses, previous course type, previous courses count, course subjects, unique subjects in courses, subjects of interest, subjects of interest diversity, career aspirations, extracurricular activities, and activities involvement count. The NeuralNetwork class can also run a cross validation of the model, extract the feature importance for the model, and tune the model hyperparameters.

# Sources and Acknowlegments:
https://discovery.cs.illinois.edu/dataset/course-catalog/ - Course Catalog with Course Names, Course Types, and Course Subject Abbreviations.

https://educationdata.org/college-enrollment-statistics - College Demographic Statistics

https://files.eric.ed.gov/fulltext/EJ1192524.pdf - Learning Style Statistics

https://data.world/len/us-first-names-database - First and Last Names Database

https://www.pewresearch.org/social-trends/2019/05/22/a-rising-share-of-undergraduates-are-from-poor-families-especially-at-less-selective-colleges/ - College Family Income Statistics

https://williamsinstitute.law.ucla.edu/publications/nonbinary-lgbtq-adults-us/ - Nonbinary Statistics

https://nces.ed.gov - Gender College Statistics

https://courses.illinois.edu/schedule/DEFAULT/DEFAULT - Course Subject Abbreviation to Course Subject Mapping

https://bigeconomics.org/college-majors-explorer/ - List of Majors, Careers

Kifer D, Machanavajjhala A. Pufferfish: A framework for mathematical privacy definitions. ACM Trans Database Syst. 2014. https://doi.org/10.1145/2514689 - Inspiration for the pufferfish privatization

Zhao J, Zhang J, Poor HV. Dependent differential privacy for correlated data, 2017;pp. 1–7. https://doi.org/10.1109/GLOCOMW.2017.8269219 - Inspiration for the dependent differential privacy

Cao L, Ou Y, Yu P. Coupled behavior analysis with applications. Knowledge Data Eng IEEE Trans. 2012;24:1–1. https://doi.org/10.1109/TKDE.2011.129 - Inspiration for the coupled behavior analysis

Python GPT and Chat GPT4o assisted in the programming process
