# SummerResearch2024
For the iCompBio REU program Summer of 2024 at the University of Tennessee Chattanooga.

Project Lead: Austin Nicolas.

Project Mentor: Dr. Shahnewaz Sakib.

## General Outline of Summer Research Project:
To build a framework that will take various inputs from college student users, protect their data by privatizing the dataset, and build a machine learning model trained on the privatized data that will solve a multilabel classification problem and output recommendations for topics that a student should consider for future study based on their inputs.

### Goal:
Take student input data and build a privatized version. From the privatized version a machine learning model will provide students with topics for future study. Then the students take these topics to advisors, professors, counselors, peers, and others. These people will help the student pick what courses to take the upcoming semester based on the topics given and the courses offered at the student’s school.

## Main Functions:

### [main](main.py):
The main file of the framework. Generates a synthetic dataset using 'data_generation', privatizes the dataset using 'privatization', calculates the privacy metrics using 'privacy_metrics', and trains a neural network on the data using 'neural_network'.

### [config](config.py):
Contains the basic configurations for the model. Most important is the ability to configure which parts of the model you want to run. The list you can pick from is: Generate Dataset, Privatize Dataset, Calculate Privacy Metrics, Run Neural Network, and Test Neural Network.

## Data Generation:

### [datafiles_for_data_construction](datafiles_for_data_construction)
Various JSON files that have lists of data and feature tuples. This folder also contains the data.py file.

### [data](datafiles_for_data_construction/data.py):
Dictionary that containts demographic information, lists of features, feature tuples, and mappings between various features of the dataset.

### [data_generation](data_generation/data_generation.py):
Generates the synthetic dataset. The dataset contains the following elements: first name, last name, race or ethnicity, gender, international student status, socioeconomic status, learning style(s), gpa, student semester, major(s), previous courses, previous course types, course subjects, subjects of interest, career aspirations, extracurricular activities, and future topics.

### [Dataset](data_generation/Dataset.csv)
Synthetic dataset. 1,000 'students'.

## Data Privatization:

### Splitting the Data:
Xp = [first name, last name, race or ethnicity, gender, international student status, socioeconomic status]

Xp columns are cut out and removed as we want to keep these hidden and they would be useless for determining a student's career aspirations or future topics.

X = [learning style(s), gpa, student semester, major(s), previous courses, previous course types, course subjects, subjects of interest, extracurricular activities]

X columns are privatized using various techniques. These will also be the features for the neural network.

Xu = [career aspirations, future topics]

Xu columns are left alone. These are the targets for the neural network.

### [privatization](data_privatization/privatization.py):
Generates the privatized dataset based on the synthetic dataset using multilabel binarization, normalization, noise addition, and shuffling. You can choose the noise addition from: Laplace, Uniform, Randomized Response. Also generates the cleaned dataset based on the synthetic dataset using multilabel binarization and normalization.

### [Privatized_Dataset](data_privatization/Privatized_Dataset.csv)
Privatized version of the synthetic dataset. GPA and student semester are normalized. All other elements are multilabel binarized. Either add noise from one of the methods, or shuffle the data.

### [Cleaned_Dataset](data_privatization/Cleaned_Dataset.csv)
Cleaned version of the synthetic dataset. GPA and student semester are normalized. All other elements are multilabel binarized.

### [privacy_metrics](data_privatization/privacy_metrics.py):
Calculates the level of data privatization using various metrics. Mean comparison, STD comparison, and Sum comparison. Also outputs the privatization method used and the parameters of the method.

### [Stats_Comparison_Dataset](data_privatization/Stats_Comparison_Dataset.csv)
Each row is a column from 'Privatized_Dataset'. The columns are the dataset column names, original mean, anonymized mean, original standard deviation, anonymized standard deviation, original sum, anonymized sum.

## Neural Network​:

### [neural_network - Under Construction](neural_network/neural_network.py):
Creates and runs a neural network on the privatized dataset. PCA is done to reduce the dimensionality of the problem. The target is 'future topics' and the features are learning style, gpa, student semester, previous courses, previous course type, previous courses count, course subjects, unique subjects in courses, subjects of interest, subjects of interest diversity, career aspirations, extracurricular activities, and activities involvement count. The NeuralNetwork class can also run a cross validation of the model, extract the feature importance for the model, and tune the model hyperparameters.

### [Feature_Importance](neural_network/Feature_Importance.csv)
Columns are features, mean feature importance, and standard deviation feature importance.

## Sources and Acknowlegments:
https://discovery.cs.illinois.edu/dataset/course-catalog/ - Course Catalog with Course Names, Course Types, and Course Subject Abbreviations.

https://educationdata.org/college-enrollment-statistics - College Demographic Statistics

https://files.eric.ed.gov/fulltext/EJ1192524.pdf - Learning Style Statistics

https://data.world/len/us-first-names-database - First and Last Names Database

https://www.pewresearch.org/social-trends/2019/05/22/a-rising-share-of-undergraduates-are-from-poor-families-especially-at-less-selective-colleges/ - College Family Income Statistics

https://williamsinstitute.law.ucla.edu/publications/nonbinary-lgbtq-adults-us/ - Nonbinary Statistics

https://nces.ed.gov - Gender College Statistics

https://courses.illinois.edu/schedule/DEFAULT/DEFAULT - Course Subject Abbreviation to Course Subject Mapping

https://bigeconomics.org/college-majors-explorer/ - List of Majors, Careers

Python GPT and Chat GPT4o assisted in the programming process
