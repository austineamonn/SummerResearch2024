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

```python
from data_generation import DataGenerator

# Create generator class
generator = DataGenerator

# Returns a synthetic dataset with 1,000
generator.generate_synthetic_dataset(1,000)
```

### [Dataset](data_generation/Dataset.csv)
Synthetic dataset. 1,000 'students'.

## Data Preprocessing:

### Splitting the Data:
Xp = [first name, last name, race or ethnicity, gender, international student status, socioeconomic status]

Xp columns are cut out and removed as we want to keep these hidden and they would be useless for determining a student's career aspirations or future topics.

X = [learning style(s), gpa, student semester, major(s), previous courses, previous course types, course subjects, subjects of interest, extracurricular activities]

X columns are privatized using various techniques. These will also be the features for the neural network.

Xu = [career aspirations, future topics]

Xu columns are left alone. These utility columns are the targets for the neural network.

### [preprocessing](data_preprocessing/preprocessing.py):
Takes in a synthetic dataset. Xp is cut out, X and Xu are multilabel binarized. PCA is run on X. Returns a preprocessed dataset.

```python
from pandas import pd
from config import load_config
from preprocessing import PreProcessing

# Import synthetic dataset CSV as a pandas dataframe
synthetic_dataset = pd.readcsv('path_to_synthetic_dataset.csv')

# Create preprocessor class
preprocesser = PreProcessing(config)

# Returns preprocessed dataset
privatizer.privatize_dataset(preprocesser.preprocess_dataset)
```

### [Preprocessed_Dataset](data_preprocessing/Preprocessed_Dataset.csv):
100 principle components and the utility (Xu) columns.

### [explained_variance_plot](data_preprocessing/explained_variance_plot.png):
Graph of the explained variance ratio of each principle component.

## Data Privatization

### [privatization](data_privatization/privatization.py):
Generates the privatized dataset based on the preprocessed dataset using various methods including: basic differential privacy (using laplace noise addition), uniform noise addition, randomized response, and random shuffling.

```python
from pandas import pd
from config import load_config
from privatization import Privatizer

# Import preprocessed dataset CSV as a pandas dataframe
preprocessed_dataset = pd.readcsv('path_to_preprocessed_dataset.csv')

# Create privatizer class
privatizer = Privatizer(config)

# Returns privatized dataset
privatizer.privatize_dataset(preprocessed_dataset, utility_cols)
```

### [Privatized_Dataset](data_privatization/Privatized_Dataset.csv)
Xp is cut out, X and Xu are multilabel binarized. PCA is run on X.

### [privacy_metrics](data_privatization/privacy_metrics.py):
Calculates the level of data privatization using various metrics: Mean comparison, STD comparison, and Sum comparison. Also outputs the privatization method used and the parameters of the method.

```python
from pandas import pd
from privacy_metrics import PrivacyMetrics

# Import preprocessed and privatized dataset CSVs as pandas dataframes
preprocessed_dataset = pd.readcsv('path_to_preprocessed_dataset.csv')
privatized_dataset = pd.readcsv('path_to_privatized_dataset.csv')

# Create privacy metrics class
metrics = PrivacyMetrics

# Returns the privacy method and its parameters
# Saves the statistical comparison to 'Stats_Comparison_Dataset.csv'
metrics.calculate_privacy_metrics(preprocessed_dataset, privatized_dataset)
```

### [Stats_Comparison_Dataset](data_privatization/Stats_Comparison_Dataset.csv)
Each row is a column from 'Privatized_Dataset' with the utility columns removed. The columns are the dataset column names, original mean, anonymized mean, original standard deviation, anonymized standard deviation, original sum, anonymized sum.

## Neural Network​:

### [neural_network - Under Construction](neural_network/neural_network.py):
Creates and runs a neural network on the privatized dataset. The target is 'future topics' and the features are the PCA columns. The NeuralNetwork class can also run a cross validation of the model, extract the feature importance for the model, and tune the model hyperparameters.

### [Feature_Importance - Under Construction](neural_network/Feature_Importance.csv)
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
