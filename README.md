# SummerResearch2024
For the iCompBio REU program Summer of 2024 at the University of Tennessee Chattanooga.

Project Lead: Austin Nicolas.

Project Mentor: Dr. Shahnewaz Sakib.

## General Outline of Summer Research Project:
First a synthetic dataset was generated based on both real life data and synthetic mappings. Within the mapping there are three column types: Xp are private data that should not be leaked, X are the data being used to calculate Xu the utility data that the machine learning model is trying to predict. Then the feature importance for the dataset was calculated based on how much each X column impacted the target Xu column. Then the data was privatized using a variety of techniques including differential privacy, random shuffling, and noise addition. Then the privacy loss - utility gain tradeoff was calculated across machine learning models and privatization techniques.

### Goal:
Take student input data and build a privatized version to train a machine learning model. The machine learning model will provide students with topics for future study and possible career paths. Then the students take these topics and paths to advisors, professors, counselors, peers, and others. These people will help the student consider next steps (picking classes, career fairs, etc.) based on the results.

### Table of Contents:
<ol>
  <li>Main Functions</li>
  <li>Data Generation</li>
  <li>Data Preprocessing</li>
  <li>Data Privatization - Under Construction</li>
  <li>Calculating Tradeoffs</li>
  <li>Neural Network - Under Construction</li>
  <li>Sources and Acknowledgments</li>
</ol>

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

### [data_generation_CPU](data_generation/data_generation_CPU.py):
Generates the synthetic dataset on the computer's CPU. The dataset contains the following elements: first name, last name, race or ethnicity, gender, international student status, socioeconomic status, learning style(s), gpa, student semester, major(s), previous courses, previous course types, course subjects, subjects of interest, career aspirations, extracurricular activities, and future topics.

```python
from data import Data
from config import load_config
from data_generation_CPU import DataGenerator

# Create generator class
generator = DataGenerator(config, data)

# Set generation levels
num_samples = 1000
batch_size = 100

# Returns a synthetic dataset with num_samples many rows ('students')
generator.generate_synthetic_dataset(num_samples, batch_size)
```

### [data_generation_GPU](data_generation/data_generation_GPU.py):
Generates the synthetic dataset on the computer's GPU. The dataset contains the following elements: first name, last name, race or ethnicity, gender, international student status, socioeconomic status, learning style(s), gpa, student semester, major(s), previous courses, previous course types, course subjects, subjects of interest, career aspirations, extracurricular activities, and future topics.

```python
from data import Data
from config import load_config
from data_generation_GPU import DataGenerator

# Create generator class
generator = DataGenerator(config, data)

# Set generation levels
num_samples = 1000
batch_size = 100

# Returns a synthetic dataset with num_samples many rows ('students')
generator.generate_synthetic_dataset(num_samples, batch_size)
```

### [data_analysis](data_generation/data_analysis.py):
Takes the synthetic dataset and produces various graphs about the data. For the numerical columns boxplots, distributions, and summary statistics are produced. For all the other columns the top ten highest count items are displayed. Calculates the percentage of empty of NaN values in each column.

```python
from pandas import pd
from config import load_config

# Import synthetic dataset CSV as a pandas dataframe
synthetic_dataset = pd.readcsv('path_to_synthetic_dataset.csv')

# Load configuration
config = load_config()

# Create generator class
analyzer = DataAnalysis(config, synthetic_dataset)
analyzer.analyze_data()

# Saves data analysis graphs to the data_analysis_graphs folder
analyzer.analyze_data()
```

### [data_analysis_graphs](data_generation/data_analysis_graphs):
This folder contains all the graphs produced by data_analysis.

### [Dataset](data_generation/Dataset.csv)
Synthetic dataset. The file here contains 25,000 'students', but you can generate as much data as you need using the data generation functions.

<p align="center">
  <img src="/extra_files/data_construction.png" width="1080" title="Data Column Details" alt="A chart giving the details of each data column">
</p>

## Data Preprocessing:

### Splitting the Data:
Xp = [first name, last name, race or ethnicity, gender, international student status, socioeconomic status]

Xp columns are cut out and removed as we want to keep these hidden and they would be useless for determining a student's career aspirations or future topics.

X = [learning style(s), gpa, student semester, major(s), previous courses, previous course types, course subjects, subjects of interest, extracurricular activities]

X columns are privatized using various techniques. These will also be the features for the neural network.

Xu = [career aspirations, future topics]

Xu columns are left alone. These utility columns are the targets for the neural network.

### [preprocessing](data_preprocessing/preprocessing.py):
preprocess_dataset() - Takes in a synthetic dataset. Xp is cut out, X and Xu are converted from lists of strings to lists of numbers. Outputs a preprocessed dataset.

run_RNN_models() - Takes in a preprocessed dataset. For each list in each column, the lists are padded so they become the same length. Then an RNN is run to reduce dimensionality such that each column becomes 1 dimensional. There are 3 RNN types (Simple, GRU, ans LSTM) and they can run with different numbers of layers (1-4).

```python
from pandas import pd
from config import load_config
from data import Data
from preprocessing import PreProcessing

# Import synthetic dataset CSV as a pandas dataframe
synthetic_dataset = pd.readcsv('path_to_synthetic_dataset.csv')

# Load configuration and data
config = load_config()
data = Data()

# Create preprocessor class
preprocesser = PreProcessing(config, data)

# Returns preprocessed dataset
preprocesser.preprocess_dataset(synthetic_dataset)

# Create the RNN models and save them to their files
# Use one of these models to reduce the dimensionality
# of the preprocessed dataset
preprocessor.create_RNN_models(synthetic_dataset)
```

### [RNN Model Files](data_preprocessing/RNN_models):
In this folder, the different RNN models for dimensionality reduction can be found. Currently I have run Simple with 1 and 2 layers, GRU with 1 layer and LSTM with 1 and 2 layers. So only these five can be found here.

### [Preprocessed_Dataset](data_preprocessing/Preprocessed_Dataset.csv):
All feature columns and utility columns are 1 dimensional. Contains 100,000 'students'.

### [feature_importance](data_preprocessing/feature_importance.py):
Run a random forest model and analyze feature importance using the built in RandomForest feature importance calculator. Calculate this feature importance among feature columns (X) for calculating both utility (Xu) columns: 'career aspirations' and 'future topics'.

```python
from pandas import pd
from config import load_config
from feature_importance import FeatureImportanceAnalyzer

# Import preprocessed dataset (with RNN dimensionality reduction) CSV as a pandas dataframe
preprocessed_dataset = pd.readcsv('RNN_model.csv')

# Load configuration and data
config = load_config()

# Create preprocessor class
feature_analyzer = FeatureImportanceAnalyzer(config, preprocessed_dataset)

# Returns preprocessed dataset
feature_analyzer.calculate_feature_importance()
```

### Feature Importance Files:
There are several folders in data_processing with the feature importance files for specific RNN dimensionality reduction methods. These are summarized in the following table which averages feature importance for both 'career aspirations' and 'future topics' across the RNN methods.

### ![average_feature_importance_comparison](data_preprocessing/average_feature_importance_comparison.png)

## Data Privatization

### [privatization - Under Construction](data_privatization/privatization.py):
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

### [Privatized_Dataset - Under Construction](data_privatization/Privatized_Dataset.csv)
Xp is cut out, X and Xu are multilabel binarized. PCA is run on X.

### [privacy_metrics - Under Construction](data_privatization/privacy_metrics.py):
Calculates the level of data privatization using various metrics: Mean comparison, STD comparison, and Sum comparison. Also outputs the privatization method used and the parameters of the method.

```python
from pandas import pd
from config import load_config
from privacy_metrics import PrivacyMetrics

# Import preprocessed and privatized dataset CSVs as pandas dataframes
preprocessed_dataset = pd.readcsv('path_to_preprocessed_dataset.csv')
privatized_dataset = pd.readcsv('path_to_privatized_dataset.csv')

# Create privacy metrics class
metrics = PrivacyMetrics(config)

# Returns the privacy method and its parameters
# Saves the statistical comparison to 'Stats_Comparison_Dataset.csv'
metrics.calculate_privacy_metrics(preprocessed_dataset, privatized_dataset)
```

### [Stats_Comparison_Dataset - Under Construction](data_privatization/Stats_Comparison_Dataset.csv)
Each row is a column from 'Privatized_Dataset' with the utility columns removed. The columns are the dataset column names, original mean, anonymized mean, original standard deviation, anonymized standard deviation, original sum, anonymized sum.

## Calculating Tradeoffs:

### [processing_private_columns](calculating_tradeoffs/processing_private_columns.py):
The private data are converted into numbered lists. This currently only converts ethnoracial group, gender, and international status.

```python
from pandas import pd
from config import load_config
from datafiles_for_data_construction.data import Data
from processing_private_columns import PrivateColumns

# Import the synthetic dataset CSV as a pandas dataframes
synthetic_dataset = pd.read_csv(path_to_synthetic_dataset.csv')

# Create a private columns class
private_cols = PrivateColumns(config, data)

# Returns the processed private columns (ethnoracial group, gender, international student status)
private_cols.get_private_cols(synthetic_dataset)
```

### [tradeoffs](calculating_tradeoffs/tradeoffs.py):
Takes a dataset and runs calculates how well the X columns can predict the private (ethnoracial group, gender, international student status) and utility columns (career aspirations, future topics).

```python
from pandas import pd
from config import load_config
from tradeoffs import CalculateTradeoffs

# Import the synthetic dataset CSV as a pandas dataframes
RNN_preprocessed_dataset = pd.read_csv('path_to_RNN_preprocessed_dataset.csv')

# Create a private columns class
predictor = CalculateTradeoffs(config, RNN_preprocessed_dataset)

# Returns the processed private columns (ethnoracial group, gender, international student status)
predictor.train_and_evaluate()
```

## Neural Network​:

### [neural_network - Under Construction](neural_network/neural_network.py):
Creates and runs a neural network on the privatized dataset. The target is 'future topics' and the features are the PCA columns. The NeuralNetwork class can also run a cross validation of the model, extract the feature importance for the model, and tune the model hyperparameters.

### [Feature_Importance - Under Construction](neural_network/Feature_Importance.csv)
Columns are features, mean feature importance, and standard deviation feature importance.

## Sources and Acknowledgments:
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
