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
  <li>Data Privatization</li>
  <li>Calculating Tradeoffs</li>
  <li>Graphics</li>
  <li>Sources and Acknowledgments</li>
</ol>

## Main Functions:

### [main - Under Construction](main.py):
The main file of the framework. What can this file do?
<ol>
  <li>Generates a synthetic dataset using either 'data_generation_CPU' or 'data_generation_GPU'</li>
  <li>Analyzes the dataset using 'data_analysis'</li>
  <li>Preprocesses and reduces the dimensionality of the dataset using 'preprocessing'</li>
  <li>Privatizes the dataset using 'privatization'</li>
  <li>Calculates the privacy metrics using 'privacy_metrics'</li>
</ol>
Use the config file to change which of the above parts of the file are run during main.

### [interactive main - Under Construction](main.ipynb):
An interactive jupyter notebook that walks through the full data pipeline process from data generation to privacy - utility tradeoffs. Essentially, this notebook follows what 'main.py' does but on a smaller, more informative, and more interactive scale.

### [config](config.py):
Contains the basic configurations for the model. Most important is the ability to configure which parts of the model you want to run. The list you can pick from is: Generate Dataset, Privatize Dataset, Calculate Privacy Metrics.

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
  <img src="/graphics/canva_generated_graphs/data_construction.png" width="1080" title="Data Column Details" alt="A chart giving the details of each data column">
</p>

## Data Preprocessing:

### Splitting the Data:
<table>
  <tr>
    <th>Data Category</th>
    <th>Features</th>
    <th>Explanation</th>
  </tr>
  <tr>
    <td>Xp</td>
    <td>first name, last name, race or ethnicity, gender, international student status, and socioeconomic status</td>
    <td>Xp columns are cut out and removed as we want to keep these hidden and they would be useless for determining a student's career aspirations or future topics.</td>
  </tr>
  <tr>
    <td>X</td>
    <td>learning style(s), gpa, student semester, major(s), previous courses, previous course types, course subjects, subjects of interest, and extracurricular activities</td>
    <td>X columns are privatized using various techniques. These will also be the features for the neural network.</td>
  </tr>
  <tr>
    <td>Xu</td>
    <td>career aspirations and future topics</td>
    <td>Xu columns are left alone. These utility columns are the targets for the neural network.</td>
  </tr>
</table>

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

### [Preprocessed_Dataset](data_preprocessing/Preprocessed_Dataset.csv):
All feature columns and utility columns have been converted into either binary lists or numerical lists. Contains 100,000 'students' in the CSV.

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

### [RNN Model Files](data_preprocessing/RNN_models):
In this folder, the different RNN models for dimensionality reduction can be found. Currently I have run Simple , GRU and LSTM all with 1 layer. The combined versions contain the preprocessed private columns while the regular versions do not.

## Data Privatization

### Privatization Methods
There are two main methods:
<ol>
  <li>Differential Privacy which adds noise to the data to reach a privatization level specified by epsilon. Epsilon is set to 0.1 and the noise type is set to 'laplace'. The list lengths can be changed based on the noise type though this is set to False.</li>
  <li>Random Shuffling which shuffles a set ratio of the data rows. The shuffle ratio is set to 10% but can be automatically set to 100% using the 'full shuffle' privatization method.</li>
</ol>
Overall, there is wide flexibility in the options for privatization method. The sensitivity for the differential privacy is calculated using the mean method.

<p align="center">
  <img src="/graphics/canva_generated_graphs/privatization_methods.png" width="1080" title="Privatization Methods Flowchart" alt="A flowchart showing the different data privatization methods">
</p>

### [privatization](data_privatization/privatization.py):
Generates the privatized dataset based on the preprocessed dataset using one of the various methods listed above.

```python
from pandas import pd
from config import load_config
from privatization import Privatizer

# Import preprocessed dataset CSV as a pandas dataframe
preprocessed_dataset = pd.readcsv('path_to_preprocessed_dataset.csv')

# Create privatizer class using differential privacy with laplace noise addition, epsilon of 0.1 and no list length changing
privatizer = Privatizer(config, data, 'basic differential privacy', True)

# Returns privatized dataset
privatizer.privatize_dataset(preprocessed_dataset)
```

### Privatized Datasets
There are a variety of privatized datasets including differential privacy with laplace and uniform noise addition and with and without list length changing (LLC) as well as the random shuffling at 10% and 100%.

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

## Calculating Tradeoffs:

### [tradeoffs](calculating_tradeoffs/tradeoffs.py):
Takes a dataset and runs calculates how well the X columns can predict the private (ethnoracial group, gender, international student status) and utility columns (career aspirations, future topics). The file will run the model to predict each of the columns as well as the private columns combined, the utility columns combined, and all the columns combined. Three different machine learning models will run: Linear Regression, Decision Tree, and Random Forest. SHAP feature importance is calculated for each prediction.

--- Be warned that the current model imputes NaN values which means students who have not gone to college suddenly have GPAs and other similar issues. ---

```python
from pandas import pd
from config import load_config
from tradeoffs import CalculateTradeoffs

# Import the synthetic dataset CSV as a pandas dataframes
RNN_preprocessed_dataset = pd.read_csv('path_to_RNN_preprocessed_dataset.csv')

# Calculate the tradeoffs for a specific RNN model
predictor = CalculateTradeoffs(config, RNN_preprocessed_dataset, RNN_model)

# Train and evaluate the models
results = predictor.train_and_evaluate()

# Save the results to a CSV file
results_csv_path = 'model_evaluation_results.csv'
predictor.save_results_to_csv(results, results_csv_path)
```

### [tradeoffs_categorical](calculating_tradeoffs/tradeoffs_categorical.py):
Since the private columns are categorical, in addition to running the tradeoff files, categorical machine learning is used for the private columns on their own.
The file takes a dataset and runs calculates how well the X columns can predict the private columns (ethnoracial group, gender, international student status). The file will run the model to predict each of the columns as well as the private columns combined. One basic neural network is run with 2 hidden layers and 2 dropout layers. SHAP feature importance is calculated for each prediction.

--- Be warned that the current model imputes NaN values which means students who have not gone to college suddenly have GPAs and other similar issues. ---

```python
from pandas import pd
from config import load_config
from tradeoffs_categorical import CalculateTradeoffsCategorical

# Import the synthetic dataset CSV as a pandas dataframes
RNN_preprocessed_dataset = pd.read_csv('path_to_RNN_preprocessed_dataset.csv')

# Calculate the categorical tradeoffs for a specific RNN model
predictor = CalculateTradeoffsCategorical(config, RNN_preprocessed_datase, RNN_model)

# Train and evaluate the models
results = predictor.train_and_evaluate()

# Save the results to a CSV file
results_csv_path = 'model_evaluation_results.csv'
predictor.save_results_to_csv(results, results_csv_path)
```

### SHAP Values
These CSV files contain the SHAP values for different dimensionality reduction models and different predictive models with different targets. SHAP Explainer was used for linear regression, SHAP TreeExplainer was used for decision tree and random forest, and SHAP KernelExplainer was used for the neural network.

### Tradeoff Results
These CSV files contain the error/accuracy values for different dimensionality reduction models and different predictive models with different targets. The error values for 'tradeoffs' are mean squared error, mean absolute error, R squared, root mean squared error, mean absolute percentage error, median absolute error, explained variance, and mean bias deviation. The accuracy values for 'categorical tradeoffs' are accuracy, precision, recall, F1 score, balanced accuracy, log loss, classification report, and confusion matrix. Both types also include training time.

## Graphics:

### [tradeoff_graphs](graphics/tradeoff_graphs/tradeoff_graphs.py):
Plots the Mean Standard Error (MSE), Mean Absolute Erro (MAE), and R-Squared (R2) for each machine learning model (Decision Tree, Random Forest, Linear Regression) and for each target (each of the private columns, each of the utility columns, all of the private columns, all of the utility columns, and all of the columns combined). This is repeated for each of the privatized dataset versions.

### [SHAP_graphs](graphics/SHAP_graphs/SHAP_graphs.py):
Graphs the Bar Graph and the SHAP Summary for the feature importance of each machine learning model and target combination for each of the privatized datasets.

```python
from config import load_config
from tradeoffs import CalculateTradeoffs

# Create a private columns class
grapher = SHAP_Grapher(config)

# Generate the feature importance graphs
grapher.generate_graphs()
```

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
