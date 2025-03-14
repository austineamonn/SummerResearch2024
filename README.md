# PipelineEDU: Creating High-Quality Synthetic Data
for Educational Research and Applications
For the iCompBio REU program Summer of 2024 at the University of Tennessee Chattanooga. Additional analysis done in the Introduction to R for Biologists course taught by Dr. Caroline Dong at Grinnell College.

Project Lead: Austin Nicolas.

Project Mentor: Dr. Shahnewaz Sakib.

Additional Mentors: Dr. Caroline Dong

## General Outline of Research Project:
First a synthetic dataset was generated based on both real life data and synthetic mappings. Within the mapping there are three column types: Xp are private data that should not be leaked, X are the data being used to calculate Xu the utility data that the machine learning model is trying to predict. Then the feature importance for the dataset was calculated based on how much each X column impacted the target Xu column. Then the data was privatized using a variety of techniques including variations on differential privacy and random shuffling. Then the privacy loss - utility gain tradeoff was calculated across machine learning models and privatization techniques.

### Goal:
Take student input data and build a privatized version to train a machine learning model. The machine learning model will provide students with topics for future study and possible career paths. Then the students take these topics and paths to advisors, professors, counselors, peers, and others. These people will help the student consider next steps (picking classes, career fairs, etc.) based on the results.

### Applications:
Data privatization techniques are vital for allowing data to be shared without risking exposing the sensitive data to being identifiable by malicious parties. Though this project uses student demographic information as the sensitive data, this work is very applicable to medical data collection and analysis.

### Regressification:

Regressification is a novel (to the best of the Author's knowledge) hybrid Classification-Regression machine learning model style. This style trains a regression model and then evaluates the data like a classification model. 

For example the in Decision Tree Regressifier, the tree is split based on lowest sum of squared errors (SSE) like a regression model, rather than entropy like a classification model would. Then the tree is evaluated using accuracy typical of a classification model, rather than mean squared error (MSE) like a regression model would.

Why use regressification?

Since the split future topics have 226 distinct categories, an enormous amount of data would be required to have enough from each category to train the model. Additionally if methods like onehot encoding were used the model would go from 1 target (the top recommended future topic is trained separate from the second most recommended future topic and so on) to 226.

Regressification skirts all these issues by training the model under regression and then evaluating it as a classification problem. Less data needed, only train with one target, and metrics like accuracy can still be used. It's the best of both worlds.

For this use case the target values are all equally spaced apart at integer values (0 to 255). However the model will assume that nearby categories are closely related since it is trained on regression. For this use case that assumption is not true, but future work could capitalize on this assumption to build more accurate models.

### Warning:
The file structures are currently going under a major overhaul to match python package best practices. The information contained within this README file still applies but the location of the files may have changed. Generally, the same structure is there but just within the src/IntelliShield folder.

### Table of Contents:
<ol>
  <li>Main Functions</li>
  <li>Data Generation</li>
  <li>Data Analysis</li>
  <li>Data Preprocessing</li>
  <li>Data Privatization</li>
  <li>Calculating Tradeoffs</li>
  <li>Model Comparison</li>
  <li>Sources and Acknowledgments</li>
</ol>

## Main Functions:

All main and config were lost in the restructuring but they will be revived soon. Main interactive notebook was also lost but has been revived.

Config will be replaced with the logger file.

### [Main - Under Construction](main.py):
The main file of the framework. What can this file do?
<ul>
  <li>Generates a synthetic dataset using either 'data_generation_CPU' or 'data_generation_GPU'</li>
  <li>Analyzes the dataset using 'data_analysis'</li>
  <li>Preprocesses and reduces the dimensionality of the dataset using 'preprocessing'</li>
  <li>Privatizes the dataset using 'privatization'</li>
  <li>Calculates the privacy metrics using 'privacy_metrics'</li>
  <li>Calculates the utility gain using regression models from 'calculating tradeoffs/regression'</li>
  <li>Calculates the utility gain for 'future topics' using alternate models from 'calculating tradeoffs/alternate'</li>
  <li>Calculates the privacy loss using classification models from 'calculating tradeoffs/classification'</li>
</ul>
Use the config file to change which of the above parts of the file are run during main. You don't need to run all of them but they will run in this order.

### [Interactive Main - Under Construction](notebooks/main.ipynb):
An interactive Jupyter Notebook that walks through the full data pipeline process from data generation to privacy - utility tradeoffs. Essentially, this notebook follows what 'main.py' does but on a smaller, more informative, and more interactive scale. This file is just to explain how the code works. <em>None of the files produced are saved</em>. There are two ways to run this code:

<table>
  <tr>
    <th>How to Run File</th>
    <th>Where is it Run?</th>
    <th>Inital Set Up</th>
    <th>Which should I use?</th>
  </tr>
  <tr>
    <td>Jupyter Notebook</td>
    <td>Locally</td>
    <td>Clone this github into a virtual environment.</td>
    <td>Want to first explore and then use the full program functionality.</td>
  </tr>
  <tr>
    <td>Google Colab</td>
    <td>Browser</td>
    <td>Download only the main.ipynb and upload it to colab, or search in the github section of open notebook in colab for 'austineamonn/SummerResearch2024' and open main.ipynb</td>
    <td>Just taking a peek at program functionality or you don't have alot of space on your computer.</td>
  </tr>
</table>

After you have completed the inital set up there are some additional instructions in the main.ipynb file to finish setup.

### [Config - Under Construction](config.py):
Contains the basic configurations for the model. Most important is the ability to configure which parts of the model you want to run. The list you can pick from is: Generate Dataset, Privatize Dataset, Calculate Privacy Metrics.

## Datasets:

There was no available dataset that contained all the information required for this application. Thus, a purely synthetic dataset was created based on empirical distributions and known mappings. Unknown distributions were considered random. Unknown mappings were based on "common sense" connections. For example an Asian American student would be assumed to be more likely to join an Asian American Association college group over a student from a different ethnoracial group. Many of these connections were generated using a large language model.

### [Datafiles for Data Construction](src/IntelliShield/data_generation/datafiles_for_data_construction):
Various JSON files that have lists of data and feature tuples. This folder also contains the data.py file. The following table describes the JSON files of this folder.

<table>
  <tr>
    <th>File Name</th>
    <th>File Contents</th>
  </tr>
  <tr>
    <td>courses.json</td>
    <td>List of courses with course name, course number, course type, course subject, and course popularity</td>
  </tr>
  <tr>
    <td>first_names.json</td>
    <td>List of first names</td>
  </tr>
  <tr>
    <td>last_names.json</td>
    <td>List of last names</td>
  </tr>
  <tr>
    <td>majors.json</td>
    <td>List of majors with major name, percent of female identifying majors, major popularity, and top five careers for majors</td>
  </tr>
</table>

### [Data](src/IntelliShield/data_generation/datafiles_for_data_construction/data.py):
Dictionary that containts demographic information, lists of features, feature tuples, and mappings between various features of the dataset.

### Synthetic Dataset Generation:

Both CPU and GPU dataset generation generate the dataset in the same way. One is just optimized for CPU usage and the other is optimized for GPU usage. The following flowchart explains how one sample is generated.

<p align="center">
  <img src="/docs/graphics/canva_generated/data_generation_flowchart.png" width="1080" title="Data Generation Flowchart" alt="A flowchart chart explaining how the synthetic data is generated.">
</p>

For the Semester Loop, the elements inside are iterated for every semester the student has been at the school. Each iteration they all impact one another.

For the Pre-Loop Features, student semester impacts GPA only in that a 0th semester student, one who has not started school yet, cannot have a GPA.

### [Data Generation with a CPU](src/IntelliShield/data_generation/data_generation_CPU.py):
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

### [Data Generation with a GPU](src/IntelliShield/data_generation/data_generation_GPU.py):
Generates the synthetic dataset on the computer's GPU. The dataset contains the following elements: first name, last name, race or ethnicity, gender, international student status, socioeconomic status, learning style(s), gpa, student semester, major(s), previous courses, previous course types, course subjects, subjects of interest, career aspirations, extracurricular activities, and future topics.

```python
from data import Data
from config import load_config
from data_generation_GPU import DataGenerator

# Create generator class
generator = DataGenerator(data, config)

# Set generation levels
num_samples = 1000
batch_size = 100

# Returns a synthetic dataset with num_samples many rows ('students')
generator.generate_synthetic_dataset(num_samples, batch_size)
```

### Dataset
Synthetic dataset. The file was removed because it was too large, but you can generate as much data as you need using the data generation functions. Note that all the example files were created using a 100,000 length dataset.

Note: data category column is explained in the splitting the data section of data preprocessing.

<p align="center">
  <img src="/docs/graphics/canva_generated/data_construction.png" width="1080" title="Data Column Details" alt="A table giving the details of each data column.">
</p>

&ast; Chosen uniformly from 2.0 to 4.0.

## Dataset Analysis:

### [Dataset Analysis Python - Under Construction](src/IntelliShield/data_generation/data_analysis.py):
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

### [Dataset Analysis Python Graphs - Under Construction](outputs/examples/data_analysis_graphs):
This folder contains all the graphs produced by data_analysis.

### [Dataset Analysis R](tests/R_tests):

The R analysis aimed to determine the accuracy of the mappings and distributions of the purely synthetic dataset. This work was the final project for my Introduction to R for Biologists course.

This HTML file contains the analysis of the purely synthetic dataset including some interactive graphics.

[R Analysis HTML](tests/R_tests/final_project/final_project.html)

This paper is the final project for the course which analyzes the results shown in the above HTML file.

[Novel Generation of Purely Synthetic Educational Dataset Effective For Research Applications](tests/R_tests/final_report/final_report.pdf)

## Data Preprocessing:

Preprocessing is needed to convert the data into file types and structures that the machine learning models used in the tradeoffs section can easily learn from.

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

The following chart gives an overview for how the data is preprocessed and turned into two datasets. One dataset is used for the classification and regression models. The other dataset is used for the regressification models.

<p align="center">
  <img src="/docs/graphics/canva_generated/data_preprocessing.gif" width="1080" title="Data Preprocessing" alt="A flowchart explaining how the data is preprocessed.">
</p>

### [Preprocessing](src/IntelliShield/data_preprocessing/preprocessing.py):
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
preprocesser = PreProcessing(data, config)

# Returns preprocessed dataset
preprocessed_dataset = preprocesser.preprocess_dataset(synthetic_dataset)

# Create the RNN models and save them to their files
# Use one of these models to reduce the dimensionality
# of the preprocessed dataset
preprocessor.create_RNN_models(preprocessed_dataset, save_files=True)
```

### [Preprocessed Dataset](outputs/examples/preprocessed_data/Preprocessed_Dataset.csv):
All feature columns and utility columns have been converted into either binary lists or numerical lists. Contains 100,000 'students' in the CSV.

### [Processing the Private Columns](src/IntelliShield/calculating_tradeoffs/processing_private_columns.py):
The private data are converted into numbered lists. This currently only converts ethnoracial group, gender, and international status.

```python
from pandas import pd
from config import load_config
from datafiles_for_data_construction.data import Data
from processing_private_columns import PrivateColumns

# Import the synthetic dataset CSV as a pandas dataframes
synthetic_dataset = pd.read_csv(path_to_synthetic_dataset.csv')

# Create a private columns class
private_cols = PrivateColumns(data, config)

# Returns the processed private columns (ethnoracial group, gender, international student status)
private_cols.get_private_cols(synthetic_dataset)
```

### [Alternative Future Topic Preprocessing](src/IntelliShield/data_preprocessing/alternate_future_topics.py):
In this alternative preprocessing method, future topics are split into 5 columns with each column containing one of the five recommended future topics per student. Takes the preprocessed dataset as an input. This method is used for the targets for the the novel 'Regressification' models.

```python
from pandas import pd
from processing_private_columns import PrivateColumns

# Import preprocessed dataset CSV as a pandas dataframe
preprocessed_dataset = pd.readcsv('path_to_preprocessed_dataset.csv')

# Specify the inputs for the classifier
privatization_type = 'Shuffling'
RNN_model = 'GRU1'

# Initiate class instance
alt_topics_getter = AltFutureTopics(privatization_type, RNN_model)

# Returns the combined columns with the alternative future topic preprocessing
alt_topics_getter.int_list_to_separate_cols(preprocessed_dataset)
```

### [Reduced Dimensionality Files](outputs/examples/reduced_dimensionality_data):
In this folder, the different RNN models for dimensionality reduction can be found. They are organized by privatization method. Within each is the three methods Simple , GRU and LSTM all with 1 layer. The combined versions contain the preprocessed private columns and utility columns while the regular versions do not. The reduced dimension utility columns can be found on their own in the 'NoPrivatization' folder.

## Data Privatization

### Privatization Methods
There are two main methods:
<ol>
  <li>Differential Privacy which adds noise to the data to reach a privatization level specified by epsilon. Epsilon is set to 0.1 and the noise type is set to 'laplace'. List length changing (LLC) can be added.</li>
  <li>Random Shuffling which shuffles a set ratio of the data rows. The shuffle ratio is set to 10% but can be automatically set to 100% using the 'full shuffle' privatization method.</li>
</ol>
Overall, there is wide flexibility in the options for privatization method. The sensitivity for the differential privacy is calculated using the mean method.

<p align="center">
  <img src="/docs/graphics/canva_generated/privatization_methods.png" width="1080" title="Privatization Methods Flowchart" alt="A flowchart showing the different data privatization methods">
</p>

So what is list length changing (LLC)?

Typically differential privacy works with numerical data, sometimes converting other types into numerical representations. Since a large amount of the synthetic dataset features are lists, LLC was used before the lists were converted to a numerical representation.

LLC works by changing the length of a list based on the noise method. This is similar to how differential privacy works on numerical data except instead of changing the value by a certain amount based on the noise, LLC changes the list length based on the noise.

### [Privatization](src/IntelliShield/data_privatization/privatization.py):
Generates the privatized dataset based on the preprocessed dataset using one of the various methods listed above.

```python
from pandas import pd
from config import load_config
from privatization import Privatizer

# Import preprocessed dataset CSV as a pandas dataframe
preprocessed_dataset = pd.readcsv('path_to_preprocessed_dataset.csv')

# Create privatizer class using differential privacy with laplace noise addition, epsilon of 0.1 and no list length changing
privatizer = Privatizer(data, config, 'basic differential privacy', True)

# Returns privatized dataset
privatizer.privatize_dataset(preprocessed_dataset)
```

### [Privatized Datasets](outputs/examples/privatized_datasets):
There are a variety of privatized datasets including differential privacy with laplace and uniform noise addition and with and without list length changing (LLC) as well as the random shuffling at 10% and 100%. Here are the models that are saved in this section:

<ul>
  <li>Differential Privacy with Laplacian Noise</li>
  <li>Differential Privacy with Laplacian Noise and List Lengthening</li>
  <li>Differential Privacy with Uniform Noise</li>
  <li>Differential Privacy with Uniform Noise and List Lengthening</li>
  <li>Shuffling at 10%</li>
  <li>Complete Shuffling (100% of the rows are shuffled)</li>
</ul>

Note that changing privatization parameters in config can allow you to make different variations of these methods.

### [Privacy Metrics - Under Construction](src/IntelliShield/data_privatization/privacy_metrics.py):
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

## [Calculating Tradeoffs](src/IntelliShield/tradeoffs.py):

This is where the privacy - utility tradeoff calculations are run on all the different privatization styles. This is done through several model types:

Classification:

<ul>
  <li>Calculate the privacy loss on the private columns</li>
  <li>Based on various prebuilt models (ex: sklearn DecisionTreeClassifier)</li>
</ul>

Regression:

<ul>
  <li>Calculate the utility gain on the utility columns</li>
  <li>Based on various prebuilt models (ex: sklearn DecisionTreeRegressor)</li>
</ul>

Regressification:

<ul>
  <li>Calculate the utility gain on the split 'future topics' columns from alternative future topics preprocessing</li>
  <li>Train a regression model, then convert predictions to integers, then calculate metrics (like accuracy) using classification methods</li>
</ul>

Each model is represented as a different class. Then a variety of functions can be called depending on the model class. Here is a list of the models:

<table>
  <tr>
    <th>Model</th>
    <th>Model Type</th>
    <th>Class Name</th>
  </tr>
  <tr>
    <td>Decision Tree Classifier</td>
    <td>Classification</td>
    <td>ISDecisionTreeClassification</td>
  </tr>
  <tr>
    <td>Decision Tree Regressifier</td>
    <td>Regressification</td>
    <td>ISDecisionTreeRegressification</td>
  </tr>
  <tr>
    <td>Decision Tree Regressor</td>
    <td>Regression</td>
    <td>ISDecisionTreeRegression</td>
  </tr>
  <tr>
    <td>Random Forest Classifier</td>
    <td>Classification</td>
    <td>ISRandomForestClassification</td>
  </tr>
   <tr>
    <td>Random Forest Regressifier</td>
    <td>Regressification</td>
    <td>ISRandomForestRegressification</td>
  </tr>
   <tr>
    <td>Random Forest Regressor</td>
    <td>Regression</td>
    <td>ISRandomForestRegression</td>
  </tr>
  <tr>
    <td>Logistic Regressor</td>
    <td>Classification</td>
    <td>ISLogisticRegression</td>
  </tr>
  <tr>
    <td>Linear Regressifier</td>
    <td>Regressification</td>
    <td>ISLinearRegressification</td>
  </tr>
  <tr>
    <td>Linear Regressor</td>
    <td>Regression</td>
    <td>ISLinearRegression</td>
  </tr>
</table>

More models are to be added.

### Decision Tree Classifier:
Takes a dataset and uses a decision tree classifier to see how well the X columns can predict each private column. Specify the privatization method, private column target, and the dimensionality reduction method. The classifier can get the best ccp alpha value (based on maximum x test accuracy) and can also run a single decision tree based on the ccp alpha value. For both you need to specify how much data you want to read in and then run the test-train split.

The model, various graphs, and the predicted y values are all saved in the outputs folder.

```python
from pandas import pd
from decision_tree_classifier import DTClassifier

# Import the combined dataset CSV as a pandas dataframes
combined_dataset = pd.readcsv('path_to_combined_dataset.csv')

# Specify the inputs for the classifier
privatization_type = 'Shuffling'
RNN_model = 'GRU1'
target = 'gender'

# Create decision tree class
classifier = DTClassifier(privatization_type, RNN_model, target, combined_dataset)

# Get the best ccp alpha value
ccp_alpha = classifier.get_best_model(return_model=False)

# Run the full model - saves graphs, model, and more in the outputs folder
classifier.run_model(ccp_alpha=ccp_alpha)
```

### Decision Tree Classifier Outputs

When the ISDecisionTreeClassifier class is run a variety of outputs can be produced. For outputs to be produced <em> an output path MUST be declared </em>.

The recommended organization method, is to organized first by privatization type, then by dimensionality reduction type, and then by target. What is saved in each folder:

<ul>
  <li>All the models (unfitted) from the ccp alpha calculation are saved with some statistics like accuracy, depth, and nodes</li>
  <li>The best ccp alpha fit model</li>
  <li>The y predictions based on the best model</li>
  <li>The classification report from the best model (with an added runtime value)</li>
  <li>The decision tree image from the best model</li>
  <li>Alpha vs accuracy</li>
  <li>Alpha vs graph nodes and graph depth</li>
  <li>Alpha vs total impurity</li>
</ul>

Note that what items are made and saved can be changed by altering inputs for the functions.

Here are some examples of what the graphs could look like for the basic differntial privacy method X GRU 1 layer dimensionality reduction model X target: ethnoracial group. Similar graphics can be produced for other combinations.

Alpha vs Accuracy:
<p align="center">
  <img src="/docs/graphics/decision_tree_classifier_example/effective_alpha_vs_accuracy.png" width="1080" title="Alpha vs Accuracy" alt="A graph comparing alpha and accuracy">
</p>

Alpha vs Graph Nodes and Depth:
<p align="center">
  <img src="/docs/graphics/decision_tree_classifier_example/effective_alpha_vs_graph_nodes_and_depth.png" width="1080" title="Alpha vs Graph Nodes and Depth" alt="Two graphs, one comparing alpha and graph nodes, the other comparing alpha and tree depth">
</p>

Alpha vs Total Impurity:
<p align="center">
  <img src="/docs/graphics/decision_tree_classifier_example/effective_alpha_vs_total_impurity.png" width="1080" title="Alpha vs Total Impurity" alt="A graph comparing alpha and impurity">
</p>

The first few splits of the best decision tree.
<p align="center">
  <img src="/docs/graphics/decision_tree_classifier_example/decision_tree_classifier.png" width="1080" title="Decision Tree Example" alt="A graphic showing the organization of a decision tree">
</p>

These SHAP value plots are specific to the Multiracial ethnoracial group. The same type of graphs can be created for all the other ethnoracial groups.

SHAP Feature Importance:
<p align="center">
  <img src="/docs/graphics/decision_tree_classifier_example/shap_bar_plot.png" width="1080" title="Decision Tree SHAP Feature Importance" alt="A bar chart comparing the feature importances for the decision tree model">
</p>

SHAP Bee Swarm Plot:
<p align="center">
  <img src="/docs/graphics/decision_tree_classifier_example/shap_bee_swarm_plot.png" width="1080" title="Decision Tree SHAP Bee Swarm Plot" alt="A bee swarm plot comparing the feature importances for the decision tree model">
</p>

SHAP Heatmap Plot:
<p align="center">
  <img src="/docs/graphics/decision_tree_classifier_example/shap_heatmap.png" width="1080" title="Decision Tree SHAP Heatmap Plot" alt="A heatmap comparing the feature importances for the decision tree model">
</p>

SHAP Violin Plot:
<p align="center">
  <img src="/docs/graphics/decision_tree_classifier_example/shap_violin_plot.png" width="1080" title="Decision Tree SHAP Violin Plot" alt="A violin plot comparing the feature importances for the decision tree model">
</p>

There are also scatter plots for each feature that compare feature value and SHAP value.

Course Subject Scatter Plot:
<p align="center">
  <img src="/docs/graphics/decision_tree_classifier_example/feature_scatter_plots/course subjects.png" width="1080" title="Course Subject Scatter Plot" alt="A scatter plot for the course subject feature of the model">
</p>

Student Semester Scatter Plot:
<p align="center">
  <img src="/docs/graphics/decision_tree_classifier_example/feature_scatter_plots/student semester.png" width="1080" title="Student Semester Scatter Plot" alt="A scatter plot for the student semester feature of the model">
</p>

### Decision Tree Alternate:
Takes a dataset and uses a decision tree alternate to see how well the X columns can predict future topics as five split columns. This builds off of the alternate preprocessing path for the future topics column that split it into 5 columns. Specify the privatization method, private column target, and the dimensionality reduction method. The classifier can get the best ccp alpha value (based on maximum x test accuracy) and can also run a single decision tree based on the ccp alpha value. For both you need to specify how much data you want to read in and then run the test-train split.

The model, various graphs, and the predicted y values are all saved in the outputs folder.

```python
from pandas import pd
from decision_tree_classifier import DTClassifier

# Import the alternate dataset CSV as a pandas dataframes
alternate_dataset = pd.readcsv('path_to_alternate_dataset.csv')

# Specify the inputs for the alternate model (regressifier)
privatization_type = 'Shuffling'
RNN_model = 'GRU1'
target = 'future topic 1'

# Create decision tree class
alternate = DTAlternate(privatization_type, RNN_model, target, alternate_dataset)

# Get the best ccp alpha value
ccp_alpha = alternate.get_best_model(return_model=False)

# Run the full model - saves graphs, model, and more in the outputs folder
alternate.run_model(ccp_alpha=ccp_alpha)
```

### Decision Tree Alternate Outputs:

Contains a variety of outputs from the decision tree alternate function. Organized first by privatization type, then by dimensionality reduction type, and then by target. What is saved in each folder:

<ul>
  <li>All the models (unfitted) from the ccp alpha calculation are saved with some statistics like accuracy, depth, and nodes</li>
  <li>The best ccp alpha fit model</li>
  <li>The y predictions based on the best model</li>
  <li>The classification report from the best model (with an added runtime value)</li>
  <li>The decision tree image from the best model</li>
  <li>Alpha vs accuracy</li>
  <li>Alpha vs graph nodes and graph depth</li>
  <li>Alpha vs total impurity</li>
</ul>

Note that what items are made and saved can be changed by altering inputs for the functions.

### Decision Tree Regressor:
Takes a dataset and uses a decision tree regressor to see how well the X columns can predict each utility column. Specify the privatization method, private column target, and the dimensionality reduction method. The classifier can get the best ccp alpha value (based on maximum x test accuracy) and can also run a single decision tree based on the ccp alpha value. For both you need to specify how much data you want to read in and then run the test-train split.

The model, various graphs, and the predicted y values are all saved in the outputs folder.

```python
from pandas import pd
from decision_tree_regression import DTRegressor

# Import the combined dataset CSV as a pandas dataframes
combined_dataset = pd.readcsv('path_to_combined_dataset.csv')

# Specify the inputs for the regressor
privatization_type = 'Shuffling'
RNN_model = 'GRU1'
target = 'gender'

# Create decision tree class
regressor = DTRegressor(privatization_type, RNN_model, target, combined_dataset)

# Get the best ccp alpha value
ccp_alpha = regressor.get_best_model(return_model=False)

# Run the full model - saves graphs, model, and more in the outputs folder
classifier.run_model(ccp_alpha=ccp_alpha)
```

### Decision Tree Regressor Outputs:

Contains a variety of outputs from the decision tree regression function. Organized first by privatization type, then by dimensionality reduction type, and then by target. What is saved in each folder:

<ul>
  <li>All the models (unfitted) from the ccp alpha calculation are saved with some statistics like accuracy, depth, and nodes</li>
  <li>The best ccp alpha fit model</li>
  <li>The y predictions based on the best model</li>
  <li>The metrics CSV which includes: mean squared error, root mean squared error, mean absolute error, median absolute error, r2 score, explained variance score, mean bias deviation, runtime</li>
  <li>The decision tree image from the best model</li>
  <li>Alpha vs accuracy</li>
  <li>Alpha vs graph nodes and graph depth</li>
  <li>Alpha vs total impurity</li>
</ul>

Note that what items are made and saved can be changed by altering inputs for the functions.

## Model Comparison:

Now that we have various models trained on the various datasets we need to compare them to decide which dimensionality reduction method was best and which privatization method was best.

### [Classification Comparison](src/IntelliShield/model_comparison/classification/comparison_classification.py):
This file compares the classification metrics of the various classification machine learning models.

### [Classification Regression](src/IntelliShield/model_comparison/regression/comparison_regression.py):
This file compares the regression metrics of the various regression machine learning models.

### [Classification Alternate](src/IntelliShield/model_comparison/alternate/comparison_alternate.py):
This file compares the alternate metrics (same as classification) of the various alternate machine learning models.

## Sources and Acknowledgments:

### Data Sources:

A variety of sources were used to generate the synthetic dataset. More details can be found in the Dataset Generation - Dataset section.

<ol>
  <li>https://discovery.cs.illinois.edu/dataset/course-catalog/ - Course Catalog with Course Names, Course Types, and Course Subject Abbreviations.</li>
  <li>https://educationdata.org/college-enrollment-statistics - College Demographic Statistics</li>
  <li>https://files.eric.ed.gov/fulltext/EJ1192524.pdf - Learning Style Statistics</li>
  <li>https://data.world/len/us-first-names-database - First and Last Names Database</li>
  <li>https://www.pewresearch.org/social-trends/2019/05/22/a-rising-share-of-undergraduates-are-from-poor-families-especially-at-less-selective-colleges/ - College Family Income Statistics</li>
  <li>https://williamsinstitute.law.ucla.edu/publications/nonbinary-lgbtq-adults-us/ - Nonbinary Statistics</li>
  <li>https://nces.ed.gov - Gender College Statistics</li>
  <li>https://courses.illinois.edu/schedule/DEFAULT/DEFAULT - Course Subject Abbreviation to Course Subject Mapping</li>
  <li>https://bigeconomics.org/college-majors-explorer/ - List of Majors, Careers</li>
</ol>

### Acknowledgements:

We thank the support of NSF REU #1852042 and #2149956, and the support from the Office of Vice Chancellor of Research at the University of Tennessee at Chattanooga.

Python GPT and Chat GPT4o assisted in the programming process.
