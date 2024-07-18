Data Generation
===============

There was no available dataset that contained all the information required for this application. Thus, data was generated synthetically with a basis in real data.

Datafiles for Data Construction:
--------------------------------
Various JSON files that have lists of data and feature tuples. This folder also contains the data.py file. The following table describes the JSON files of this folder.

| File Name        | File Contents                                                                                 |
|------------------|-----------------------------------------------------------------------------------------------|
| courses.json     | List of courses with course name, course number, course type, course subject, and course popularity |
| first_names.json | List of first names                                                                           |
| last_names.json  | List of last names                                                                            |
| majors.json      | List of majors with major name, percent of female identifying majors, major popularity, and top five careers for majors |

Data:
-----
Dictionary that contains demographic information, lists of features, feature tuples, and mappings between various features of the dataset.

Data Generation with a CPU:
---------------------------
Generates the synthetic dataset on the computer's CPU. The dataset contains the following elements: first name, last name, race or ethnicity, gender, international student status, socioeconomic status, learning style(s), gpa, student semester, major(s), previous courses, previous course types, course subjects, subjects of interest, career aspirations, extracurricular activities, and future topics.

.. code-block:: python

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

Data Generation with a GPU:
---------------------------
Generates the synthetic dataset on the computer's GPU. The dataset contains the following elements: first name, last name, race or ethnicity, gender, international student status, socioeconomic status, learning style(s), gpa, student semester, major(s), previous courses, previous course types, course subjects, subjects of interest, career aspirations, extracurricular activities, and future topics.

.. code-block:: python

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

Data Analysis - Under Construction:
-----------------------------------
Takes the synthetic dataset and produces various graphs about the data. For the numerical columns boxplots, distributions, and summary statistics are produced. For all the other columns the top ten highest count items are displayed. Calculates the percentage of empty or NaN values in each column.

.. code-block:: python

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

Data Analysis Graphs - Under Construction:
------------------------------------------
This folder contains all the graphs produced by data_analysis.

Dataset:
--------
Synthetic dataset. The file was removed because it was too large, but you can generate as much data as you need using the data generation functions. Note that all the example files were created using a 100,000 length dataset.

.. image:: /docs/graphics/canva_generated/data_construction.png
   :width: 1080
   :alt: A chart giving the details of each data column
   :align: center
