Data Preprocessing
==================

Preprocessing is needed to convert the data into file types and structures that the machine learning models used in the tradeoffs section can easily learn from.

Splitting the Data:
-------------------
| Data Category | Features                                                                                           | Explanation                                                                                                     |
|---------------|----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| Xp            | first name, last name, race or ethnicity, gender, international student status, and socioeconomic status | Xp columns are cut out and removed as we want to keep these hidden and they would be useless for determining a student's career aspirations or future topics. |
| X             | learning style(s), gpa, student semester, major(s), previous courses, previous course types, course subjects, subjects of interest, and extracurricular activities | X columns are privatized using various techniques. These will also be the features for the neural network.      |
| Xu            | career aspirations and future topics                                                               | Xu columns are left alone. These utility columns are the targets for the neural network.                        |

Preprocessing:
--------------
`preprocess_dataset()`: Takes in a synthetic dataset. Xp is cut out, X and Xu are converted from lists of strings to lists of numbers. Outputs a preprocessed dataset.

`run_RNN_models()`: Takes in a preprocessed dataset. For each list in each column, the lists are padded so they become the same length. Then an RNN is run to reduce dimensionality such that each column becomes 1 dimensional. There are 3 RNN types (Simple, GRU, and LSTM) and they can run with different numbers of layers (1-4).

.. code-block:: python

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

Preprocessed Dataset:
---------------------
All feature columns and utility columns have been converted into either binary lists or numerical lists. Contains 100,000 'students' in the CSV.

Processing the Private Columns:
-------------------------------
The private data are converted into numbered lists. This currently only converts ethnoracial group, gender, and international status.

.. code-block:: python

    from pandas import pd
    from config import load_config
    from datafiles_for_data_construction.data import Data
    from processing_private_columns import PrivateColumns

    # Import the synthetic dataset CSV as a pandas dataframe
    synthetic_dataset = pd.read_csv('path_to_synthetic_dataset.csv')

    # Create a private columns class
    private_cols = PrivateColumns(data, config)

    # Returns the processed private columns (ethnoracial group, gender, international student status)
    private_cols.get_private_cols(synthetic_dataset)

Alternative Future Topic Preprocessing:
---------------------------------------
In this alternative preprocessing method, future topics are split into 5 columns with each column containing one of the five recommended future topics per student. Takes the preprocessed dataset as an input. This method is used for the targets for the novel 'Regressification' models.

.. code-block:: python

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

Reduced Dimensionality Files:
-----------------------------
In this folder, the different RNN models for dimensionality reduction can be found. They are organized by privatization method. Within each is the three methods Simple, GRU, and LSTM all with 1 layer. The combined versions contain the preprocessed private columns and utility columns while the regular versions do not. The reduced dimension utility columns can be found on their own in the 'NoPrivatization' folder.
