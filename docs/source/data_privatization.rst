Data Privatization
==================

Privatization Methods:
----------------------
There are two main methods:
1. Differential Privacy which adds noise to the data to reach a privatization level specified by epsilon. Epsilon is set to 0.1 and the noise type is set to 'laplace'. List length changing (LLC) can be added.
2. Random Shuffling which shuffles a set ratio of the data rows. The shuffle ratio is set to 10% but can be automatically set to 100% using the 'full shuffle' privatization method.

Overall, there is wide flexibility in the options for privatization method. The sensitivity for the differential privacy is calculated using the mean method.

.. image:: /docs/graphics/canva_generated/privatization_methods.png
   :width: 1080
   :alt: A flowchart showing the different data privatization methods
   :align: center

So what is list length changing (LLC)?

Typically differential privacy works with numerical data, sometimes converting other types into numerical representations. Since a large amount of the synthetic dataset features are lists, LLC was used before the lists were converted to a numerical representation.

LLC works by changing the length of a list based on the noise method. This is similar to how differential privacy works on numerical data except instead of changing the value by a certain amount based on the noise, LLC changes the list length based on the noise.

Privatization:
--------------
Generates the privatized dataset based on the preprocessed dataset using one of the various methods listed above.

.. code-block:: python

    from pandas import pd
    from config import load_config
    from privatization import Privatizer

    # Import preprocessed dataset CSV as a pandas dataframe
    preprocessed_dataset = pd.readcsv('path_to_preprocessed_dataset.csv')

    # Create privatizer class using differential privacy with laplace noise addition, epsilon of 0.1 and no list length changing
    privatizer = Privatizer(data, config, 'basic differential privacy', True)

    # Returns privatized dataset
    privatizer.privatize_dataset(preprocessed_dataset)

Privatized Datasets:
--------------------
There are a variety of privatized datasets including differential privacy with laplace and uniform noise addition and with and without list length changing (LLC) as well as the random shuffling at 10% and 100%. Here are the models that are saved in this section:

- Differential Privacy with Laplacian Noise
- Differential Privacy with Laplacian Noise and List Lengthening
- Differential Privacy with Uniform Noise
- Differential Privacy with Uniform Noise and List Lengthening
- Shuffling at 10%
- Complete Shuffling (100% of the rows are shuffled)

Note that changing privatization parameters in config can allow you to make different variations of these methods.

Privacy Metrics - Under Construction:
-------------------------------------
Calculates the level of data privatization using various metrics: Mean comparison, STD comparison, and Sum comparison. Also outputs the privatization method used and the parameters of the method.

.. code-block:: python

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
