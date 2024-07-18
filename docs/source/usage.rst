Usage
=====

Main Functions:
---------------
All three of these files were lost in the restructuring but they will be revived soon.

Main - Under Construction:
--------------------------
The main file of the framework. What can this file do?
- Generates a synthetic dataset using either 'data_generation_CPU' or 'data_generation_GPU'
- Analyzes the dataset using 'data_analysis'
- Preprocesses and reduces the dimensionality of the dataset using 'preprocessing'
- Privatizes the dataset using 'privatization'
- Calculates the privacy metrics using 'privacy_metrics'
- Calculates the utility gain using regression models from 'calculating tradeoffs/regression'
- Calculates the utility gain for 'future topics' using alternate models from 'calculating tradeoffs/alternate'
- Calculates the privacy loss using classification models from 'calculating tradeoffs/classification'

Use the config file to change which of the above parts of the file are run during main. You don't need to run all of them but they will run in this order.

Interactive Main - Under Construction:
--------------------------------------
An interactive Jupyter Notebook that walks through the full data pipeline process from data generation to privacy - utility tradeoffs. Essentially, this notebook follows what 'main.py' does but on a smaller, more informative, and more interactive scale. This file is just to explain how the code works. *None of the files produced are saved*. There are two ways to run this code:

| How to Run File       | Where is it Run? | Initial Set Up                                                                                       | Which should I use?                                                                                  |
|-----------------------|------------------|------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| Jupyter Notebook      | Locally          | Clone this github into a virtual environment.                                                        | Want to first explore and then use the full program functionality.                                    |
| Google Colab          | Browser          | Download only the main.ipynb and upload it to colab, or search in the github section of open notebook | Just taking a peek at program functionality or you don't have a lot of space on your computer.        |
|                       |                  | in colab for 'austineamonn/SummerResearch2024' and open main.ipynb                                    |                                                                                                       |

After you have completed the initial set up there are some additional instructions in the main.ipynb file to finish setup.

Config - Under Construction:
-----------------------------
Contains the basic configurations for the model. Most important is the ability to configure which parts of the model you want to run. The list you can pick from is: Generate Dataset, Privatize Dataset, Calculate Privacy Metrics.
