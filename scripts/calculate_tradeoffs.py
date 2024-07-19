import logging
import cProfile
import pstats
import pandas as pd
from ast import literal_eval
from src.IntelliShield.tradeoffs import ISLogisticRegression, ISDecisionTreeClassification, ISLinearRegression, ISDecisionTreeRegressification, ISDecisionTreeRegression, ISRandomForestClassification, ISRandomForestRegression, ISRandomForestRegressification, ISLinearRegressification, pipeline

"""
Script to get the tradeoff values from each model. Will run the model, make the graphics, calculate the feature importance, and save it all to the outputs folder. Requires a data folder that contains the reduced dimensionality data. Assumes a structure of:

"reduced_dimensionality_folder/privatization_type/reduced_dimensionality_type_combined.csv"
"""

# Classification Models

def get_tradeoffs_ISDecisionTreeClassification(data_folder, output_path=None):
    # Get output path
    if output_path == None:
        output_path = 'outputs/testing_tradeoffs/decision_tree_classification'

    # List of RNN models to run
    RNN_model_list = ['GRU1', 'LSTM1', 'Simple1']

    # List of Privatization Types to run
    privatization_types = ['NoPrivatization', 'Basic_DP', 'Basic_DP_LLC', 'Uniform', 'Uniform_LLC', 'Shuffling', 'Complete_Shuffling']

    # List of targets for the model 
    targets = ['ethnoracial group', 'gender', 'international status', 'socioeconomic status']

    # Get the runtime values for the function
    profiler = cProfile.Profile()
    profiler.enable()

    for privatization_type in privatization_types:
        logging.info(f"Starting {privatization_type}")
        for RNN_model in RNN_model_list:
            logging.info(f"Starting {RNN_model}")
            
            # Data Path
            data_path = f'{data_folder}/{privatization_type}/{RNN_model}_combined.csv'

            # Data
            data = pd.read_csv(data_path, converters={
                'learning style': literal_eval,
                'major': literal_eval,
                'previous courses': literal_eval,
                'course types': literal_eval,
                'course subjects': literal_eval,
                'subjects of interest': literal_eval,
                'extracurricular activities': literal_eval,
                'career aspirations': literal_eval,
                'future topics': literal_eval
            })

            for target in targets:
                logging.info(f"Starting {target}")

                target_name = target.replace(' ', '_')

                target_path = f'{output_path}/outputs/{privatization_type}/{RNN_model}/{target_name}'

                # Initiate classifier
                classifier = ISDecisionTreeClassification(privatization_type, RNN_model, target, data=data, output_path=target_path)
                pipeline(classifier, full_run=True)

    # Save the profiling stats to a file
    profile_stats_file = f"{output_path}/profile_stats.txt"
    with open(profile_stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats()

def get_tradeoffs_ISLogisticRegression(data_folder, output_path=None):
    # Get output path
    if output_path == None:
        output_path = 'outputs/testing_tradeoffs/logistic_regression'

    # List of RNN models to run
    RNN_model_list = ['GRU1', 'LSTM1', 'Simple1']

    # List of Privatization Types to run
    privatization_types = ['NoPrivatization', 'Basic_DP', 'Basic_DP_LLC', 'Uniform', 'Uniform_LLC', 'Shuffling', 'Complete_Shuffling']

    # List of targets for the model 
    targets = ['ethnoracial group', 'gender', 'international status', 'socioeconomic status']

    # Get the runtime values for the function
    profiler = cProfile.Profile()
    profiler.enable()

    for privatization_type in privatization_types:
        logging.info(f"Starting {privatization_type}")
        for RNN_model in RNN_model_list:
            logging.info(f"Starting {RNN_model}")
            
            # Data Path
            data_path = f'{data_folder}/{privatization_type}/{RNN_model}_combined.csv'

            # Data
            data = pd.read_csv(data_path, converters={
                'learning style': literal_eval,
                'major': literal_eval,
                'previous courses': literal_eval,
                'course types': literal_eval,
                'course subjects': literal_eval,
                'subjects of interest': literal_eval,
                'extracurricular activities': literal_eval,
                'career aspirations': literal_eval,
                'future topics': literal_eval
            })

            for target in targets:
                logging.info(f"Starting {target}")

                target_name = target.replace(' ', '_')

                target_path = f'{output_path}/outputs/{privatization_type}/{RNN_model}/{target_name}'

                # Initiate regressor
                regressor = ISLogisticRegression(privatization_type, RNN_model, target, data=data, output_path=target_path)
                pipeline(regressor, full_run=True)

    # Save the profiling stats to a file
    profile_stats_file = f"{output_path}/profile_stats.txt"
    with open(profile_stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats()

def get_tradeoffs_ISRandomForestClassification(data_folder, output_path=None):
    # Get output path
    if output_path == None:
        output_path = 'outputs/testing_tradeoffs/randon_forest_classification'

    # List of RNN models to run
    RNN_model_list = ['GRU1', 'LSTM1', 'Simple1']

    # List of Privatization Types to run
    privatization_types = ['NoPrivatization', 'Basic_DP', 'Basic_DP_LLC', 'Uniform', 'Uniform_LLC', 'Shuffling', 'Complete_Shuffling']

    # List of targets for the model 
    targets = ['ethnoracial group', 'gender', 'international status', 'socioeconomic status']

    # Get the runtime values for the function
    profiler = cProfile.Profile()
    profiler.enable()

    for privatization_type in privatization_types:
        logging.info(f"Starting {privatization_type}")
        for RNN_model in RNN_model_list:
            logging.info(f"Starting {RNN_model}")
            
            # Data Path
            data_path = f'{data_folder}/{privatization_type}/{RNN_model}_combined.csv'

            # Data
            data = pd.read_csv(data_path, converters={
                'learning style': literal_eval,
                'major': literal_eval,
                'previous courses': literal_eval,
                'course types': literal_eval,
                'course subjects': literal_eval,
                'subjects of interest': literal_eval,
                'extracurricular activities': literal_eval,
                'career aspirations': literal_eval,
                'future topics': literal_eval
            })

            for target in targets:
                logging.info(f"Starting {target}")

                target_name = target.replace(' ', '_')

                target_path = f'{output_path}/outputs/{privatization_type}/{RNN_model}/{target_name}'

                # Initiate classifier
                classifier = ISRandomForestClassification(privatization_type, RNN_model, target, data=data, output_path=target_path)
                pipeline(classifier, full_run=True)

    # Save the profiling stats to a file
    profile_stats_file = f"{output_path}/profile_stats.txt"
    with open(profile_stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats()

# Regressification Models

def get_tradeoffs_ISDecisionTreeRegressification(data_folder, output_path=None):
    # Get output path
    if output_path == None:
        output_path = 'outputs/testing_tradeoffs/decision_tree_regressification'

    # List of RNN models to run
    RNN_model_list = ['GRU1', 'LSTM1', 'Simple1']

    # List of Privatization Types to run
    privatization_types = ['NoPrivatization', 'Basic_DP', 'Basic_DP_LLC', 'Uniform', 'Uniform_LLC', 'Shuffling', 'Complete_Shuffling']

    # List of targets for the model 
    targets = ['future topic 1', 'future topic 2', 'future topic 3', 'future topic 4', 'future topic 5']

    # Get the runtime values for the function
    profiler = cProfile.Profile()
    profiler.enable()

    for privatization_type in privatization_types:
        logging.info(f"Starting {privatization_type}")
        for RNN_model in RNN_model_list:
            logging.info(f"Starting {RNN_model}")
            
            # Data Path
            data_path = f'{data_folder}/{privatization_type}/{RNN_model}_alt_future_topics.csv'

            # Data
            data = pd.read_csv(data_path, converters={
                'learning style': literal_eval,
                'major': literal_eval,
                'previous courses': literal_eval,
                'course types': literal_eval,
                'course subjects': literal_eval,
                'subjects of interest': literal_eval,
                'extracurricular activities': literal_eval,
                'career aspirations': literal_eval,
                'future topics': literal_eval
            })

            for target in targets:
                logging.info(f"Starting {target}")

                target_name = target.replace(' ', '_')

                target_path = f'{output_path}/outputs/{privatization_type}/{RNN_model}/{target_name}'

                # Initiate regressifier
                regressifier = ISDecisionTreeRegressification(privatization_type, RNN_model, target, data=data, output_path=target_path)
                pipeline(regressifier, full_run=True)

    # Save the profiling stats to a file
    profile_stats_file = f"{output_path}/profile_stats.txt"
    with open(profile_stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats()

def get_tradeoffs_ISLinearRegressification(data_folder, output_path=None):
    # Get output path
    if output_path == None:
        output_path = 'outputs/testing_tradeoffs/linear_regressification'

    # List of RNN models to run
    RNN_model_list = ['GRU1', 'LSTM1', 'Simple1']

    # List of Privatization Types to run
    privatization_types = ['NoPrivatization', 'Basic_DP', 'Basic_DP_LLC', 'Uniform', 'Uniform_LLC', 'Shuffling', 'Complete_Shuffling']

    # List of targets for the model 
    targets = ['future topic 1', 'future topic 2', 'future topic 3', 'future topic 4', 'future topic 5']

    # Get the runtime values for the function
    profiler = cProfile.Profile()
    profiler.enable()

    for privatization_type in privatization_types:
        logging.info(f"Starting {privatization_type}")
        for RNN_model in RNN_model_list:
            logging.info(f"Starting {RNN_model}")
            
            # Data Path
            data_path = f'{data_folder}/{privatization_type}/{RNN_model}_alt_future_topics.csv'

            # Data
            data = pd.read_csv(data_path, converters={
                'learning style': literal_eval,
                'major': literal_eval,
                'previous courses': literal_eval,
                'course types': literal_eval,
                'course subjects': literal_eval,
                'subjects of interest': literal_eval,
                'extracurricular activities': literal_eval,
                'career aspirations': literal_eval,
                'future topics': literal_eval
            })

            for target in targets:
                logging.info(f"Starting {target}")

                target_name = target.replace(' ', '_')

                target_path = f'{output_path}/outputs/{privatization_type}/{RNN_model}/{target_name}'

                # Initiate regressifier
                regressifier = ISLinearRegressification(privatization_type, RNN_model, target, data=data, output_path=target_path)
                pipeline(regressifier, full_run=True)

    # Save the profiling stats to a file
    profile_stats_file = f"{output_path}/profile_stats.txt"
    with open(profile_stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats()

def get_tradeoffs_ISRandomForestRegressification(data_folder, output_path=None):
    # Get output path
    if output_path == None:
        output_path = 'outputs/testing_tradeoffs/random_forest_regressification'

    # List of RNN models to run
    RNN_model_list = ['GRU1', 'LSTM1', 'Simple1']

    # List of Privatization Types to run
    privatization_types = ['NoPrivatization', 'Basic_DP', 'Basic_DP_LLC', 'Uniform', 'Uniform_LLC', 'Shuffling', 'Complete_Shuffling']

    # List of targets for the model 
    targets = ['future topic 1', 'future topic 2', 'future topic 3', 'future topic 4', 'future topic 5']

    # Get the runtime values for the function
    profiler = cProfile.Profile()
    profiler.enable()

    for privatization_type in privatization_types:
        logging.info(f"Starting {privatization_type}")
        for RNN_model in RNN_model_list:
            logging.info(f"Starting {RNN_model}")
            
            # Data Path
            data_path = f'{data_folder}/{privatization_type}/{RNN_model}_alt_future_topics.csv'

            # Data
            data = pd.read_csv(data_path, converters={
                'learning style': literal_eval,
                'major': literal_eval,
                'previous courses': literal_eval,
                'course types': literal_eval,
                'course subjects': literal_eval,
                'subjects of interest': literal_eval,
                'extracurricular activities': literal_eval,
                'career aspirations': literal_eval,
                'future topics': literal_eval
            })

            for target in targets:
                logging.info(f"Starting {target}")

                target_name = target.replace(' ', '_')

                target_path = f'{output_path}/outputs/{privatization_type}/{RNN_model}/{target_name}'

                # Initiate regressifier
                regressifier = ISRandomForestRegressification(privatization_type, RNN_model, target, data=data, output_path=target_path)
                pipeline(regressifier, full_run=True)

    # Save the profiling stats to a file
    profile_stats_file = f"{output_path}/profile_stats.txt"
    with open(profile_stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats()

# Regression Models

def get_tradeoffs_ISDecisionTreeRegression(data_folder, output_path=None):
    # Get output path
    if output_path == None:
        output_path = 'outputs/testing_tradeoffs/decision_tree_regression'

    # List of RNN models to run
    RNN_model_list = ['GRU1', 'LSTM1', 'Simple1']

    # List of Privatization Types to run
    privatization_types = ['NoPrivatization', 'Basic_DP', 'Basic_DP_LLC', 'Uniform', 'Uniform_LLC', 'Shuffling', 'Complete_Shuffling']

    # List of targets for the model 
    targets = ['career aspirations', 'future topics']

    # Get the runtime values for the function
    profiler = cProfile.Profile()
    profiler.enable()

    for privatization_type in privatization_types:
        logging.info(f"Starting {privatization_type}")
        for RNN_model in RNN_model_list:
            logging.info(f"Starting {RNN_model}")
            
            # Data Path
            data_path = f'{data_folder}/{privatization_type}/{RNN_model}_combined.csv'

            # Data
            data = pd.read_csv(data_path, converters={
                'learning style': literal_eval,
                'major': literal_eval,
                'previous courses': literal_eval,
                'course types': literal_eval,
                'course subjects': literal_eval,
                'subjects of interest': literal_eval,
                'extracurricular activities': literal_eval,
                'career aspirations': literal_eval,
                'future topics': literal_eval
            })

            for target in targets:
                logging.info(f"Starting {target}")

                target_name = target.replace(' ', '_')

                target_path = f'{output_path}/outputs/{privatization_type}/{RNN_model}/{target_name}'

                # Initiate regressor
                regressor = ISDecisionTreeRegression(privatization_type, RNN_model, target, data=data, output_path=target_path)
                pipeline(regressor, full_run=True)

    # Save the profiling stats to a file
    profile_stats_file = f"{output_path}/profile_stats.txt"
    with open(profile_stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats()

def get_tradeoffs_ISLinearRegression(data_folder, output_path=None):
    # Get output path
    if output_path == None:
        output_path = 'outputs/testing_tradeoffs/linear_regression'

    # List of RNN models to run
    RNN_model_list = ['GRU1', 'LSTM1', 'Simple1']

    # List of Privatization Types to run
    privatization_types = ['NoPrivatization', 'Basic_DP', 'Basic_DP_LLC', 'Uniform', 'Uniform_LLC', 'Shuffling', 'Complete_Shuffling']

    # List of targets for the model 
    targets = ['career aspirations', 'future topics']

    # Get the runtime values for the function
    profiler = cProfile.Profile()
    profiler.enable()

    for privatization_type in privatization_types:
        logging.info(f"Starting {privatization_type}")
        for RNN_model in RNN_model_list:
            logging.info(f"Starting {RNN_model}")
            
            # Data Path
            data_path = f'{data_folder}/{privatization_type}/{RNN_model}_combined.csv'

            # Data
            data = pd.read_csv(data_path, converters={
                'learning style': literal_eval,
                'major': literal_eval,
                'previous courses': literal_eval,
                'course types': literal_eval,
                'course subjects': literal_eval,
                'subjects of interest': literal_eval,
                'extracurricular activities': literal_eval,
                'career aspirations': literal_eval,
                'future topics': literal_eval
            })

            for target in targets:
                logging.info(f"Starting {target}")

                target_name = target.replace(' ', '_')

                target_path = f'{output_path}/outputs/{privatization_type}/{RNN_model}/{target_name}'

                # Initiate regressor
                regressor = ISLinearRegression(privatization_type, RNN_model, target, data=data, output_path=target_path)
                pipeline(regressor, full_run=True)

    # Save the profiling stats to a file
    profile_stats_file = f"{output_path}/profile_stats.txt"
    with open(profile_stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats()

def get_tradeoffs_ISRandomForestRegression(data_folder, output_path=None):
    # Get output path
    if output_path == None:
        output_path = 'outputs/testing_tradeoffs/random_forest_regression'

    # List of RNN models to run
    RNN_model_list = ['GRU1', 'LSTM1', 'Simple1']

    # List of Privatization Types to run
    privatization_types = ['NoPrivatization', 'Basic_DP', 'Basic_DP_LLC', 'Uniform', 'Uniform_LLC', 'Shuffling', 'Complete_Shuffling']

    # List of targets for the model 
    targets = ['career aspirations', 'future topics']

    # Get the runtime values for the function
    profiler = cProfile.Profile()
    profiler.enable()

    for privatization_type in privatization_types:
        logging.info(f"Starting {privatization_type}")
        for RNN_model in RNN_model_list:
            logging.info(f"Starting {RNN_model}")
            
            # Data Path
            data_path = f'{data_folder}/{privatization_type}/{RNN_model}_combined.csv'

            # Data
            data = pd.read_csv(data_path, converters={
                'learning style': literal_eval,
                'major': literal_eval,
                'previous courses': literal_eval,
                'course types': literal_eval,
                'course subjects': literal_eval,
                'subjects of interest': literal_eval,
                'extracurricular activities': literal_eval,
                'career aspirations': literal_eval,
                'future topics': literal_eval
            })

            for target in targets:
                logging.info(f"Starting {target}")

                target_name = target.replace(' ', '_')

                target_path = f'{output_path}/outputs/{privatization_type}/{RNN_model}/{target_name}'

                # Initiate regressor
                regressor = ISRandomForestRegression(privatization_type, RNN_model, target, data=data, output_path=target_path)
                pipeline(regressor, full_run=True)

    # Save the profiling stats to a file
    profile_stats_file = f"{output_path}/profile_stats.txt"
    with open(profile_stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats()

# List of Functions to run

data_folder = 'outputs/examples/reduced_dimensionality_data' # Add path to reduced dimensionality data

# Linear Models
get_tradeoffs_ISLinearRegression(data_folder, 'past_work/calculating_tradeoffs/regression/linear_regression')
get_tradeoffs_ISLinearRegressification(data_folder, 'past_work/calculating_tradeoffs/alternate/linear_alternate')
get_tradeoffs_ISLogisticRegression(data_folder, 'past_work/calculating_tradeoffs/classification/logistic_regression')

# Decision Trees
get_tradeoffs_ISDecisionTreeRegression(data_folder, 'past_work/calculating_tradeoffs/regression/decision_tree_regression')
get_tradeoffs_ISDecisionTreeRegressification(data_folder, 'past_work/calculating_tradeoffs/alternate/decision_tree_alternate')
get_tradeoffs_ISDecisionTreeClassification(data_folder, 'past_work/calculating_tradeoffs/classification/decision_tree_classifier')

# Random Forest
get_tradeoffs_ISRandomForestRegression(data_folder, 'past_work/calculating_tradeoffs/regression/random_forest_regression')
get_tradeoffs_ISRandomForestRegressification(data_folder, 'past_work/calculating_tradeoffs/alternate/random_forest_alternate')
get_tradeoffs_ISRandomForestClassification(data_folder, 'past_work/calculating_tradeoffs/classification/random_forest_classifier')