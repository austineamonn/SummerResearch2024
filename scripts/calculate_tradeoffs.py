import logging
import cProfile
import pstats
import pandas as pd
from ast import literal_eval
from src.IntelliShield.tradeoffs import ISLogisticRegression, ISDecisionTreeClassification, make_folders, get_best_model, run_model, calculate_shap_values, load_shap_values, plot_shap_values, tree_plotter, confusion_matrix_plotter,load_model, split_data

"""
Script to get the tradeoff values from each model. Will run the model, make the graphics, calculate the feature importance, and save it all to the outputs folder. Requires a data folder that contains the reduced dimensionality data. Assumes a structure of:

"reduced_dimensionality_folder/privatization_type/reduced_dimensionality_type_combined.csv"
"""

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
                make_folders(regressor)
                run_model(regressor)
                confusion_matrix_plotter(regressor, save_fig=True)
                calculate_shap_values(regressor, save_fig=True)
                load_shap_values(regressor, f'{target_path}/shap_values.npy')
                plot_shap_values(regressor)

    # Save the profiling stats to a file
    profile_stats_file = f"{output_path}/profile_stats.txt"
    with open(profile_stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats()

def get_tradeoffs_ISDecisionTreeClassification(data_folder, output_path=None):
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

                # Initiate classifier
                classifier = ISDecisionTreeClassification(privatization_type, RNN_model, target, data=data, output_path=target_path)
                get_best_model(classifier)
                make_folders(classifier)
                run_model(classifier)
                confusion_matrix_plotter(classifier, save_fig=True)
                tree_plotter(classifier, save_fig=True)
                calculate_shap_values(classifier)
                load_shap_values(classifier, f'{target_path}/shap_values.npy')
                plot_shap_values(classifier)

    # Save the profiling stats to a file
    profile_stats_file = f"{output_path}/profile_stats.txt"
    with open(profile_stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats()

# List of Functions to run

data_folder = '' # Add path to reduced dimensionality data

get_tradeoffs_ISLogisticRegression(data_folder)
get_tradeoffs_ISDecisionTreeClassification(data_folder)