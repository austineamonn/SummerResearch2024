import logging
import cProfile
import pstats
import pandas as pd
from ast import literal_eval
from src.IntelliShield.tradeoffs import ISLogisticRegression, ISDecisionTreeClassification, make_folders, get_best_model, run_model, calculate_shap_values, load_shap_values, plot_shap_values, tree_plotter, confusion_matrix_plotter,load_model, split_data

"""
Testing file for the tradeoffs file. If no output path is declared, the models will create an outputs folder for all their outputs. Tests 'NoPrivatization' ,'GRU1', and 'ethnoracial group' for the categorical models.
"""

# TODO: Add a test for each tradeoff class

def testingtradeoffs_ISLogisticRegression(output_path=None):
    # Get output path
    if output_path == None:
        output_path = 'outputs/testing_tradeoffs/logistic_regression'

    # Get the runtime values for the function
    profiler = cProfile.Profile()
    profiler.enable()

            
    # Data Path
    data_path = f'src/Intellishield/data_preprocessing/reduced_dimensionality_data/NoPrivatization/GRU1_combined.csv'

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

    target_path = f'{output_path}/outputs/NoPrivatization/GRU1/ethnoracial_group'

    # Initiate regressor
    regressor = ISLogisticRegression('NoPrivatization', 'GRU1', 'ethnoracial group', data=data, output_path=target_path)
    make_folders(regressor)
    run_model(regressor)
    #split_data(regressor, full_model=True)
    #load_model(regressor, f'{target_path}/logistic_regressor_model.pkl')
    confusion_matrix_plotter(regressor, save_fig=True)
    calculate_shap_values(regressor, save_fig=True)
    load_shap_values(regressor, f'{target_path}/shap_values.npy')
    plot_shap_values(regressor)

    # Save the profiling stats to a file
    profile_stats_file = f"{output_path}/profile_stats.txt"
    with open(profile_stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats()

def testingtradeoffs_ISDecisionTreeClassification(output_path=None):
    # Get output path
    if output_path == None:
        output_path = 'outputs/testing_tradeoffs/decision_tree_classification'

    # Get the runtime values for the function
    profiler = cProfile.Profile()
    profiler.enable()

            
    # Data Path
    data_path = f'src/Intellishield/data_preprocessing/reduced_dimensionality_data/NoPrivatization/GRU1_combined.csv'

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

    target_path = f'{output_path}/outputs/NoPrivatization/GRU1/ethnoracial_group'

    # Initiate classifier
    classifier = ISDecisionTreeClassification('NoPrivatization', 'GRU1', 'ethnoracial group', data=data, output_path=target_path)
    get_best_model(classifier)
    make_folders(classifier)
    run_model(classifier)
    #split_data(classifier, full_model=True)
    #load_model(classifier, f'{target_path}/decision_tree_classifier_model.pkl')
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

# List of Functions to test

#testingtradeoffs_ISLogisticRegression()
testingtradeoffs_ISDecisionTreeClassification()