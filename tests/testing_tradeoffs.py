import logging
import cProfile
import pstats
import pandas as pd
from ast import literal_eval
from src.IntelliShield.tradeoffs import ISLogisticRegression, ISDecisionTreeClassification, ISDecisionTreeRegressification, ISDecisionTreeRegression, ISLinearRegression, ISRandomForestRegression, ISRandomForestClassification, ISLinearRegressification, ISRandomForestRegressification, pipeline

"""
Testing file for the tradeoffs file. If no output path is declared, the models will create an outputs folder for all their outputs. These models rely on the example reduced dimensionality data.

Tests the combination of 'NoPrivatization' and 'GRU1'.

'ethnoracial group' is the target for the categorical models.
'future topic 1' is the target for regressification models.
'future topics' is the target for regression models.
"""

# TODO: Add a test for each tradeoff class

# Classification Models

def testingtradeoffs_ISDecisionTreeClassification(output_path=None):
    # Get output path
    if output_path == None:
        output_path = 'outputs/testing_tradeoffs/decision_tree_classification'

    # Get the runtime values for the function
    profiler = cProfile.Profile()
    profiler.enable()
            
    # Data Path
    data_path = f'outputs/examples/reduced_dimensionality_data/NoPrivatization/GRU1_combined.csv'

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
    pipeline(classifier, full_run=True)

    # Save the profiling stats to a file
    profile_stats_file = f"{output_path}/profile_stats.txt"
    with open(profile_stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats()

def testingtradeoffs_ISLogisticRegression(output_path=None):
    # Get output path
    if output_path == None:
        output_path = 'outputs/testing_tradeoffs/logistic_regression'

    # Get the runtime values for the function
    profiler = cProfile.Profile()
    profiler.enable()
            
    # Data Path
    data_path = f'outputs/examples/reduced_dimensionality_data/NoPrivatization/GRU1_combined.csv'

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
    pipeline(regressor, full_run=True)

    # Save the profiling stats to a file
    profile_stats_file = f"{output_path}/profile_stats.txt"
    with open(profile_stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats()

def testingtradeoffs_ISRandomForestClassification(output_path=None):
    # Get output path
    if output_path == None:
        output_path = 'outputs/testing_tradeoffs/random_forest_classification'

    # Get the runtime values for the function
    profiler = cProfile.Profile()
    profiler.enable()
            
    # Data Path
    data_path = f'outputs/examples/reduced_dimensionality_data/NoPrivatization/GRU1_combined.csv'

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
    classifier = ISRandomForestClassification('NoPrivatization', 'GRU1', 'ethnoracial group', data=data, output_path=target_path)
    pipeline(classifier, full_run=True)

    # Save the profiling stats to a file
    profile_stats_file = f"{output_path}/profile_stats.txt"
    with open(profile_stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats()

# Regressification Models

def testingtradeoffs_ISDecisionTreeRegressification(output_path=None):
    # Get output path
    if output_path == None:
        output_path = 'outputs/testing_tradeoffs/decision_tree_regressification'

    # Get the runtime values for the function
    profiler = cProfile.Profile()
    profiler.enable()
     
    # Data Path
    data_path = f'outputs/examples/reduced_dimensionality_data/NoPrivatization/GRU1_alt_future_topics.csv'

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

    target_path = f'{output_path}/outputs/NoPrivatization/GRU1/future_topic_1'

    # Initiate regressifier
    regressifier = ISDecisionTreeRegressification('NoPrivatization', 'GRU1', 'future topic 1', data=data, output_path=target_path)
    pipeline(regressifier, full_run=True)

    # Save the profiling stats to a file
    profile_stats_file = f"{output_path}/profile_stats.txt"
    with open(profile_stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats()

def testingtradeoffs_ISLinearRegressification(output_path=None):
    # Get output path
    if output_path == None:
        output_path = 'outputs/testing_tradeoffs/linear_regressification'

    # Get the runtime values for the function
    profiler = cProfile.Profile()
    profiler.enable()
     
    # Data Path
    data_path = f'outputs/examples/reduced_dimensionality_data/NoPrivatization/GRU1_alt_future_topics.csv'

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

    target_path = f'{output_path}/outputs/NoPrivatization/GRU1/future_topic_1'

    # Initiate regressifier
    regressifier = ISLinearRegressification('NoPrivatization', 'GRU1', 'future topic 1', data=data, output_path=target_path)
    pipeline(regressifier, full_run=True)

    # Save the profiling stats to a file
    profile_stats_file = f"{output_path}/profile_stats.txt"
    with open(profile_stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats()

def testingtradeoffs_ISRandomForestRegressification(output_path=None):
    # Get output path
    if output_path == None:
        output_path = 'outputs/testing_tradeoffs/random_forest_regressification'

    # Get the runtime values for the function
    profiler = cProfile.Profile()
    profiler.enable()
     
    # Data Path
    data_path = f'outputs/examples/reduced_dimensionality_data/NoPrivatization/GRU1_alt_future_topics.csv'

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

    target_path = f'{output_path}/outputs/NoPrivatization/GRU1/future_topic_1'

    # Initiate regressifier
    regressifier = ISRandomForestRegressification('NoPrivatization', 'GRU1', 'future topic 1', data=data, output_path=target_path)
    pipeline(regressifier, full_run=True)

    # Save the profiling stats to a file
    profile_stats_file = f"{output_path}/profile_stats.txt"
    with open(profile_stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats()

# Regression Models

def testingtradeoffs_ISDecisionTreeRegression(output_path=None):
    # Get output path
    if output_path == None:
        output_path = 'outputs/testing_tradeoffs/decision_tree_regression'

    # Get the runtime values for the function
    profiler = cProfile.Profile()
    profiler.enable()
            
    # Data Path
    data_path = f'outputs/examples/reduced_dimensionality_data/NoPrivatization/GRU1_combined.csv'

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

    target_path = f'{output_path}/outputs/NoPrivatization/GRU1/future_topics'

    # Initiate regressor
    regressor = ISDecisionTreeRegression('NoPrivatization', 'GRU1', 'future topics', data=data, output_path=target_path)
    pipeline(regressor, full_run=True)

    # Save the profiling stats to a file
    profile_stats_file = f"{output_path}/profile_stats.txt"
    with open(profile_stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats()

def testingtradeoffs_ISLinearRegression(output_path=None):
    # Get output path
    if output_path == None:
        output_path = 'outputs/testing_tradeoffs/linear_regression'

    # Get the runtime values for the function
    profiler = cProfile.Profile()
    profiler.enable()
            
    # Data Path
    data_path = f'outputs/examples/reduced_dimensionality_data/NoPrivatization/GRU1_combined.csv'

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

    target_path = f'{output_path}/outputs/NoPrivatization/GRU1/future_topics'

    # Initiate regressor
    regressor = ISLinearRegression('NoPrivatization', 'GRU1', 'future topics', data=data, output_path=target_path)
    pipeline(regressor, full_run=True)

    # Save the profiling stats to a file
    profile_stats_file = f"{output_path}/profile_stats.txt"
    with open(profile_stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats()

def testingtradeoffs_ISRandomForestRegression(output_path=None):
    # Get output path
    if output_path == None:
        output_path = 'outputs/testing_tradeoffs/random_forest_regression'

    # Get the runtime values for the function
    profiler = cProfile.Profile()
    profiler.enable()
            
    # Data Path
    data_path = f'outputs/examples/reduced_dimensionality_data/NoPrivatization/GRU1_combined.csv'

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

    target_path = f'{output_path}/outputs/NoPrivatization/GRU1/future_topics'

    # Initiate regressor
    regressor = ISRandomForestRegression('NoPrivatization', 'GRU1', 'future topics', data=data, output_path=target_path)
    pipeline(regressor, full_run=True)

    # Save the profiling stats to a file
    profile_stats_file = f"{output_path}/profile_stats.txt"
    with open(profile_stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats()

# List of Functions to test

#testingtradeoffs_ISLogisticRegression()
#testingtradeoffs_ISDecisionTreeClassification()
#testingtradeoffs_ISDecisionTreeRegressification()
#testingtradeoffs_ISDecisionTreeRegression()
#testingtradeoffs_ISLinearRegression()
#testingtradeoffs_ISRandomForestRegression()
testingtradeoffs_ISRandomForestClassification()
#testingtradeoffs_ISLinearRegressification()
#testingtradeoffs_ISRandomForestRegressification()