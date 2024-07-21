import logging

from src.IntelliShield.comparison import Comparison, pipeline
from src.IntelliShield.tradeoffs import (
    ISDecisionTreeClassification,
    ISLogisticRegression,
    ISRandomForestClassification,
    ISDecisionTreeRegressification,
    ISLinearRegressification,
    ISRandomForestRegressification,
    ISDecisionTreeRegression,
    ISLinearRegression,
    ISRandomForestRegression
)
from src.IntelliShield.logger import setup_logger

def test_comparison_regression(logger=None):
    model_path_dict = {
        ISDecisionTreeRegression: ['decision_tree_regression'],
        ISLinearRegression: ['linear_regression'],
        ISRandomForestRegression: ['random_forest_regression']
    }
    model_list = model_path_dict.keys()
    privatization_list = ['NoPrivatization', 'Basic_DP', 'Basic_DP_LLC', 'Uniform', 'Uniform_LLC', 'Shuffling', 'Complete_Shuffling']
    reduction_list = ['GRU1', 'LSTM1', 'Simple1']
    target_list = ['career aspirations', 'future topics']
    input_path = 'past_work/calculating_tradeoffs'
    output_path = 'past_work/comparison/regression'

    comparison = Comparison(model_list, privatization_list, reduction_list, target_list, input_path, output_path, model_path_dict, logger)

    pipeline(comparison, compare_models=True, compare_reduction=True, compare_privatization=True, compare_target=True)
    logger.info("regression comparison complete")

def test_comparison_regressification(logger=None):
    model_path_dict = {
        ISDecisionTreeRegressification: ['decision_tree_regressification', 'decision_tree_alternate'],
        ISLinearRegressification: ['linear_regressification', 'linear_alternate'],
        ISRandomForestRegressification: ['random_forest_regressification', 'random_forest_alternate']
    }
    model_list = model_path_dict.keys()
    privatization_list = ['NoPrivatization', 'Basic_DP', 'Basic_DP_LLC', 'Uniform', 'Uniform_LLC', 'Shuffling', 'Complete_Shuffling']
    reduction_list = ['GRU1', 'LSTM1', 'Simple1']
    target_list = ['future topic 1', 'future topic 2', 'future topic 3', 'future topic 4', 'future topic 5']
    input_path = 'past_work/calculating_tradeoffs'
    output_path = 'past_work/comparison/regressification'

    comparison = Comparison(model_list, privatization_list, reduction_list, target_list, input_path, output_path, model_path_dict, logger)

    pipeline(comparison, compare_models=True, compare_reduction=True, compare_privatization=True, compare_target=True)
    logger.info("regressification comparison complete")

def test_comparison_classification(logger=None):
    model_path_dict = {
        ISDecisionTreeClassification: ['decision_tree_classification', 'decision_tree_classifier'],
        ISLogisticRegression: ['logistic_regression'],
        ISRandomForestClassification: ['random_forest_classification', 'random_forest_classifier']
    }
    model_list = model_path_dict.keys()
    privatization_list = ['NoPrivatization', 'Basic_DP', 'Basic_DP_LLC', 'Uniform', 'Uniform_LLC', 'Shuffling', 'Complete_Shuffling']
    reduction_list = ['GRU1', 'LSTM1', 'Simple1']
    target_list = ['ethnoracial group', 'gender', 'international status', 'socioeconomic status']
    input_path = 'past_work/calculating_tradeoffs'
    output_path = 'past_work/comparison/classification'

    comparison = Comparison(model_list, privatization_list, reduction_list, target_list, input_path, output_path, model_path_dict, logger)

    pipeline(comparison, compare_models=True, compare_reduction=True, compare_privatization=True, compare_target=True)
    logger.info("classification comparison complete")


# Load logger
logger = setup_logger('testing_comparison_logger', 'testing_comparison.log', level=logging.INFO)

test_comparison_regression(logger)
test_comparison_regressification(logger)
test_comparison_classification(logger)