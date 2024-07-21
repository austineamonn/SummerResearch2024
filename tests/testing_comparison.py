import logging

from src.IntelliShield.comparison import Comparison, make_comparison
from src.IntelliShield.tradeoffs import ISDecisionTreeClassification, ISDecisionTreeRegressification, ISDecisionTreeRegression, ISLogisticRegression, ISRandomForestClassification, ISLinearRegressification, ISLinearRegression, ISRandomForestRegressification, ISRandomForestRegression
from src.IntelliShield.logger import setup_logger

def test_comparison():
    # Load logger
    logger = setup_logger('testing_comparison_logger', 'testing_comparison.log', level=logging.INFO)

    """model_path_dict = {
        ISDecisionTreeClassification: ['decision_tree_classification', 'decision_tree_classifier'],
        ISDecisionTreeRegressification: ['decision_tree_regressification', 'decision_tree_alternate'],
        ISDecisionTreeRegression: ['decision_tree_regression'],
        ISLogisticRegression: ['logistic_regression'],
        ISLinearRegressification: ['linear_regressification', 'linear_alternate'],
        ISLinearRegression: ['linear_regression'],
        ISRandomForestClassification: ['random_forest_classification', 'random_forest_classifier'],
        ISRandomForestRegressification: ['random_forest_regressification', 'random_forest_alternate'],
        ISRandomForestRegression: ['random_forest_regression']
    }"""
    model_path_dict = {
        ISDecisionTreeRegression: ['decision_tree_regression'],
        ISLinearRegression: ['linear_regression'],
        ISRandomForestRegression: ['random_forest_regression']
    }
    model_list = model_path_dict.keys()
    print(model_list)
    privatization_list = ['NoPrivatization', 'Basic_DP', 'Basic_DP_LLC', 'Uniform', 'Uniform_LLC', 'Shuffling', 'Complete_Shuffling']
    reduction_list = ['GRU1', 'LSTM1', 'Simple1']
    #target_list = ['ethnoracial group', 'gender', 'international status', 'socioeconomic status']
    target_list = ['career aspirations', 'future topics']
    input_path = 'past_work/calculating_tradeoffs'
    output_path = 'past_work/comparison'

    comparison = Comparison(model_list, privatization_list, reduction_list, target_list, input_path, output_path, model_path_dict, logger)

    order = [comparison.privatization_list, comparison.reduction_list, comparison.target_list, comparison.model_list]
    make_comparison(comparison, order)

test_comparison()