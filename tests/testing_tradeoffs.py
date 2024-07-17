"""
Testing file for the tradeoffs file
"""
import logging
import cProfile
import pstats
import src.IntelliShield.tradeoffs

def testingtradeoffs_ISLogisticRegression(output_path=None):
    # Get output path
    if output_path == None:
        output_path = '../outputs/testing_tradeoffs/logistic_regression'

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
            
            # Skip if privatization_type is 'NoPrivatization' and RNN_model is 'GRU1'
            if privatization_type == 'NoPrivatization' and RNN_model == 'GRU1':
                logging.info(f"Skipping {privatization_type} and {RNN_model}")
                continue

            for target in targets:
                logging.info(f"Starting {target}")

                target_name = target.replace(' ', '_')

                # Initiate regressor
                regressor = (privatization_type, RNN_model, target)
                regressor.run_model(get_shap=False)
                #regressor.split_data(full_model=True)
                #regressor.load_model(f'outputs/{privatization_type}/{RNN_model}/{target_name}/logistic_regressor_model.pkl')
                regressor.calculate_shap_values()
                regressor.load_shap_values(f'outputs/{privatization_type}/{RNN_model}/{target_name}/shap_values.npy')
                regressor.plot_shap_values()

    # Save the profiling stats to a file
    profile_stats_file = "profile_stats.txt"
    with open(profile_stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats()

    