def load_config():
    return {
        # Logging Level
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(levelname)s - %(message)s"
        },
        # How large of a dataset should be generated
        "synthetic_data": {
            "num_samples": 100
        },
        # A variety of parameters used in the privatization methods.
        "privacy": {
            # Mechanism Options: Pufferfish, DDP, CBA, Random, Gaussian, Laplace, Exponential, Gamma, Uniform
            "mechanism": "CBA",
            "sensitivity": 1.0,
            "epsilon": 0.1,
            "delta": 1e-5,
            "noise_level": 0.1,
            "scale": 1,
            "shape": 2,
            "low": -1,
            "high": 1,
            "lam": 1,
            "salt_prob": 0.05,
            "pepper_prob": 0.05,
            "variance": 0.04,
            "flip_prob": 0.01,
            "snr": 20,
            # Generalization levels: full, broad, slight, none
            "generalization_level": "broad",
            "mutation_rate": 0.05
        },
        "preprocessing": {
            "numerical_columns": ["gpa", "student semester", "previous courses count", "subjects diversity", "activities involvement count", "unique subjects in courses"],
            "remove_columns": ["first_name", "last_name", "race_ethnicity", "gender", "international", "socioeconomic status"]
        },
        "running_model": {
            # A list of the parts of main.py that you want to run. You can add any of the following to the list: Generate Dataset,
            # Privatize Dataset, Calculate Privacy Metrics, Clean Privatized Dataset, Run Neural Network, Test Neural Network, Simulate Data Attack
            "parts_to_run": ['Run Neural Network', 'Test Neural Network'],
            # ['Generate Dataset', 'Privatize Dataset', 'Calculate Privacy Metrics', 'Clean Privatized Dataset', 'Run Neural Network', 'Test Neural Network'],
            "data path": '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/Dataset.csv',
            "privatized data path": '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/Privatized_Dataset.csv',
            "cleaned data path": '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/Cleaned_Dataset.csv'
        },
        "neural_network": {
            "optimizer": 'adam',
            "optimizer_params": {
                'learning_rate': 0.001
            },
            "loss": 'binary_crossentropy',
            "metrics": ['accuracy']
        }
    }
