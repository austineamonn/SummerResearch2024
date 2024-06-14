def load_config():
    return {
        # Logging Level
        "logging": {
            "level": "DEBUG",
            "format": "%(asctime)s - %(levelname)s - %(message)s"
        },
        # How large of a dataset should be generated
        "synthetic_data": {
            "num_samples": 10,

            # "real" uses the real statistical distributions
            # "uniform" uses uniform distributions
            "ethnoracial_group": "real",
            "gender": "real",
            "international_status": "real",
            "socioeconomic_status": "real",
            "learning_style": "real"
        },
        "preprocessing": {
            "n_components": 100
        },
        # A variety of parameters used in the privatization methods.
        "privacy": {
            # Data Columns: first name,last name,ethnoracial group,gender,
            # international status,socioeconomic status,learning_style,gpa,
            # student semester,major,previous courses,course types,course subjects,
            # subjects of interest,career aspirations,extracurricular activities,
            # future topics
            # Split them into Xp, X and Xu
            "Xp_list": [
                'first name','last name','ethnoracial group','gender',
                'international status','socioeconomic status'
            ],
            "X_list": [
                'learning style', 'gpa', 'student semester' ,'major' ,
                'previous courses','course types','course subjects',
                'subjects of interest', 'extracurricular activities'
            ],
            "Xu_list": [
                'career aspirations', 'future topics'
            ],
            "numerical_columns": [
                "gpa", "student semester"
            ],

            # Normalization Parameters
            "normalize_type": 'Zscore',

            # Style Options: basic differential privacy, uniform, randomized, shuffle
            "style": "shuffle",

            # Sensitivity Options: mean, sum
            "basic differential privacy": {
                "sensitivity": 'mean',
                "epsilon": 0.1
            },
            "uniform": {
                "low": -1,
                "high": 1
            },
            "randomized": {
                "p": 0.1
            },
            "shuffle": {
                "shuffle_ratio": 0.1
            }
        },
        "running_model": {
            # A list of the parts of main.py that you want to run. You can add any of the following to the list: Generate Dataset,
            # Preprocess Dataset, Privatize Dataset, Calculate Privacy Metrics, Run Neural Network, Test Neural Network, Simulate Data Attack
            "parts_to_run": ['Generate Dataset'],
            # ['Generate Dataset', 'Preprocess Dataset', 'Privatize Dataset', 'Calculate Privacy Metrics', 'Run Neural Network', 'Tune Neural Network', 'Test Neural Network'],
            "analyze_PCA": True,
            "data path": '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_generation/Dataset.csv',
            "privatized data path": '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_privatization/Privatized_Dataset.csv',
            "statistics comparison path": '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_privatization/Stats_Comparison_Dataset.csv',
            "preprocessed data path": '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_preprocessing/Preprocessed_Dataset.csv',
            "PCA explained variance path": '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_preprocessing/explained_variance_plot.png',
            "directory": '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024'
        },
        "neural_network": {
            "optimizer": 'adam',
            "optimizer_params": {
                'learning_rate': 0.001
            },
            "loss": 'binary_crossentropy',
            "metrics": ['accuracy'],
            "epochs": 50,
            "batch_size": 32
        }
    }