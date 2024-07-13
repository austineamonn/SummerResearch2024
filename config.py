def load_config():
    return {
        # Logging Level
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "handlers": ["console"]
        },
        "running_model": {
            # A list of the parts of main.py that you want to run. You can add any of the following to the list: Generate Dataset,
            # Analyze Dataset, Preprocess Dataset, Create RNNs, Privatize Dataset, Calculate Privacy Metrics, Run Neural Network, Test Neural Network
            "parts_to_run": ['Create RNNs'],
            "processing_unit": 'CPU',
            "analyze_PCA": True,
            # Add your data file paths here:
            #"data path": '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_generation/Dataset.csv',

            "data_generation_paths": {
                # Data Generation Paths
                "data path": '/Users/austinnicolas/Documents/SummerREU2024/saved_research_files/Dataset.csv',

                "data path 2": '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_generation/Dataset2.csv',
            },

            "privatized_data_paths": {
                "basic differential privacy privatized data path": '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_privatization/privatized_datasets/Basic_DP_Privatized_Dataset.csv',

                "basic differential privacy LLC privatized data path": '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_privatization/privatized_datasets/Basic_DP_LLC_Privatized_Dataset.csv',

                "uniform noise privatized data path": '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_privatization/privatized_datasets/Uniform_Privatized_Dataset.csv',

                "uniform noise LLC privatized data path": '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_privatization/privatized_datasets/Uniform_LLC_Privatized_Dataset.csv',

                "random shuffling privatized data path": '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_privatization/privatized_datasets/Shuffling_Privatized_Dataset.csv',

                "complete shuffling privatized data path": '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_privatization/privatized_datasets/Complete_Shuffling_Privatized_Dataset.csv',
            },

            "preprocessed_data_paths": {
                # Preprocessed Data Paths

                "private columns path": '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/calculating_tradeoffs/Private_Columns.csv',

                "preprocessed data path": '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_preprocessing/Preprocessed_Dataset.csv',
            },

            "completely_preprocessed_data_paths": {
                "NoPrivatization": {
                    # GRU1
                    "GRU1": '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_preprocessing/RNN_models/GRU1.csv',

                    "GRU1_combined": '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_preprocessing/RNN_models/GRU1_combined.csv',

                    # LSTM1
                    "LSTM1": '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_preprocessing/RNN_models/LSTM1.csv',

                    "LSTM1_combined": '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_preprocessing/RNN_models/LSTM1_combined.csv',

                    # Simple1
                    "Simple1": '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_preprocessing/RNN_models/Simple1.csv',

                    "Simple1_combined": '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_preprocessing/RNN_models/Simple1_combined.csv',
                }
            },

            "directory": '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024'
        },
        # How large of a dataset should be generated
        "synthetic_data": {
            "num_samples": 10,
            "batch_size": 10,
            "rewrite": True,

            # "real" uses the real statistical distributions
            # "uniform" uses uniform distributions
            "ethnoracial_group": "real",
            "gender": "real",
            "international_status": "real",
            "socioeconomic_status": "real",
            "learning_style": "real"
        },
        "data_analysis":{
            "output_dir": 'data_generation/data_analysis_graphs',
            "string_cols": [
                'first name', 'last name', 'ethnoracial group', 'gender', 
                'international status', 'socioeconomic status'
            ],
            "list_cols": [
                'learning style', 'major', 'previous courses', 'course types', 
                'course subjects', 'subjects of interest', 'extracurricular activities', 
                'career aspirations', 'future topics'
            ]
        },
        "preprocessing": {
            "n_components": 100
        },
        # A variety of parameters used in the privatization methods.
        "privacy": {
            # Data Columns: first name,last name,ethnoracial group,gender,
            # international status,socioeconomic status,learning_style,gpa,
            # student semester,major,previous courses,course types,course subjects,
            # subjects of interest, extracurricular activities,
            # career aspirations, future topics
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

            # Style Options: basic differential privacy, uniform, shuffle
            "style": "shuffle",

            "basic differential privacy": {
                # method choices are 'laplace' and 'uniform'
                "method": 'laplace',
                "epsilon": 0.1
            },
            "shuffle": {
                "shuffle_ratio": 0.1
            },

            # Boolean on whether to increase list lengths or not
            "list_len": False
        },
        "calculating_tradeoffs":{
            "privacy_cols": [
                'ethnoracial group', 'gender', 'international status'
            ]
        }
    }