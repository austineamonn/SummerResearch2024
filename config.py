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
                'learning_style', 'gpa', 'student semester' ,'major' ,
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

            # Style Options: laplace, uniform, randomized, shuffle
            "style": "shuffle",

            # Sensitivity Options: mean, sum
            "laplace": {
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
            # Privatize Dataset, Calculate Privacy Metrics, Run Neural Network, Test Neural Network, Simulate Data Attack
            "parts_to_run": ['Generate Dataset'],
            # ['Generate Dataset', 'Privatize Dataset', 'Calculate Privacy Metrics', 'Run Neural Network', 'Tune Neural Network', 'Test Neural Network'],
            "data path": '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_generation/Dataset.csv',
            "privatized data path": '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_privatization/Privatized_Dataset.csv',
            "cleaned data path": '/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_ml_preprocessing/Cleaned_Dataset.csv',
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

"""REU Training Session
10-11 am, June 12th, ECS 312

REU Training Session
10-11 am, July 9th, ECS 312

Online REU Workshop 
12-1 pm, July 9th
https://psu.zoom.us/j/94205819806

Online REU Workshop
12-1 pm, July 16th
https://psu.zoom.us/j/98526259710


if you don't see the paycheck by Friday morning email Dr. Wang

Put all the travel costs in one thing once you get home

send photos of all the reciepts to Dr. Qin and then Dr. Wang
to get reimbursments

"""