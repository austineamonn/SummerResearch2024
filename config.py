def load_config():
    return {
        # Logging Level
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(levelname)s - %(message)s"
        },
        # How large of a dataset should be generated
        "synthetic_data": {
            "num_samples": 1000
        },
        # A variety of parameters used in the privatization methods.
        "privacy": {
            # Mechanism Options: Pufferfish, DDP, CBA, Random, Gaussian, Laplace, Exponential, Gamma, Uniform
            "mechanism": "CBA",
            "sensitivity": 1.0,
            "epsilon": 0.1,
            "delta": 1e-5,
            # Generalization levels: full, broad, slight, none
            "generalization_level": "broad",
            "noise_level": 1
        },
        "preprocessing": {
            "numerical_columns": ["gpa", "student semester", "previous courses count", "subjects diversity", "activities involvement count", "unique subjects in courses"]
        }
    }
