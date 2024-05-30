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
            "mechanism": "DDP",
            "basic_mechanism": "Gaussian",
            "sensitivity": 1.0,
            "epsilon": 0.1,
            "delta": 1e-5,
            "secrets": ["gpa", "class year"],
            "discriminative_pairs": [("Male", "Female"), ("Domestic", "International")],
            "ddp": {
                "correlation_coefficient": 0.5
            }
        }
    }
