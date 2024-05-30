def load_config():
    return {
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(levelname)s - %(message)s"
        },
        "synthetic_data": {
            "num_samples": 1000
        },
        "privacy": {
            "mechanism": "Gaussian",
            "sensitivity": 1.0,
            "epsilon": 0.1,
            "delta": 1e-5
        }
    }
