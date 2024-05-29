import json

def load_config(config_file="config.json"):
    """
    Utility function for loading configurations
    """

    with open(config_file, "r") as file:
        config = json.load(file)
    return config
