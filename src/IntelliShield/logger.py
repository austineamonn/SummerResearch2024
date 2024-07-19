import logging
import os

def find_project_root(marker_files=('setup.py', 'README.md'), project_root=None):
    """
    Function to find the project root. It can take a project root as an input, look for marker files, or use the environment variable.
    """
    # Check if PROJECT_ROOT environment variable is set
    if project_root is None:
        project_root = os.getenv('PROJECT_ROOT')
        if project_root and os.path.exists(project_root):
            return project_root
    else:
        return project_root

    # If not set, determine the root based on marker files
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while current_dir != os.path.abspath(os.sep):
        for marker in marker_files:
            if os.path.exists(os.path.join(current_dir, marker)):
                return current_dir
        current_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    return os.path.abspath(os.path.dirname(__file__))  # Default to the script's directory

def setup_logger(name, log_file, level=logging.INFO):
    """Function to set up a logger; logs to the specified file in the logs directory."""
    # Ensure the logs directory exists
    project_root = find_project_root()
    logs_dir = os.path.join(project_root, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Construct the full log file path
    log_file_path = os.path.join(logs_dir, log_file)

    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create handlers
    f_handler = logging.FileHandler(log_file_path)
    f_handler.setLevel(level)

    # Create formatters and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(formatter)

    # Add handlers to the logger
    if not logger.hasHandlers():
        logger.addHandler(f_handler)

    return logger