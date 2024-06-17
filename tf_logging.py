#import tensorflow as tf
import logging

# Custom filter to suppress specific TensorFlow log messages
class TensorFlowFilter(logging.Filter):
    def filter(self, record):
        suppressed_messages = [
            "Created TensorFlow device",
            "Could not identify NUMA node"
        ]
        return not any(msg in record.getMessage() for msg in suppressed_messages)
"""
# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Add the filter to TensorFlow logger
tf_logger = tf.get_logger()
tf_logger.setLevel(logging.INFO)
tf_logger.addFilter(TensorFlowFilter())

# Example of logging at different levels
logger.info("This is an INFO level log message")
logger.debug("This is a DEBUG level log message")

# TensorFlow specific logging to demonstrate suppression
tf_logger.info("This should appear")
tf_logger.info("Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory)")
tf_logger.info("Could not identify NUMA node of platform GPU ID 0, defaulting to 0")"""