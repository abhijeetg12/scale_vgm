# src/utils/logging_utils.py
import logging
from google.cloud import logging as cloud_logging

def setup_logger(name):
    """
    Set up logger with both console and Cloud Logging handlers.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Console handler
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
        # Cloud Logging handler
        try:
            client = cloud_logging.Client()
            cloud_handler = cloud_logging.handlers.CloudLoggingHandler(client)
            cloud_handler.setFormatter(formatter)
            logger.addHandler(cloud_handler)
        except Exception as e:
            logger.warning(f"Could not set up Cloud Logging: {str(e)}")
    
    return logger