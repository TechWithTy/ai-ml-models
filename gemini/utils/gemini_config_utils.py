"""
Configuration utilities for the Gemini SDK, including logging.
"""
import logging

def get_gemini_logger(name: str) -> logging.Logger:
    """! Returns a configured logger instance."""
    logger = logging.getLogger(name)
    # Basic configuration - can be expanded as needed
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
