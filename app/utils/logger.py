"""
Logging utilities for the IPL Data Explorer app.
This module provides functions for configuring and using logging across the app.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import os

def setup_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> logging.Logger:
    """
    Set up a logger with the given configuration.
    
    Args:
        name: Name of the logger (defaults to root logger if None)
        level: Logging level
        log_file: Path to log file (if None, logs to console only)
        log_format: Format string for log messages
        
    Returns:
        logging.Logger: Configured logger
    """
    # Get the logger
    logger = logging.getLogger(name)
    
    # Only configure if it hasn't been configured yet
    if not logger.handlers:
        # Set the level
        logger.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(log_format)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Create file handler if log_file is specified
        if log_file:
            # Create parent directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger

def get_module_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        module_name: Name of the module
        
    Returns:
        logging.Logger: Logger for the module
    """
    return logging.getLogger(module_name)

def configure_app_logging(log_level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """
    Configure logging for the entire application.
    
    Args:
        log_level: Logging level
        log_file: Path to log file (if None, logs to console only)
    """
    # Set up the root logger
    setup_logger(None, log_level, log_file)
    
    # Set the level for external libraries to avoid verbose logging
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('streamlit').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('s3transfer').setLevel(logging.WARNING)
    
    # Get the app logger
    logger = get_module_logger('app')
    logger.debug("Logging configured successfully") 