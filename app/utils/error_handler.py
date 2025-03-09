"""
Error handling utilities for the IPL Data Explorer app.
This module provides functions for consistent error handling across the app.
"""

import streamlit as st
import logging
import traceback
import sys
from typing import Optional, Callable, Any
from functools import wraps

# Configure logging
logger = logging.getLogger(__name__)

def display_error(message: str, exception: Optional[Exception] = None, level: str = "error") -> None:
    """
    Display an error message with consistent formatting and optional exception details.
    
    Args:
        message: User-friendly error message
        exception: Optional exception object
        level: Error level (error, warning, info)
    """
    if exception:
        logger.error(f"{message}: {str(exception)}")
        # Log the full stack trace to the console but don't show to users
        logger.debug(f"Stack trace: {traceback.format_exc()}")
    
    if level == "error":
        st.error(message)
    elif level == "warning":
        st.warning(message)
    elif level == "info":
        st.info(message)

def error_boundary(func: Callable) -> Callable:
    """
    Decorator for component functions to catch and handle exceptions gracefully.
    
    Args:
        func: Function to wrap with error handling
        
    Returns:
        Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            component_name = func.__name__
            error_message = f"Error in {component_name}: {str(e)}"
            logger.error(error_message)
            logger.debug(f"Stack trace: {traceback.format_exc()}")
            st.error(f"Something went wrong in this section. Please try refreshing the page.")
            
            # Add expandable section with technical details (for debugging)
            with st.expander("Technical Details (for developers)", expanded=False):
                st.code(f"Error in {component_name}: {str(e)}")
    
    return wrapper

class ErrorBoundary:
    """
    Context manager for error boundaries in components.
    
    Example:
        with ErrorBoundary("Team Analysis Section"):
            # Component code that might fail
    """
    
    def __init__(self, component_name: str, show_details: bool = False):
        """
        Initialize the error boundary.
        
        Args:
            component_name: Name of the component for error messages
            show_details: Whether to show technical details to the user
        """
        self.component_name = component_name
        self.show_details = show_details
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            error_message = f"Error in {self.component_name}"
            logger.error(f"{error_message}: {str(exc_val)}")
            logger.debug(f"Stack trace: {traceback.format_exc()}")
            
            st.error(f"Something went wrong in this section. Please try refreshing the page.")
            
            if self.show_details:
                with st.expander("Technical Details (for developers)", expanded=False):
                    st.code(f"{error_message}: {str(exc_val)}")
            
            # Return True to suppress the exception
            return True 