"""
State management utilities for the IPL Data Explorer app.
This module provides functions for managing session state across app components.
"""

import streamlit as st
from typing import Any, Dict, Optional, List
import logging

# Configure logging
logger = logging.getLogger(__name__)

def initialize_session_state(keys_with_defaults: Dict[str, Any]) -> None:
    """
    Initialize session state variables if they don't exist.
    
    Args:
        keys_with_defaults: Dictionary of state keys and their default values
    """
    for key, default_value in keys_with_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
            logger.debug(f"Initialized session state key '{key}' with default value: {default_value}")

def get_state(key: str, default: Any = None) -> Any:
    """
    Get a value from session state with a fallback default.
    
    Args:
        key: The session state key
        default: Default value if key doesn't exist
        
    Returns:
        The value from session state or the default
    """
    return st.session_state.get(key, default)

def set_state(key: str, value: Any) -> None:
    """
    Set a value in session state.
    
    Args:
        key: The session state key
        value: The value to set
    """
    st.session_state[key] = value
    logger.debug(f"Set session state key '{key}' to value: {value}")

def update_states(updates: Dict[str, Any]) -> None:
    """
    Update multiple session state values at once.
    
    Args:
        updates: Dictionary of keys and values to update
    """
    for key, value in updates.items():
        st.session_state[key] = value
    logger.debug(f"Updated {len(updates)} session state values")

def delete_state(key: str) -> None:
    """
    Delete a key from session state if it exists.
    
    Args:
        key: The session state key to delete
    """
    if key in st.session_state:
        del st.session_state[key]
        logger.debug(f"Deleted session state key '{key}'")

def reset_filters() -> None:
    """
    Reset all filter-related session state variables.
    """
    filter_keys = [k for k in st.session_state.keys() if k.startswith('filter_')]
    for key in filter_keys:
        del st.session_state[key]
    logger.debug(f"Reset {len(filter_keys)} filter-related session state variables")

def is_mobile() -> bool:
    """
    Check if the current device is a mobile device.
    
    Returns:
        bool: True if mobile device, False otherwise
    """
    return get_state('device_type', 'desktop') == 'mobile' 