"""
Reusable UI components for the IPL Data Explorer app.
This module provides functions for creating consistent UI elements across the app.
"""

import streamlit as st
from pathlib import Path
import sys
import logging
from typing import List, Dict, Any, Optional, Callable, Union, Tuple

# Add parent directory to path to allow importing from config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import APP_TITLE, APP_ICON, DATA_SOURCE_URL, DATA_SOURCE_NAME

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_css() -> None:
    """
    Load CSS styles for the application.
    """
    try:
        css_file = Path(__file__).parent.parent / "static" / "style.css"
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        
        # Add additional CSS to fix tab underline and button hover issues
        st.markdown("""
        <style>
        /* Fix for red underline in tabs */
        [data-testid="stTabs"] [role="tab"][aria-selected="true"]::before,
        [data-testid="stTabs"] [role="tab"][aria-selected="true"]::after,
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] > div::before,
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] > div::after,
        .st-emotion-cache-1inwz65,
        .st-emotion-cache-1y4pk3h,
        .st-emotion-cache-16idsys,
        .st-emotion-cache-1inwz65 div,
        .st-emotion-cache-1y4pk3h div,
        .st-emotion-cache-16idsys div {
            display: none !important;
            border: none !important;
            border-bottom: none !important;
            background: none !important;
        }
        
        /* Fix for button hover */
        button, .stButton > button, [data-testid="stTabs"] button, [data-testid="stTabs"] [role="tab"] {
            position: relative !important;
            z-index: 10 !important;
            margin-top: 5px !important;
        }
        
        button:hover, .stButton > button:hover, [data-testid="stTabs"] button:hover, [data-testid="stTabs"] [role="tab"]:hover {
            z-index: 100 !important;
            transform: translateY(-2px) !important;
        }
        
        /* Fix for the red underline in the main tabs */
        .st-emotion-cache-1inwz65, 
        .st-emotion-cache-1y4pk3h,
        .st-emotion-cache-16idsys,
        div[data-testid="stTabs"] [role="tab"][aria-selected="true"]::after,
        div[data-testid="stTabs"] [role="tab"][aria-selected="true"]::before {
            border-bottom-color: transparent !important;
            border-bottom: none !important;
            background: none !important;
            display: none !important;
        }
        </style>
        """, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Failed to load CSS: {e}")
        st.error("Failed to load styles. The application may not display correctly.")

def display_app_header() -> None:
    """
    Display the application header with title and subtitle.
    """
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="font-size: 3rem; margin-bottom: 10px; text-shadow: 0 0 10px #33ffcc, 0 0 20px #33ffcc;">
            IPL Data Explorer 🏏
        </h1>
        <p style="font-size: 1.2rem; opacity: 0.8; font-style: italic; margin-bottom: 20px;">
            Uncovering insights from Indian Premier League cricket data
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar_header() -> None:
    """
    Display the sidebar header with navigation title.
    """
    st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h3 style="margin-bottom: 10px;">Analysis Navigator 🧭</h3>
        <p style="font-size: 0.9rem; opacity: 0.8;">Explore different aspects of IPL data</p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar_footer(min_season: int, max_season: int) -> None:
    """
    Display the sidebar footer with data source information.
    
    Args:
        min_season: Minimum season year in the dataset
        max_season: Maximum season year in the dataset
    """
    season_range = f"{min_season}-{max_season}"
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div style="opacity: 0.7; font-size: 0.8rem; text-align: center;">
        <p>Made with ❤️ for cricket analytics</p>
        <p>Data from {season_range} IPL seasons</p>
        <p>Data source: <a href="{DATA_SOURCE_URL}" target="_blank">{DATA_SOURCE_NAME}</a></p>
    </div>
    """, unsafe_allow_html=True)

def display_error_message(message: str, exception: Optional[Exception] = None) -> None:
    """
    Display a user-friendly error message.
    
    Args:
        message: The error message to display
        exception: Optional exception object for logging
    """
    if exception:
        logger.error(f"{message}: {exception}")
    
    st.error(message)

def display_info_message(message: str) -> None:
    """
    Display an information message.
    
    Args:
        message: The information message to display
    """
    st.info(message)

def display_success_message(message: str) -> None:
    """
    Display a success message.
    
    Args:
        message: The success message to display
    """
    st.success(message)

def display_warning_message(message: str) -> None:
    """
    Display a warning message.
    
    Args:
        message: The warning message to display
    """
    st.warning(message)

def create_metrics_display(metrics: Dict[str, Any], columns: int = 3) -> None:
    """
    Create a metrics display with the given metrics.
    
    Args:
        metrics: Dictionary of metric names and values
        columns: Number of columns to display metrics in
    """
    # Create columns
    cols = st.columns(columns)
    
    # Distribute metrics across columns
    for i, (label, value) in enumerate(metrics.items()):
        col_idx = i % columns
        with cols[col_idx]:
            st.metric(label=label, value=value)

def create_tabs(tab_names: List[str]) -> List[st.tabs]:
    """
    Create tabs with the given names.
    
    Args:
        tab_names: List of tab names
        
    Returns:
        List of tab objects
    """
    return st.tabs(tab_names)

def create_expander(title: str, expanded: bool = False) -> st.expander:
    """
    Create an expander with the given title.
    
    Args:
        title: Expander title
        expanded: Whether the expander is expanded by default
        
    Returns:
        Expander object
    """
    return st.expander(title, expanded=expanded)

def create_download_button(data: Any, file_name: str, label: str, mime: str = "text/csv") -> None:
    """
    Create a download button for the given data.
    
    Args:
        data: Data to download
        file_name: Name of the downloaded file
        label: Button label
        mime: MIME type of the data
    """
    st.download_button(label=label, data=data, file_name=file_name, mime=mime)

def create_section_header(title: str, description: Optional[str] = None) -> None:
    """
    Create a section header with title and optional description.
    
    Args:
        title: Section title
        description: Optional section description
    """
    st.header(title)
    if description:
        st.markdown(description)

def create_subsection_header(title: str, description: Optional[str] = None) -> None:
    """
    Create a subsection header with title and optional description.
    
    Args:
        title: Subsection title
        description: Optional subsection description
    """
    st.subheader(title)
    if description:
        st.markdown(description) 