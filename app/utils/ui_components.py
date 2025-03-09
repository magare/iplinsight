"""
Reusable UI components for the IPL Data Explorer app.
This module provides functions for creating consistent UI elements across the app.
"""

import streamlit as st
from pathlib import Path
import sys
import logging
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
import pandas as pd
import plotly.graph_objects as go

# Add parent directory to path to allow importing from config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import APP_TITLE, APP_ICON, DATA_SOURCE_URL, DATA_SOURCE_NAME
from utils.state_manager import is_mobile

# Configure logging
logger = logging.getLogger(__name__)

# UI Component Classes
class UIComponent:
    """Base class for UI components with error handling."""
    
    @staticmethod
    def render_safely(component_func: Callable, *args, **kwargs):
        """
        Render a component with error handling.
        
        Args:
            component_func: Function to render the component
            *args, **kwargs: Arguments to pass to the component function
        """
        try:
            return component_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error rendering component {component_func.__name__}: {e}")
            st.error("Failed to render this component. Please try refreshing the page.")
            with st.expander("Technical Details", expanded=False):
                st.write(f"Error in {component_func.__name__}: {str(e)}")
            return None

class HeaderComponent:
    """Component for application headers."""
    
    @staticmethod
    def main_header():
        """Display the application header with title and subtitle."""
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
    
    @staticmethod
    def section_header(title: str, description: Optional[str] = None):
        """
        Create a section header with title and optional description.
        
        Args:
            title: Section title
            description: Optional section description
        """
        st.header(title)
        if description:
            st.markdown(description)
    
    @staticmethod
    def subsection_header(title: str, description: Optional[str] = None):
        """
        Create a subsection header with title and optional description.
        
        Args:
            title: Subsection title
            description: Optional subsection description
        """
        st.subheader(title)
        if description:
            st.markdown(description)

class SidebarComponent:
    """Component for sidebar elements."""
    
    @staticmethod
    def sidebar_header():
        """Display the sidebar header with navigation title."""
        st.sidebar.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h3 style="margin-bottom: 10px;">Analysis Navigator 🧭</h3>
            <p style="font-size: 0.9rem; opacity: 0.8;">Explore different aspects of IPL data</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def sidebar_footer(min_season: int, max_season: int):
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
    
    @staticmethod
    def filter_section(title: str, filters_dict: Dict[str, Callable], reset_button: bool = True):
        """
        Create a sidebar filter section with multiple filters.
        
        Args:
            title: Filter section title
            filters_dict: Dictionary of filter labels and filter functions
            reset_button: Whether to show a reset button
        """
        with st.sidebar.expander(title, expanded=True):
            for label, filter_func in filters_dict.items():
                filter_func()
                
            if reset_button:
                if st.button("Reset Filters", key=f"reset_{title.lower().replace(' ', '_')}"):
                    # Reset the filters in session state
                    for key in st.session_state.keys():
                        if key.startswith("filter_"):
                            del st.session_state[key]
                    st.experimental_rerun()

class MessageComponent:
    """Component for different types of messages."""
    
    @staticmethod
    def error(message: str, exception: Optional[Exception] = None):
        """
        Display a user-friendly error message.
        
        Args:
            message: The error message to display
            exception: Optional exception object for logging
        """
        if exception:
            logger.error(f"{message}: {exception}")
        
        st.error(message)
    
    @staticmethod
    def info(message: str):
        """
        Display an information message.
        
        Args:
            message: The information message to display
        """
        st.info(message)
    
    @staticmethod
    def success(message: str):
        """
        Display a success message.
        
        Args:
            message: The success message to display
        """
        st.success(message)
    
    @staticmethod
    def warning(message: str):
        """
        Display a warning message.
        
        Args:
            message: The warning message to display
        """
        st.warning(message)

class DataDisplayComponent:
    """Component for displaying data in various formats."""
    
    @staticmethod
    def metrics_display(metrics: Dict[str, Any], columns: int = 3):
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
    
    @staticmethod
    def table_with_download(data: pd.DataFrame, title: Optional[str] = None, 
                           download_label: str = "Download CSV", 
                           filename: str = "data.csv"):
        """
        Display a table with a download button.
        
        Args:
            data: Dataframe to display
            title: Optional title for the table
            download_label: Label for the download button
            filename: Filename for the downloaded file
        """
        if title:
            st.subheader(title)
            
        st.dataframe(data)
        
        # Add download button
        csv = data.to_csv(index=False)
        st.download_button(
            label=download_label,
            data=csv,
            file_name=filename,
            mime="text/csv"
        )
    
    @staticmethod
    def conditional_display(data: pd.DataFrame, condition: Callable[[pd.DataFrame], bool], 
                           true_message: str, component_func: Callable, *args, **kwargs):
        """
        Conditionally display a component based on a condition.
        
        Args:
            data: Dataframe to check
            condition: Function that returns True/False based on the data
            true_message: Message to display if condition is True
            component_func: Function to call if condition is False
            *args, **kwargs: Arguments to pass to component_func
        """
        if condition(data):
            st.info(true_message)
        else:
            component_func(*args, **kwargs)

class LayoutComponent:
    """Component for layout elements."""
    
    @staticmethod
    def tabs(tab_names: List[str]) -> List:
        """
        Create tabs with the given names.
        
        Args:
            tab_names: List of tab names
            
        Returns:
            List of tab objects
        """
        return st.tabs(tab_names)
    
    @staticmethod
    def expander(title: str, expanded: bool = False):
        """
        Create an expander with the given title.
        
        Args:
            title: Expander title
            expanded: Whether the expander is expanded by default
            
        Returns:
            Expander object
        """
        return st.expander(title, expanded=expanded)
    
    @staticmethod
    def responsive_columns(num_desktop: int = 3, num_mobile: int = 1) -> List[Any]:
        """
        Create responsive columns based on device type.
        
        Args:
            num_desktop: Number of columns on desktop
            num_mobile: Number of columns on mobile
            
        Returns:
            List of column objects
        """
        device_type = st.session_state.get('device_type', 'desktop')
        num_cols = num_mobile if device_type == 'mobile' else num_desktop
        return st.columns(num_cols)

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

# Legacy function aliases for backward compatibility
display_app_header = HeaderComponent.main_header
display_sidebar_header = SidebarComponent.sidebar_header
display_sidebar_footer = SidebarComponent.sidebar_footer
display_error_message = MessageComponent.error
display_info_message = MessageComponent.info
display_success_message = MessageComponent.success
display_warning_message = MessageComponent.warning
create_metrics_display = DataDisplayComponent.metrics_display
create_tabs = LayoutComponent.tabs
create_expander = LayoutComponent.expander
create_section_header = HeaderComponent.section_header
create_subsection_header = HeaderComponent.subsection_header 