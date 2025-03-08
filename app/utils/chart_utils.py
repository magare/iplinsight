"""
Utility functions for creating and styling charts in the IPL Data Explorer app.
This module provides functions for responsive chart layouts and device detection.
"""

import streamlit as st
import json
import logging
from typing import Dict, Any, Optional, Union, List
import sys
from pathlib import Path
import plotly.graph_objects as go

# Add parent directory to path to allow importing from config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CHART_HEIGHT, CHART_HEIGHT_MOBILE, CHART_MARGIN_DESKTOP, CHART_MARGIN_MOBILE
from .color_palette import NEON_COLORS, CHART_STYLING, apply_neon_style

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_device_type() -> str:
    """
    Check if the user is on a mobile device based on session state.
    
    Returns:
        str: 'mobile' or 'desktop'
    """
    # Use the session state to determine device type
    # Default to desktop if not set
    return st.session_state.get('device_type', 'desktop')

def ensure_all_ticks_shown(fig: go.Figure) -> go.Figure:
    """
    Updates a figure to ensure that all y-axis ticks are shown when possible.
    This improves readability of charts by displaying more detailed y-axis labels.
    
    Args:
        fig: Plotly figure object
        
    Returns:
        The updated Plotly figure object
    """
    try:
        # Configure y-axis to show all ticks using 'linear' tick mode
        # This ensures all values are shown rather than a subset
        fig.update_yaxes(
            tickmode='auto',
            nticks=20,  # Increased number of ticks
            automargin=True  # Ensure there's enough margin for labels
        )
    except Exception as e:
        logger.warning(f"Error ensuring all ticks shown: {e}")
    
    return fig

def update_chart_for_responsive_layout(fig: go.Figure, device_type: Optional[str] = None) -> go.Figure:
    """
    Update chart layout to be responsive to device type.
    For mobile devices, move the legend to the top to save horizontal space.
    
    Args:
        fig: Plotly figure object
        device_type: Optional device type ('mobile' or 'desktop')
                     If not provided, it will be detected
    
    Returns:
        The updated Plotly figure object
    """
    if device_type is None:
        device_type = get_device_type()
    
    try:
        # Update layout based on device type
        if device_type == 'mobile':
            # Place legend at the top of the chart on mobile with more space
            fig.update_layout(
                legend=dict(
                    orientation="h",  # horizontal orientation
                    yanchor="bottom",
                    y=1.15,           # position further above the chart to avoid overlap
                    xanchor="center",
                    x=0.5,            # centered horizontally
                    font=CHART_STYLING['legend_font'],
                    itemsizing='constant',  # Make legend items consistent size
                    itemwidth=30,           # Limit width of legend items
                    entrywidth=70           # Limit width of legend entries
                ),
                margin=CHART_MARGIN_MOBILE,  # Add more top margin to accommodate the legend
                height=CHART_HEIGHT_MOBILE,  # Slightly reduce height for better mobile viewing
                title=dict(
                    y=0.95,           # Move title down slightly to avoid overlap with legend
                    x=0.5,
                    xanchor='center',
                    yanchor='top',
                    font=dict(size=14)
                )
            )
            
            # Make axis titles and text smaller on mobile
            fig.update_layout(
                xaxis=dict(
                    title_font=dict(size=10),
                    tickfont=dict(size=8)
                ),
                yaxis=dict(
                    title_font=dict(size=10),
                    tickfont=dict(size=8)
                )
            )
            
            # Reduce tick density on mobile
            if hasattr(fig, 'data') and len(fig.data) > 0:
                if hasattr(fig.data[0], 'x') and fig.data[0].x is not None and len(fig.data[0].x) > 8:
                    fig.update_xaxes(nticks=8)  # Limit number of ticks on x-axis
        else:
            # Default legend position for desktop (usually to the right)
            fig.update_layout(
                legend=dict(
                    orientation="v",  # vertical orientation
                    yanchor="middle",
                    y=0.5,            # middle of the chart
                    xanchor="right",
                    x=1.02,           # just to the right of the chart
                    font=CHART_STYLING['legend_font']
                ),
                margin=CHART_MARGIN_DESKTOP,  # Standard margins for desktop
                height=CHART_HEIGHT
            )
            
            # Ensure all y-axis ticks are shown when possible on desktop
            fig = ensure_all_ticks_shown(fig)
        
        # Apply neon styling to the chart
        fig = apply_neon_style(fig)
    except Exception as e:
        logger.error(f"Error updating chart for responsive layout: {e}")
    
    return fig

def responsive_plotly_chart(fig: go.Figure, use_container_width: bool = True, **kwargs):
    """
    Render a responsive Plotly chart with appropriate layout for the current device.
    
    Args:
        fig: Plotly figure object
        use_container_width: Whether to use the container width
        **kwargs: Additional arguments to pass to st.plotly_chart
    """
    try:
        # Detect device type
        device_type = get_device_type()
        
        # Update chart layout for responsive design
        fig = update_chart_for_responsive_layout(fig, device_type)
        
        # Render the chart
        st.plotly_chart(fig, use_container_width=use_container_width, **kwargs)
    except Exception as e:
        logger.error(f"Error rendering responsive plotly chart: {e}")
        st.error("Failed to render chart. Please try refreshing the page.")

def get_neon_color_discrete_sequence(n: Optional[int] = None) -> List[str]:
    """
    Get a list of neon colors for discrete sequences in charts.
    
    Args:
        n: Optional number of colors to return. If None, returns all colors.
        
    Returns:
        List of neon color hex codes
    """
    if n is None or n > len(NEON_COLORS):
        return NEON_COLORS
    return NEON_COLORS[:n]

def init_device_detection() -> str:
    """
    Initialize device detection in session state when the app loads.
    Always set to mobile to ensure responsive layout for all users.
    
    Returns:
        str: The detected device type ('mobile' or 'desktop')
    """
    try:
        # Always set to mobile for better responsive layout
        st.session_state.device_type = 'mobile'
        
        # Add a simple CSS-based device detection that just logs the device type
        device_detector_css = """
        <style>
        /* This is just a placeholder to keep the HTML component */
        .device-detector { 
            display: none;
        }
        </style>
        <div class="device-detector"></div>
        """
        
        # Just render the CSS without expecting any callbacks
        st.components.v1.html(device_detector_css, height=0)
    except Exception as e:
        logger.error(f"Error initializing device detection: {e}")
        # Default to mobile if there's an error
        st.session_state.device_type = 'mobile'
    
    return st.session_state.device_type 