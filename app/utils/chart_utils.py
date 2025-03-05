import streamlit as st
import json
from typing import Dict, Any, Optional, Union
from .color_palette import NEON_COLORS, CHART_STYLING, apply_neon_style
import plotly.graph_objects as go

def get_device_type() -> str:
    """
    Check if the user is on a mobile device based on session state.
    
    Returns:
        str: 'mobile' or 'desktop'
    """
    # Use the session state to determine device type
    # Default to desktop if not set
    return st.session_state.get('device_type', 'desktop')

def update_chart_for_responsive_layout(fig: Any, device_type: Optional[str] = None) -> Any:
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
            margin=dict(t=100, l=30, r=30, b=50),  # Add more top margin to accommodate the legend
            height=350,           # Slightly reduce height for better mobile viewing
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
            margin=dict(t=60, l=50, r=50, b=50)  # Standard margins for desktop
        )
    
    # Apply neon styling to the chart
    fig = apply_neon_style(fig)
    
    return fig

def responsive_plotly_chart(fig: Any, use_container_width: bool = True, **kwargs):
    """
    Render a responsive Plotly chart with appropriate layout for the current device.
    
    Args:
        fig: Plotly figure object
        use_container_width: Whether to use the container width
        **kwargs: Additional arguments to pass to st.plotly_chart
    """
    # Detect device type
    device_type = get_device_type()
    
    # Update chart layout for responsive design
    fig = update_chart_for_responsive_layout(fig, device_type)
    
    # Render the chart
    st.plotly_chart(fig, use_container_width=use_container_width, **kwargs)

def get_neon_color_discrete_sequence(n=None):
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

def init_device_detection():
    """
    Initialize device detection in session state when the app loads.
    Always set to mobile to ensure responsive layout for all users.
    """
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
    
    return st.session_state.device_type 