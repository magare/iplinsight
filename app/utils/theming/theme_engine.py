"""
Theme Engine for Streamlit App

This module provides the core functionality for the theming system, including:
- Theme definition and management
- Theme storage and retrieval
- Theme application to the Streamlit app
"""

import streamlit as st
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging
import base64
import re
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Base paths
THEME_DIR = Path(__file__).resolve().parent.parent.parent / 'static' / 'themes'
THEME_DIR.mkdir(exist_ok=True)

# Default themes
DEFAULT_THEMES = {
    "modern": {
        "name": "Modern",
        "description": "A clean, modern interface with vibrant colors",
        "colors": {
            "primary": "#00ffaa",
            "secondary": "#ff66cc",
            "background": "#0a0a12",
            "text": "#e0e0ff",
            "accent": "#33ccff",
            "success": "#00ff99",
            "warning": "#ffff00",
            "error": "#ff3366",
            "info": "#66ffff"
        },
        "fonts": {
            "base_family": "sans-serif",
            "heading_family": "sans-serif",
            "code_family": "monospace",
            "base_size": "16px",
            "heading_size": "24px",
            "code_size": "14px",
            "base_weight": "400",
            "heading_weight": "600"
        },
        "components": {
            "border_radius": "10px",
            "button_style": "rounded",
            "card_style": "shadow",
            "input_style": "outline"
        },
        "spacing": {
            "base": "1rem",
            "small": "0.5rem",
            "large": "2rem",
            "content_width": "1200px"
        },
        "effects": {
            "use_animations": True,
            "use_shadows": True,
            "glow_effect": True
        }
    },
    "classic": {
        "name": "Classic",
        "description": "A traditional interface with a professional look",
        "colors": {
            "primary": "#1E88E5",
            "secondary": "#6C757D",
            "background": "#FFFFFF",
            "text": "#212529",
            "accent": "#FFC107",
            "success": "#28A745",
            "warning": "#FFC107",
            "error": "#DC3545",
            "info": "#17A2B8"
        },
        "fonts": {
            "base_family": "serif",
            "heading_family": "serif",
            "code_family": "monospace",
            "base_size": "16px",
            "heading_size": "24px",
            "code_size": "14px",
            "base_weight": "400",
            "heading_weight": "700"
        },
        "components": {
            "border_radius": "4px",
            "button_style": "square",
            "card_style": "border",
            "input_style": "underline"
        },
        "spacing": {
            "base": "1rem",
            "small": "0.5rem",
            "large": "2rem",
            "content_width": "1140px"
        },
        "effects": {
            "use_animations": False,
            "use_shadows": False,
            "glow_effect": False
        }
    },
    "minimal": {
        "name": "Minimal",
        "description": "A minimalist interface with focus on content",
        "colors": {
            "primary": "#000000",
            "secondary": "#666666",
            "background": "#FFFFFF",
            "text": "#333333",
            "accent": "#CCCCCC",
            "success": "#4CAF50",
            "warning": "#FF9800",
            "error": "#F44336",
            "info": "#2196F3"
        },
        "fonts": {
            "base_family": "sans-serif",
            "heading_family": "sans-serif",
            "code_family": "monospace",
            "base_size": "16px",
            "heading_size": "24px",
            "code_size": "14px",
            "base_weight": "300",
            "heading_weight": "500"
        },
        "components": {
            "border_radius": "0px",
            "button_style": "flat",
            "card_style": "flat",
            "input_style": "minimal"
        },
        "spacing": {
            "base": "1rem",
            "small": "0.5rem",
            "large": "2rem",
            "content_width": "1000px"
        },
        "effects": {
            "use_animations": False,
            "use_shadows": False,
            "glow_effect": False
        }
    },
    "dark": {
        "name": "Dark",
        "description": "A dark interface with high contrast",
        "colors": {
            "primary": "#BB86FC",
            "secondary": "#03DAC6",
            "background": "#121212",
            "text": "#FFFFFF",
            "accent": "#CF6679",
            "success": "#00C853",
            "warning": "#FFD600",
            "error": "#CF6679",
            "info": "#2196F3"
        },
        "fonts": {
            "base_family": "sans-serif",
            "heading_family": "sans-serif",
            "code_family": "monospace",
            "base_size": "16px",
            "heading_size": "24px",
            "code_size": "14px",
            "base_weight": "400",
            "heading_weight": "600"
        },
        "components": {
            "border_radius": "8px",
            "button_style": "rounded",
            "card_style": "shadow",
            "input_style": "filled"
        },
        "spacing": {
            "base": "1rem",
            "small": "0.5rem",
            "large": "2rem",
            "content_width": "1200px"
        },
        "effects": {
            "use_animations": True,
            "use_shadows": True,
            "glow_effect": True
        }
    },
    "light": {
        "name": "Light",
        "description": "A light interface with subtle colors",
        "colors": {
            "primary": "#1976D2",
            "secondary": "#26A69A",
            "background": "#F5F5F5",
            "text": "#212121",
            "accent": "#FF4081",
            "success": "#4CAF50",
            "warning": "#FFC107",
            "error": "#F44336",
            "info": "#2196F3"
        },
        "fonts": {
            "base_family": "sans-serif",
            "heading_family": "sans-serif",
            "code_family": "monospace",
            "base_size": "16px",
            "heading_size": "24px",
            "code_size": "14px",
            "base_weight": "400",
            "heading_weight": "600"
        },
        "components": {
            "border_radius": "4px",
            "button_style": "rounded",
            "card_style": "shadow",
            "input_style": "outline"
        },
        "spacing": {
            "base": "1rem",
            "small": "0.5rem",
            "large": "2rem",
            "content_width": "1200px"
        },
        "effects": {
            "use_animations": True,
            "use_shadows": True,
            "glow_effect": False
        }
    },
    "cricket": {
        "name": "Cricket",
        "description": "A cricket-themed interface with grass green and pitch brown",
        "colors": {
            "primary": "#4CAF50",  # Grass green
            "secondary": "#8D6E63",  # Pitch brown
            "background": "#FFFFFF",
            "text": "#212121",
            "accent": "#FFC107",  # Cricket ball red
            "success": "#4CAF50",
            "warning": "#FF9800",
            "error": "#F44336",
            "info": "#2196F3"
        },
        "fonts": {
            "base_family": "sans-serif",
            "heading_family": "sans-serif",
            "code_family": "monospace",
            "base_size": "16px",
            "heading_size": "24px",
            "code_size": "14px",
            "base_weight": "400",
            "heading_weight": "600"
        },
        "components": {
            "border_radius": "4px",
            "button_style": "rounded",
            "card_style": "shadow",
            "input_style": "outline"
        },
        "spacing": {
            "base": "1rem",
            "small": "0.5rem",
            "large": "2rem",
            "content_width": "1200px"
        },
        "effects": {
            "use_animations": True,
            "use_shadows": True,
            "glow_effect": False
        }
    }
}

class ThemeEngine:
    """Core class for managing and applying themes."""
    
    @staticmethod
    def initialize() -> None:
        """Initialize the theming system."""
        # Create theme directory if it doesn't exist
        THEME_DIR.mkdir(exist_ok=True)
        
        # Initialize session state for theming
        if 'theme' not in st.session_state:
            st.session_state.theme = DEFAULT_THEMES['modern']
        
        if 'theme_name' not in st.session_state:
            st.session_state.theme_name = 'modern'
        
        if 'custom_themes' not in st.session_state:
            st.session_state.custom_themes = ThemeEngine._load_custom_themes()
        
        if 'custom_css' not in st.session_state:
            st.session_state.custom_css = ""
        
        # Apply the current theme
        ThemeEngine.apply_theme(st.session_state.theme)
    
    @staticmethod
    def get_theme(theme_name: str) -> Dict[str, Any]:
        """
        Get a theme by name.
        
        Args:
            theme_name: Name of the theme to retrieve
            
        Returns:
            The theme dictionary
        """
        if theme_name in DEFAULT_THEMES:
            return DEFAULT_THEMES[theme_name]
        
        if theme_name in st.session_state.custom_themes:
            return st.session_state.custom_themes[theme_name]
        
        logger.warning(f"Theme '{theme_name}' not found, returning default theme")
        return DEFAULT_THEMES['modern']
    
    @staticmethod
    def set_theme(theme_name: str) -> None:
        """
        Set the current theme by name.
        
        Args:
            theme_name: Name of the theme to set
        """
        theme = ThemeEngine.get_theme(theme_name)
        st.session_state.theme = theme
        st.session_state.theme_name = theme_name
        ThemeEngine.apply_theme(theme)
        logger.info(f"Set theme to '{theme_name}'")
    
    @staticmethod
    def apply_theme(theme: Dict[str, Any]) -> None:
        """
        Apply a theme to the Streamlit app.
        
        Args:
            theme: Theme dictionary to apply
        """
        # Generate CSS from theme
        css = ThemeEngine._generate_css(theme)
        
        # Add custom CSS if any
        if st.session_state.custom_css:
            css += f"\n\n/* Custom CSS */\n{st.session_state.custom_css}"
        
        # Apply CSS
        ThemeEngine._inject_css(css)
    
    @staticmethod
    def save_theme(theme: Dict[str, Any], name: str) -> None:
        """
        Save a custom theme.
        
        Args:
            theme: Theme dictionary to save
            name: Name to save the theme as
        """
        # Add metadata
        theme['name'] = name
        theme['created'] = datetime.now().isoformat()
        
        # Save to session state
        st.session_state.custom_themes[name] = theme
        
        # Save to file
        ThemeEngine._save_custom_themes()
        
        logger.info(f"Saved custom theme '{name}'")
    
    @staticmethod
    def delete_theme(name: str) -> bool:
        """
        Delete a custom theme.
        
        Args:
            name: Name of the theme to delete
            
        Returns:
            True if deleted, False if not found or is a default theme
        """
        if name in DEFAULT_THEMES:
            logger.warning(f"Cannot delete default theme '{name}'")
            return False
        
        if name in st.session_state.custom_themes:
            del st.session_state.custom_themes[name]
            ThemeEngine._save_custom_themes()
            logger.info(f"Deleted custom theme '{name}'")
            return True
        
        logger.warning(f"Theme '{name}' not found for deletion")
        return False
    
    @staticmethod
    def export_theme(name: str) -> Optional[str]:
        """
        Export a theme as JSON.
        
        Args:
            name: Name of the theme to export
            
        Returns:
            JSON string of the theme or None if not found
        """
        theme = None
        if name in DEFAULT_THEMES:
            theme = DEFAULT_THEMES[name]
        elif name in st.session_state.custom_themes:
            theme = st.session_state.custom_themes[name]
        
        if theme:
            return json.dumps(theme, indent=2)
        
        logger.warning(f"Theme '{name}' not found for export")
        return None
    
    @staticmethod
    def import_theme(theme_json: str) -> Optional[Dict[str, Any]]:
        """
        Import a theme from JSON.
        
        Args:
            theme_json: JSON string of the theme
            
        Returns:
            The imported theme dictionary or None if invalid
        """
        try:
            theme = json.loads(theme_json)
            
            # Validate theme structure
            required_keys = ['name', 'colors', 'fonts', 'components', 'spacing', 'effects']
            if not all(key in theme for key in required_keys):
                logger.error(f"Invalid theme format, missing required keys: {required_keys}")
                return None
            
            # Save the theme
            name = theme['name']
            ThemeEngine.save_theme(theme, name)
            return theme
            
        except json.JSONDecodeError:
            logger.error("Invalid JSON format for theme import")
            return None
        except Exception as e:
            logger.error(f"Error importing theme: {e}")
            return None
    
    @staticmethod
    def set_custom_css(css: str) -> None:
        """
        Set custom CSS to be applied on top of the theme.
        
        Args:
            css: Custom CSS string
        """
        st.session_state.custom_css = css
        ThemeEngine.apply_theme(st.session_state.theme)
        logger.info("Applied custom CSS")
    
    @staticmethod
    def get_all_themes() -> Dict[str, Dict[str, Any]]:
        """
        Get all available themes.
        
        Returns:
            Dictionary of all themes (default and custom)
        """
        all_themes = {**DEFAULT_THEMES, **st.session_state.custom_themes}
        return all_themes
    
    @staticmethod
    def _generate_css(theme: Dict[str, Any]) -> str:
        """
        Generate CSS from a theme dictionary.
        
        Args:
            theme: Theme dictionary
            
        Returns:
            CSS string
        """
        colors = theme['colors']
        fonts = theme['fonts']
        components = theme['components']
        spacing = theme['spacing']
        effects = theme['effects']
        
        # Base CSS
        css = f"""
        /* Generated theme CSS */
        
        /* Main app background and text colors */
        .stApp {{
            background-color: {colors['background']};
            color: {colors['text']};
            font-family: {fonts['base_family']};
            font-size: {fonts['base_size']};
            font-weight: {fonts['base_weight']};
        }}
        
        /* Sidebar styling */
        .css-1d391kg, .css-1siy2j7 {{
            background-color: {ThemeEngine._adjust_color(colors['background'], 0.1)} !important;
        }}
        
        /* Headers */
        h1, h2, h3, h4 {{
            font-family: {fonts['heading_family']} !important;
            font-weight: {fonts['heading_weight']} !important;
            color: {colors['primary']} !important;
        """
        
        # Add glow effect if enabled
        if effects['glow_effect']:
            css += f"""
            text-shadow: 0 0 8px {ThemeEngine._with_alpha(colors['primary'], 0.5)} !important;
            """
        
        css += """
        }
        
        h1 {
            font-size: 2.5rem !important;
        }
        
        h2 {
            font-size: 2rem !important;
        }
        
        h3 {
            font-size: 1.5rem !important;
        }
        """
        
        # Buttons
        css += f"""
        /* Buttons */
        button, .stButton > button, [data-testid="stTabs"] button {{
            background-color: {colors['primary']} !important;
            color: {ThemeEngine._contrast_color(colors['primary'])} !important;
            border-radius: {components['border_radius']} !important;
            border: none !important;
            padding: {spacing['small']} {spacing['base']} !important;
            transition: all 0.3s ease !important;
        """
        
        if effects['use_shadows']:
            css += f"""
            box-shadow: 0 2px 5px {ThemeEngine._with_alpha('#000000', 0.2)} !important;
            """
        
        css += """
        }
        
        button:hover, .stButton > button:hover, [data-testid="stTabs"] button:hover {
        """
        
        if effects['use_shadows']:
            css += f"""
            box-shadow: 0 4px 10px {ThemeEngine._with_alpha('#000000', 0.3)} !important;
            """
        
        css += f"""
            background-color: {ThemeEngine._adjust_color(colors['primary'], 0.1)} !important;
            transform: translateY(-2px);
        }}
        """
        
        # Inputs
        css += f"""
        /* Inputs */
        .stTextInput > div > div > input, .stNumberInput > div > div > input {{
            border-radius: {components['border_radius']} !important;
            border-color: {colors['accent']} !important;
        }}
        
        .stTextInput > div > div > input:focus, .stNumberInput > div > div > input:focus {{
            border-color: {colors['primary']} !important;
            box-shadow: 0 0 0 2px {ThemeEngine._with_alpha(colors['primary'], 0.3)} !important;
        }}
        
        /* Selectbox */
        .stSelectbox > div > div {{
            border-radius: {components['border_radius']} !important;
        }}
        
        /* Slider */
        .stSlider > div > div > div {{
            background-color: {colors['primary']} !important;
        }}
        
        /* Radio and Checkbox */
        .stRadio > div, .stCheckbox > div {{
            background-color: {ThemeEngine._with_alpha(colors['background'], 0.5)} !important;
            padding: {spacing['small']} !important;
            border-radius: {components['border_radius']} !important;
        }}
        """
        
        # Cards and containers
        css += f"""
        /* Cards and containers */
        .element-container, .stMarkdown, .stDataFrame {{
            background-color: {ThemeEngine._with_alpha(colors['background'], 0.5)} !important;
            border-radius: {components['border_radius']} !important;
            padding: {spacing['small']} !important;
            margin-bottom: {spacing['base']} !important;
        """
        
        if components['card_style'] == 'shadow' and effects['use_shadows']:
            css += f"""
            box-shadow: 0 4px 12px {ThemeEngine._with_alpha('#000000', 0.1)} !important;
            """
        elif components['card_style'] == 'border':
            css += f"""
            border: 1px solid {ThemeEngine._with_alpha(colors['accent'], 0.3)} !important;
            """
        
        css += """
        }
        """
        
        # Tables
        css += f"""
        /* Tables */
        .stDataFrame table {{
            border-collapse: collapse !important;
            width: 100% !important;
        }}
        
        .stDataFrame th {{
            background-color: {colors['primary']} !important;
            color: {ThemeEngine._contrast_color(colors['primary'])} !important;
            padding: {spacing['small']} !important;
        }}
        
        .stDataFrame td {{
            padding: {spacing['small']} !important;
            border-bottom: 1px solid {ThemeEngine._with_alpha(colors['accent'], 0.2)} !important;
        }}
        
        /* Expanders */
        .streamlit-expanderHeader {{
            background-color: {ThemeEngine._with_alpha(colors['secondary'], 0.1)} !important;
            border-radius: {components['border_radius']} !important;
        }}
        """
        
        # Scrollbars
        css += f"""
        /* Scrollbars */
        ::-webkit-scrollbar {{
            width: 10px;
            height: 10px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: {ThemeEngine._with_alpha(colors['background'], 0.8)};
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: {colors['secondary']};
            border-radius: 5px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: {colors['primary']};
        }}
        """
        
        # Responsive design
        css += """
        /* Responsive design */
        @media (max-width: 768px) {
            h1 {
                font-size: 2rem !important;
            }
            
            h2 {
                font-size: 1.5rem !important;
            }
            
            h3 {
                font-size: 1.2rem !important;
            }
            
            .row-widget.stHorizontal {
                flex-wrap: wrap;
            }
            
            .block-container {
                padding: 1rem !important;
            }
            
            .stMarkdown {
                padding: 0.5rem !important;
            }
        }
        """
        
        # Animations if enabled
        if effects['use_animations']:
            css += """
            /* Animations */
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .element-container {
                animation: fadeIn 0.5s ease-out;
            }
            """
        
        return css
    
    @staticmethod
    def _inject_css(css: str) -> None:
        """
        Inject CSS into the Streamlit app.
        
        Args:
            css: CSS string to inject
        """
        st.markdown(f"""
        <style>
        {css}
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def _load_custom_themes() -> Dict[str, Dict[str, Any]]:
        """
        Load custom themes from files.
        
        Returns:
            Dictionary of custom themes
        """
        custom_themes = {}
        
        try:
            # Create theme directory if it doesn't exist
            THEME_DIR.mkdir(exist_ok=True)
            
            # Load each theme file
            for theme_file in THEME_DIR.glob('*.json'):
                try:
                    with open(theme_file, 'r') as f:
                        theme = json.load(f)
                        name = theme.get('name', theme_file.stem)
                        custom_themes[name] = theme
                except Exception as e:
                    logger.error(f"Error loading theme file {theme_file}: {e}")
        except Exception as e:
            logger.error(f"Error loading custom themes: {e}")
        
        return custom_themes
    
    @staticmethod
    def _save_custom_themes() -> None:
        """Save custom themes to files."""
        try:
            # Create theme directory if it doesn't exist
            THEME_DIR.mkdir(exist_ok=True)
            
            # Clear existing theme files
            for theme_file in THEME_DIR.glob('*.json'):
                theme_file.unlink()
            
            # Save each custom theme
            for name, theme in st.session_state.custom_themes.items():
                file_path = THEME_DIR / f"{name}.json"
                with open(file_path, 'w') as f:
                    json.dump(theme, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving custom themes: {e}")
    
    @staticmethod
    def _adjust_color(color: str, amount: float) -> str:
        """
        Lighten or darken a color.
        
        Args:
            color: Hex color string
            amount: Amount to adjust (positive for lighter, negative for darker)
            
        Returns:
            Adjusted hex color string
        """
        # Convert hex to RGB
        color = color.lstrip('#')
        r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
        
        # Adjust RGB values
        if amount >= 0:
            # Lighten
            r = min(255, int(r + (255 - r) * amount))
            g = min(255, int(g + (255 - g) * amount))
            b = min(255, int(b + (255 - b) * amount))
        else:
            # Darken
            amount = abs(amount)
            r = max(0, int(r * (1 - amount)))
            g = max(0, int(g * (1 - amount)))
            b = max(0, int(b * (1 - amount)))
        
        # Convert back to hex
        return f"#{r:02x}{g:02x}{b:02x}"
    
    @staticmethod
    def _with_alpha(color: str, alpha: float) -> str:
        """
        Add alpha channel to a color.
        
        Args:
            color: Hex color string
            alpha: Alpha value (0-1)
            
        Returns:
            RGBA color string
        """
        color = color.lstrip('#')
        r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    
    @staticmethod
    def _contrast_color(color: str) -> str:
        """
        Get a contrasting color (black or white) for text on a background.
        
        Args:
            color: Hex color string
            
        Returns:
            '#FFFFFF' or '#000000' depending on contrast
        """
        color = color.lstrip('#')
        r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
        
        # Calculate luminance
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        
        # Return black for light colors, white for dark colors
        return '#000000' if luminance > 0.5 else '#FFFFFF' 