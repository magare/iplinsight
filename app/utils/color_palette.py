"""
This module provides consistent color palettes for the IPL Data Analysis app.
It ensures that all charts and visualizations have a cohesive, neon-styled look.
"""

# Main neon color palette
NEON_COLORS = [
    '#00ffaa',  # Neon green
    '#ff66cc',  # Neon pink
    '#66ffff',  # Neon cyan
    '#ff9900',  # Neon orange
    '#ffff00',  # Neon yellow
    '#ff3366',  # Neon red
    '#cc99ff',  # Neon purple
    '#00ff99',  # Neon mint
    '#ff6600',  # Neon amber
    '#33ccff',  # Neon blue
]

# Extended color palette for when more colors are needed
EXTENDED_NEON_COLORS = NEON_COLORS + [
    '#ff0066',  # Bright magenta
    '#00ff66',  # Bright lime
    '#cc00ff',  # Bright violet
    '#ffcc00',  # Bright gold
    '#00ccff',  # Bright sky blue
    '#ff3399',  # Bright rose
    '#33ff33',  # Bright green
    '#9966ff',  # Bright lavender
    '#ff6633',  # Bright coral
    '#66ff33',  # Bright chartreuse
]

# Team-specific colors (mapping IPL teams to neon colors)
TEAM_COLORS = {
    'Mumbai Indians': '#33ccff',        # Neon blue
    'Chennai Super Kings': '#ffff00',   # Neon yellow
    'Royal Challengers Bangalore': '#ff3366',  # Neon red
    'Kolkata Knight Riders': '#cc99ff',  # Neon purple
    'Delhi Capitals': '#66ffff',        # Neon cyan
    'Punjab Kings': '#ff6600',          # Neon amber
    'Rajasthan Royals': '#ff66cc',      # Neon pink
    'Sunrisers Hyderabad': '#ff9900',   # Neon orange
    'Gujarat Titans': '#00ffaa',        # Neon green
    'Lucknow Super Giants': '#33ff33',  # Bright green
    'Deccan Chargers': '#ff0066',       # Bright magenta
    'Pune Warriors': '#00ff66',         # Bright lime
    'Rising Pune Supergiant': '#cc00ff',  # Bright violet
    'Kochi Tuskers Kerala': '#ffcc00',  # Bright gold
    'Gujarat Lions': '#ff3399',         # Bright rose
    'Delhi Daredevils': '#9966ff',      # Bright lavender
    # Fallback for any other teams
    'default': '#00ff99',               # Neon mint
}

# Chart specific color maps
METRIC_COLORS = {
    'positive': '#00ffaa',  # Neon green for positive metrics
    'negative': '#ff3366',  # Neon red for negative metrics
    'neutral': '#66ffff',   # Neon cyan for neutral metrics
}

# Background and styling settings for Plotly charts
CHART_STYLING = {
    'template': 'plotly_dark',
    'plot_bgcolor': 'rgba(0,0,0,0)',  # Transparent background
    'paper_bgcolor': 'rgba(0,0,0,0)',  # Transparent paper
    'font': {'color': '#e0e0ff'},     # Light blue-white text
    'title_font': {'color': '#33ffcc'},  # Neon teal for titles
    'legend_font': {'color': '#e0e0ff'},  # Light blue-white for legend
    'colorway': NEON_COLORS,  # Default colorway
}

# Function to get team color or default if not found
def get_team_color(team_name):
    """Get the neon color for a specific team or default if not found."""
    return TEAM_COLORS.get(team_name, TEAM_COLORS['default'])

# Function to style a Plotly figure with the neon theme
def apply_neon_style(fig):
    """Apply the neon styling to a Plotly figure."""
    fig.update_layout(
        plot_bgcolor=CHART_STYLING['plot_bgcolor'],
        paper_bgcolor=CHART_STYLING['paper_bgcolor'],
        font=CHART_STYLING['font'],
        title_font=CHART_STYLING['title_font'],
        colorway=CHART_STYLING['colorway'],
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.1)',
            zerolinecolor='rgba(128,128,128,0.3)'
        ),
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.1)',
            zerolinecolor='rgba(128,128,128,0.3)'
        )
    )
    return fig 