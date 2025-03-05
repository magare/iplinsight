import streamlit as st
st.set_page_config(
    page_title="IPL Data Explorer 🏏",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

import base64
from pathlib import Path
from utils.data_loader import load_data, calculate_basic_stats, format_large_number
from utils.chart_utils import init_device_detection, responsive_plotly_chart
from components.overview import (
    plot_tournament_growth,
    display_team_participation,
    display_dataset_info,
    display_key_questions,
    display_data_limitations
)
from components.team_analysis import display_team_analysis
from components.player_analysis import (
    display_batting_analysis,
    display_bowling_analysis,
    display_allrounder_analysis,
    display_head_to_head_analysis
)
from components.match_analysis import (
    display_match_result_analysis,
    display_toss_analysis,
    display_scoring_analysis,
    display_high_low_scoring_analysis,
    display_match_details
)
from components.season_analysis import display_season_analysis
from components.venue_analysis import display_venue_analysis
from components.dream_team_analysis import DreamTeamAnalysis

# Load CSS
def load_css():
    css_file = Path(__file__).parent / "static" / "style.css"
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
    </style>
    """, unsafe_allow_html=True)

try:
    load_css()
except Exception as e:
    st.error(f"Failed to load CSS: {e}")

# Initialize device detection
device_type = init_device_detection()

# Add a note about mobile optimization
if device_type == 'mobile':
    st.sidebar.markdown("""
    <div style="background-color: rgba(30, 30, 60, 0.7); padding: 10px; border-radius: 5px; margin-bottom: 15px; font-size: 0.8rem;">
        <p style="margin: 0;">📱 <b>Mobile-optimized view enabled</b></p>
        <p style="margin: 0; opacity: 0.8;">Charts and layouts are adjusted for better mobile experience.</p>
    </div>
    """, unsafe_allow_html=True)

# Cache data loading to improve performance and wrap it with a spinner for user feedback
@st.cache_data(show_spinner=False)
def cached_load_data():
    return load_data()

try:
    with st.spinner("Loading data..."):
        matches, deliveries = cached_load_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    matches, deliveries = None, None

# Define functions for each tab to enable lazy loading
def render_overview():
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
    
    st.header("IPL Data Analysis Overview")
    st.markdown("""
    The Indian Premier League (IPL) has captivated cricket fans worldwide since its inception in 2008. 
    This analysis dives deep into the rich history of IPL, uncovering patterns and insights from over a decade of matches.
    """)
    # Calculate basic stats
    stats = calculate_basic_stats(matches, deliveries)
    # Display metrics in three columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Tournament Scale")
        st.metric("Total Matches", format_large_number(stats['total_matches']))
        st.metric("Unique Teams", format_large_number(stats['total_teams']))
        st.metric("Seasons", format_large_number(stats['total_seasons']))
        st.metric("Venues", format_large_number(stats['total_venues']))
        st.metric("Cities", format_large_number(stats['total_cities']))
    with col2:
        st.subheader("Batting Insights")
        st.metric("Total Runs", format_large_number(stats['total_runs']))
        st.metric("Total Boundaries", format_large_number(stats['total_boundaries']))
        st.metric("Total Sixes", format_large_number(stats['total_sixes']))
        st.metric("Avg. Runs/Match", f"{stats['avg_runs_per_match']:.1f}")
    with col3:
        st.subheader("Bowling Insights")
        st.metric("Total Wickets", format_large_number(stats['total_wickets']))
        st.metric("Total Overs", format_large_number(int(stats['total_overs'])))
        st.metric("Avg. Wickets/Match", f"{stats['avg_wickets_per_match']:.1f}")
    
    st.subheader("Tournament Growth Over Years")
    plot_tournament_growth(matches, deliveries)
    
    # Team Participation Table
    display_team_participation(matches)
    
    # Key questions and data limitations
    display_key_questions()
    display_data_limitations()
    
    # Dataset structure information (moved to the bottom)
    display_dataset_info()

def render_team_analysis():
    display_team_analysis(matches, deliveries)

def render_player_analysis():
    st.header("Player Analysis")
    
    # Get device type to determine layout
    device_type = st.session_state.get('device_type', 'mobile')
    
    # Define analysis options
    analysis_options = ["Batting Analysis", "Bowling Analysis", "All-Rounder Analysis", "Head-to-Head Analysis"]
    
    if device_type == 'mobile':
        # Use a selectbox for mobile view
        selected_analysis = st.selectbox("Select Analysis", analysis_options)
        
        # Display the selected analysis
        if selected_analysis == "Batting Analysis":
            display_batting_analysis(deliveries)
        elif selected_analysis == "Bowling Analysis":
            display_bowling_analysis(deliveries)
        elif selected_analysis == "All-Rounder Analysis":
            display_allrounder_analysis(deliveries)
        elif selected_analysis == "Head-to-Head Analysis":
            display_head_to_head_analysis(deliveries)
    else:
        # Use tabs for desktop view
        tabs = st.tabs(analysis_options)
        with tabs[0]:
            display_batting_analysis(deliveries)
        with tabs[1]:
            display_bowling_analysis(deliveries)
        with tabs[2]:
            display_allrounder_analysis(deliveries)
        with tabs[3]:
            display_head_to_head_analysis(deliveries)

def render_match_analysis():
    st.header("Match Analysis")
    st.markdown("""
    This section provides a detailed analysis of match outcomes, toss decisions, and their impact on the game.
    We explore various aspects like victory margins, toss advantage, and scoring patterns.
    """)
    
    # Get device type to determine layout
    device_type = st.session_state.get('device_type', 'mobile')
    
    # Define analysis options
    match_options = ["Match Details", "Match Results", "Toss Analysis", "Scoring Patterns", "High/Low Scoring Matches"]
    
    # Use a selectbox for both mobile and desktop for consistency
    match_tab = st.selectbox("Select Match Analysis", match_options)
    
    if match_tab == "Match Results":
        display_match_result_analysis(matches)
    elif match_tab == "Toss Analysis":
        display_toss_analysis(matches)
    elif match_tab == "Scoring Patterns":
        display_scoring_analysis(matches, deliveries)
    elif match_tab == "High/Low Scoring Matches":
        display_high_low_scoring_analysis(matches, deliveries)
    elif match_tab == "Match Details":
        display_match_details(matches)

def render_season_analysis():
    display_season_analysis(matches, deliveries)

def render_venue_analysis():
    display_venue_analysis(matches, deliveries)

def render_dream_team_analysis():
    dt_instance = DreamTeamAnalysis()
    dt_instance.create_layout()

# Enhance the sidebar with a header and navigation sections
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <h3 style="margin-bottom: 10px;">Analysis Navigator 🧭</h3>
    <p style="font-size: 0.9rem; opacity: 0.8;">Explore different aspects of IPL data</p>
</div>
""", unsafe_allow_html=True)

# Main navigation using a sidebar radio button for lazy loading
selected_tab = st.sidebar.radio("", 
                                  ["Overview", "Team Analysis", "Player Analysis", 
                                   "Match Analysis", "Season Analysis", 
                                   "Venue Analysis", "Dream Team Analysis"])

# Add information at the bottom of the sidebar
st.sidebar.markdown("---")

# Get the min and max seasons from the data
min_season = int(matches['season'].min())
max_season = int(matches['season'].max())
season_range = f"{min_season}-{max_season}"

st.sidebar.markdown(f"""
<div style="opacity: 0.7; font-size: 0.8rem; text-align: center;">
    <p>Made with ❤️ for cricket analytics</p>
    <p>Data from {season_range} IPL seasons</p>
    <p>Data source: <a href="https://cricsheet.org/" target="_blank">CricSheet</a></p>
</div>
""", unsafe_allow_html=True)

# Apply custom styling to fix tab underline issue
st.markdown("""
<style>
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

/* Fix for button hover issue */
button, .stButton > button, [data-testid="stTabs"] button, [data-testid="stTabs"] [role="tab"] {
    position: relative !important;
    z-index: 10 !important;
    margin-top: 5px !important;
}

button:hover, .stButton > button:hover, [data-testid="stTabs"] button:hover, [data-testid="stTabs"] [role="tab"]:hover {
    z-index: 100 !important;
    transform: translateY(-2px) !important;
}
</style>
""", unsafe_allow_html=True)

if selected_tab == "Overview":
    render_overview()
elif selected_tab == "Team Analysis":
    render_team_analysis()
elif selected_tab == "Player Analysis":
    render_player_analysis()
elif selected_tab == "Match Analysis":
    render_match_analysis()
elif selected_tab == "Season Analysis":
    render_season_analysis()
elif selected_tab == "Venue Analysis":
    render_venue_analysis()
elif selected_tab == "Dream Team Analysis":
    render_dream_team_analysis()
