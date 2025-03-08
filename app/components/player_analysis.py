import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from utils.chart_utils import responsive_plotly_chart
from pathlib import Path

def load_player_analysis_data():
    """Load pre-computed data for player analysis section."""
    base_path = Path(__file__).resolve().parent.parent / "data"
    
    # Load pre-computed datasets
    batting_stats = pd.read_parquet(base_path / "player_batting_stats.parquet")
    batting_phase_stats = pd.read_parquet(base_path / "player_batting_phase_stats.parquet")
    position_stats = pd.read_parquet(base_path / "player_position_stats.parquet")
    bowling_stats = pd.read_parquet(base_path / "player_bowling_stats.parquet")
    wicket_types = pd.read_parquet(base_path / "player_wicket_types.parquet")
    bowling_phase_stats = pd.read_parquet(base_path / "player_bowling_phase_stats.parquet")
    allrounder_stats = pd.read_parquet(base_path / "player_allrounder_stats.parquet")
    h2h_stats = pd.read_parquet(base_path / "player_h2h_stats.parquet")
    
    return {
        'batting_stats': batting_stats,
        'batting_phase_stats': batting_phase_stats,
        'position_stats': position_stats,
        'bowling_stats': bowling_stats,
        'wicket_types': wicket_types,
        'bowling_phase_stats': bowling_phase_stats,
        'allrounder_stats': allrounder_stats,
        'h2h_stats': h2h_stats
    }

def calculate_batting_stats(deliveries_df=None):
    """Calculate comprehensive batting statistics using pre-computed data when available."""
    # Try to load pre-computed data first
    try:
        data = load_player_analysis_data()
        return data['batting_stats']
    except (FileNotFoundError, KeyError):
        # Fall back to calculating from scratch if pre-computed data is not available
        if deliveries_df is None:
            raise ValueError("Deliveries DataFrame is required when pre-computed data is not available")
            
        # Group by batter for basic stats
        batting_stats = deliveries_df.groupby('batter').agg({
            'batsman_runs': ['sum', 'count'],  # runs and balls faced
            'match_id': 'nunique',  # number of matches
            'is_wicket': 'sum'  # number of dismissals
        }).reset_index()
        
        # Flatten column names
        batting_stats.columns = ['batter', 'runs', 'balls_faced', 'matches', 'dismissals']
        
        # Calculate derived metrics
        batting_stats['batting_average'] = batting_stats['runs'] / batting_stats['dismissals'].replace(0, 1)
        batting_stats['batting_strike_rate'] = (batting_stats['runs'] / batting_stats['balls_faced']) * 100
        
        # Calculate boundary stats
        boundary_stats = deliveries_df.groupby('batter').agg({
            'ball': 'count',  # total balls
            'batsman_runs': [
                ('dot_balls', lambda x: (x == 0).sum()),  # dot balls
                ('boundaries', lambda x: ((x == 4) | (x == 6)).sum()),  # boundaries
                ('fours', lambda x: (x == 4).sum()),  # fours
                ('sixes', lambda x: (x == 6).sum())  # sixes
            ]
        }).reset_index()
        
        # Flatten column names
        boundary_stats.columns = ['batter', 'total_balls', 'dot_balls', 'boundaries', 'fours', 'sixes']
        
        # Merge stats
        batting_stats = pd.merge(batting_stats, boundary_stats, on='batter')
        
        # Calculate percentages
        batting_stats['dot_ball_percentage'] = (batting_stats['dot_balls'] / batting_stats['total_balls']) * 100
        batting_stats['boundary_percentage'] = (batting_stats['boundaries'] / batting_stats['total_balls']) * 100
        batting_stats['runs_per_boundary'] = batting_stats['runs'] / batting_stats['boundaries'].replace(0, 1)
        
        # Rename matches column to be consistent with merge suffixes
        batting_stats = batting_stats.rename(columns={'matches': 'matches_batting'})
        
        return batting_stats

def calculate_milestone_stats(deliveries_df=None):
    """Calculate batting milestones (30s, 50s, 100s) using pre-computed data when available."""
    # Try to load pre-computed data first
    try:
        data = load_player_analysis_data()
        batting_stats = data['batting_stats']
        return batting_stats[['batter', 'thirties', 'fifties', 'hundreds']]
    except (FileNotFoundError, KeyError):
        # Fall back to calculating from scratch if pre-computed data is not available
        if deliveries_df is None:
            raise ValueError("Deliveries DataFrame is required when pre-computed data is not available")
            
        # Group by match and batter to get individual innings scores
        innings_scores = deliveries_df.groupby(['match_id', 'batter'])['batsman_runs'].sum().reset_index()
        
        # Calculate milestones for each batter
        milestones = innings_scores.groupby('batter').agg({
            'batsman_runs': lambda x: [
                sum((x >= 30) & (x < 50)),  # 30s
                sum((x >= 50) & (x < 100)),  # 50s
                sum(x >= 100)  # 100s
            ]
        }).reset_index()
        
        # Convert list to separate columns
        milestones[['thirties', 'fifties', 'hundreds']] = pd.DataFrame(
            milestones['batsman_runs'].tolist(), 
            index=milestones.index
        )
        
        return milestones[['batter', 'thirties', 'fifties', 'hundreds']]

def calculate_phase_stats(deliveries_df=None):
    """Calculate batting stats by match phase using pre-computed data when available."""
    # Try to load pre-computed data first
    try:
        data = load_player_analysis_data()
        return data['batting_phase_stats']
    except (FileNotFoundError, KeyError):
        # Fall back to calculating from scratch if pre-computed data is not available
        if deliveries_df is None:
            raise ValueError("Deliveries DataFrame is required when pre-computed data is not available")
            
        # Define match phases
        deliveries_df['phase'] = pd.cut(
            deliveries_df['over'],
            bins=[-1, 5, 15, 20],
            labels=['Powerplay', 'Middle Overs', 'Death Overs']
        )
        
        # Calculate stats by phase
        phase_stats = deliveries_df.groupby(['batter', 'phase']).agg({
            'batsman_runs': ['sum', 'count'],
            'is_wicket': 'sum'
        }).reset_index()
        
        # Flatten columns and calculate strike rate
        phase_stats.columns = ['batter', 'phase', 'runs', 'balls', 'dismissals']
        phase_stats['batting_strike_rate'] = (phase_stats['runs'] / phase_stats['balls']) * 100
        phase_stats['batting_average'] = phase_stats['runs'] / phase_stats['dismissals'].replace(0, 1)
        
        return phase_stats

def calculate_position_stats(deliveries_df=None):
    """Calculate batting stats by batting position using pre-computed data when available."""
    # Try to load pre-computed data first
    try:
        data = load_player_analysis_data()
        return data['position_stats']
    except (FileNotFoundError, KeyError):
        # Fall back to calculating from scratch if pre-computed data is not available
        if deliveries_df is None:
            raise ValueError("Deliveries DataFrame is required when pre-computed data is not available")
            
        if 'batter_position' not in deliveries_df.columns:
            # Create a copy to avoid modifying original DataFrame
            deliveries_df = deliveries_df.copy()
            # Compute batter_position as the order of appearance per match, inning, and batting_team
            deliveries_df['batter_position'] = deliveries_df.groupby(['match_id', 'inning', 'batting_team'])['batter'].transform(lambda x: pd.factorize(x)[0] + 1)
    
        position_stats = deliveries_df.groupby(['batter', 'batter_position']).agg({
            'batsman_runs': ['sum', 'count'],
            'is_wicket': 'sum',
            'match_id': 'nunique'
        }).reset_index()
        
        # Flatten columns
        position_stats.columns = ['batter', 'position', 'runs', 'balls', 'dismissals', 'innings']
        
        # Calculate metrics
        position_stats['batting_average'] = position_stats['runs'] / position_stats['dismissals'].replace(0, 1)
        position_stats['batting_strike_rate'] = (position_stats['runs'] / position_stats['balls']) * 100
        
        return position_stats

def display_batting_analysis(deliveries_df=None):
    """Display batting analysis dashboard"""
    st.subheader("Batting Analysis")
    
    # Add custom CSS to fix tab styling
    st.markdown("""
    <style>
    /* Fix for red underline in tabs */
    [data-testid="stTabs"] [role="tab"][aria-selected="true"]::before,
    [data-testid="stTabs"] [role="tab"][aria-selected="true"]::after,
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] > div::before,
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] > div::after,
    .st-emotion-cache-1inwz65,
    .st-emotion-cache-1y4pk3h,
    .st-emotion-cache-16idsys {
        display: none !important;
        border: none !important;
        border-bottom: none !important;
        background: none !important;
    }
    
    /* Fix for button hover */
    [data-testid="stTabs"] [role="tab"] {
        position: relative !important;
        z-index: 10 !important;
        margin-top: 5px !important;
    }
    
    [data-testid="stTabs"] [role="tab"]:hover {
        z-index: 100 !important;
        transform: translateY(-2px) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Load data
    if deliveries_df is None:
        st.error("No data available for analysis")
        return
    
    # Calculate stats
    batting_stats = calculate_batting_stats(deliveries_df)
    milestone_stats = calculate_milestone_stats(deliveries_df)
    phase_stats = calculate_phase_stats(deliveries_df)
    position_stats = calculate_position_stats(deliveries_df)
    
    # Filters
    st.markdown("### Filter Players")
    col1, col2 = st.columns(2)
    with col1:
        min_matches = st.slider("Minimum Matches", 1, 100, 10)
    with col2:
        min_runs = st.slider("Minimum Runs", 0, 5000, 200)
    
    # Filter data based on user selection
    filtered_stats = batting_stats[
        (batting_stats['matches_batting'] >= min_matches) & 
        (batting_stats['runs'] >= min_runs)
    ]
    
    # Create tabs for different analyses
    analysis_tabs = st.tabs([
        "Overall Stats",
        "Phase Analysis",
        "Position Analysis"
    ])
    
    # Overall Stats Tab
    with analysis_tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top run scorers
            fig = px.bar(
                filtered_stats.nlargest(10, 'runs'),
                x='batter',
                y='runs',
                title='Top 10 Run Scorers',
                hover_data=['batting_average', 'batting_strike_rate', 'matches_batting']
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                xaxis=dict(
                    tickmode='array',  # Use array tick mode to show all players
                    tickvals=list(range(len(filtered_stats.nlargest(10, 'runs')))),  # Ensure a tick for each player
                    ticktext=filtered_stats.nlargest(10, 'runs')['batter'],
                    automargin=True  # Ensure labels don't get cut off
                )
            )
            responsive_plotly_chart(fig, use_container_width=True)
            
            # Strike Rate vs Average scatter plot
            st.write("### Strike Rate vs Average")
            
            # Create a filtered dataframe with minimum matches requirement to reduce density
            min_matches = 10  # Minimum matches required to be included
            qualified_players = filtered_stats[filtered_stats['matches_batting'] >= min_matches].copy()
            
            # Calculate quadrant boundaries based on median values
            avg_median = qualified_players['batting_average'].median()
            sr_median = qualified_players['batting_strike_rate'].median()
            
            # Label quadrants for interpretation
            qualified_players['performance_quadrant'] = qualified_players.apply(
                lambda x: (
                    "High SR, High Avg<br>(Match Winners)" if x['batting_strike_rate'] >= sr_median and x['batting_average'] >= avg_median else
                    "High SR, Low Avg<br>(Aggressive)" if x['batting_strike_rate'] >= sr_median else
                    "Low SR, High Avg<br>(Anchors)" if x['batting_average'] >= avg_median else
                    "Low SR, Low Avg<br>(Struggling)"
                ),
                axis=1
            )
            
            # Add size based on runs scored for additional context
            qualified_players['marker_size'] = qualified_players['runs'].apply(lambda x: max(10, min(x/50, 30)))
            
            # Color based on boundary percentage for additional insight
            qualified_players['color_value'] = qualified_players['boundary_percentage'] 
            
            # Create the improved scatter plot
            fig = px.scatter(
                qualified_players,
                x='batting_average',
                y='batting_strike_rate',
                color='color_value',
                size='marker_size',
                color_continuous_scale='Viridis',
                hover_name='batter',
                hover_data={
                    'batting_average': ':.2f', 
                    'batting_strike_rate': ':.2f',
                    'runs': True,
                    'matches_batting': True,
                    'boundary_percentage': ':.1f%',
                    'performance_quadrant': True,
                    'marker_size': False,
                    'color_value': False
                },
                labels={
                    'batting_average': 'Batting Average',
                    'batting_strike_rate': 'Strike Rate',
                    'color_value': 'Boundary %'
                }
            )
            
            # Add quadrant lines
            fig.add_shape(type="line", x0=avg_median, y0=0, x1=avg_median, y1=200,
                          line=dict(color="rgba(255,255,255,0.5)", width=1, dash="dash"))
            fig.add_shape(type="line", x0=0, y0=sr_median, x1=100, y1=sr_median,
                          line=dict(color="rgba(255,255,255,0.5)", width=1, dash="dash"))
            
            # Add quadrant labels
            fig.add_annotation(x=avg_median + (100-avg_median)/2, y=sr_median + (200-sr_median)/2, 
                              text="Match Winners<br>High SR, High Avg", showarrow=False, 
                              font=dict(size=10, color="white"), align="center",
                              bgcolor="rgba(0,0,0,0.5)", borderpad=4)
            fig.add_annotation(x=avg_median/2, y=sr_median + (200-sr_median)/2, 
                              text="Aggressive<br>High SR, Low Avg", showarrow=False, 
                              font=dict(size=10, color="white"), align="center",
                              bgcolor="rgba(0,0,0,0.5)", borderpad=4)
            fig.add_annotation(x=avg_median + (100-avg_median)/2, y=sr_median/2, 
                              text="Anchors<br>Low SR, High Avg", showarrow=False, 
                              font=dict(size=10, color="white"), align="center",
                              bgcolor="rgba(0,0,0,0.5)", borderpad=4)
            fig.add_annotation(x=avg_median/2, y=sr_median/2, 
                              text="Struggling<br>Low SR, Low Avg", showarrow=False, 
                              font=dict(size=10, color="white"), align="center",
                              bgcolor="rgba(0,0,0,0.5)", borderpad=4)
            
            # Update layout for better readability
            fig.update_layout(
                coloraxis_colorbar=dict(title="Boundary %"),
                title=None,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=50, r=50, t=30, b=50),
                height=450,
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(
                    gridcolor='rgba(255,255,255,0.1)',
                    tickmode='linear',
                    dtick=100,  # Set appropriate interval for runs
                    automargin=True
                )
            )
            
            # Show player names for top performers only
            top_performers = qualified_players.nlargest(8, 'runs')
            for i, row in top_performers.iterrows():
                fig.add_annotation(
                    x=row['batting_average'],
                    y=row['batting_strike_rate'],
                    text=row['batter'],
                    showarrow=False,
                    font=dict(size=10, color="white"),
                    bgcolor="rgba(0,0,0,0.5)",
                    borderpad=2,
                    yshift=10
                )
            
            responsive_plotly_chart(fig, use_container_width=True)
            
            # Add an explanation for the chart
            st.markdown("""
            <div style="background-color: rgba(30, 30, 60, 0.7); padding: 10px; border-radius: 10px; margin-top: 15px; font-size: 0.9em;">
                <p><b>How to read this chart:</b></p>
                <ul>
                    <li>Each bubble represents a player who has played at least 10 matches</li>
                    <li>Bubble size indicates total runs scored</li>
                    <li>Color represents boundary percentage (brighter = more boundaries)</li>
                    <li>The dashed lines divide players into four performance quadrants</li>
                    <li>Only top run-scorers are labeled to reduce clutter</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Boundary percentage vs Dot Ball percentage - improved
            st.write("### Boundary % vs Dot Ball %")
            
            # Use the same qualified players dataframe to reduce density
            # Add computed field for scoring efficiency
            qualified_players['scoring_efficiency'] = qualified_players['batting_strike_rate'] / 100
            
            # Create the improved scatter plot
            fig = px.scatter(
                qualified_players,
                x='dot_ball_percentage',
                y='boundary_percentage',
                size='marker_size',  # Reuse the same size mapping
                color='scoring_efficiency',
                color_continuous_scale='RdYlGn',
                hover_name='batter',
                hover_data={
                    'dot_ball_percentage': ':.1f%', 
                    'boundary_percentage': ':.1f%',
                    'batting_strike_rate': ':.2f',
                    'runs': True,
                    'matches_batting': True,
                    'marker_size': False,
                    'scoring_efficiency': False
                },
                labels={
                    'dot_ball_percentage': 'Dot Ball %',
                    'boundary_percentage': 'Boundary %',
                    'scoring_efficiency': 'Scoring Efficiency'
                }
            )
            
            # Calculate median values for quadrant lines
            dot_ball_median = qualified_players['dot_ball_percentage'].median()
            boundary_median = qualified_players['boundary_percentage'].median()
            
            # Add quadrant lines
            fig.add_shape(type="line", x0=dot_ball_median, y0=0, x1=dot_ball_median, y1=50,
                         line=dict(color="rgba(255,255,255,0.5)", width=1, dash="dash"))
            fig.add_shape(type="line", x0=0, y0=boundary_median, x1=100, y1=boundary_median,
                         line=dict(color="rgba(255,255,255,0.5)", width=1, dash="dash"))
            
            # Add quadrant labels
            fig.add_annotation(x=dot_ball_median + (100-dot_ball_median)/2, y=boundary_median + (50-boundary_median)/2, 
                              text="High Risk, High Reward<br>(Explosive)", showarrow=False, 
                              font=dict(size=10, color="white"), align="center",
                              bgcolor="rgba(0,0,0,0.5)", borderpad=4)
            fig.add_annotation(x=dot_ball_median/2, y=boundary_median + (50-boundary_median)/2, 
                              text="Low Risk, High Reward<br>(Efficient)", showarrow=False, 
                              font=dict(size=10, color="white"), align="center",
                              bgcolor="rgba(0,0,0,0.5)", borderpad=4)
            fig.add_annotation(x=dot_ball_median + (100-dot_ball_median)/2, y=boundary_median/2, 
                              text="High Risk, Low Reward<br>(Inefficient)", showarrow=False, 
                              font=dict(size=10, color="white"), align="center",
                              bgcolor="rgba(0,0,0,0.5)", borderpad=4)
            fig.add_annotation(x=dot_ball_median/2, y=boundary_median/2, 
                              text="Low Risk, Low Reward<br>(Conservative)", showarrow=False, 
                              font=dict(size=10, color="white"), align="center",
                              bgcolor="rgba(0,0,0,0.5)", borderpad=4)
            
            # Draw a visual reference line for "break-even" efficiency
            x_vals = np.linspace(0, 100, 100)
            y_vals = [100 - x for x in x_vals]  # Simplified model
            fig.add_trace(go.Scatter(
                x=x_vals, 
                y=y_vals, 
                mode='lines', 
                line=dict(color='rgba(255,255,255,0.2)', dash='dot'),
                name='Theoretical Efficiency Line',
                hoverinfo='skip'
            ))
            
            # Update layout for better readability
            fig.update_layout(
                coloraxis_colorbar=dict(title="Scoring<br>Efficiency"),
                title=None,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=50, r=50, t=30, b=50),
                height=450,
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(
                    gridcolor='rgba(255,255,255,0.1)',
                    tickmode='linear',
                    dtick=5,  # Set appropriate interval for wickets
                    automargin=True
                )
            )
            
            # Show player names for top performers only
            # Use different metrics for selecting which players to label
            # Label those with highest boundary % and lowest dot ball %
            efficiency_performers = qualified_players.nlargest(4, 'boundary_percentage')
            efficiency_performers = pd.concat([
                efficiency_performers,
                qualified_players.nsmallest(4, 'dot_ball_percentage')
            ]).drop_duplicates()
            
            for i, row in efficiency_performers.iterrows():
                fig.add_annotation(
                    x=row['dot_ball_percentage'],
                    y=row['boundary_percentage'],
                    text=row['batter'],
                    showarrow=False,
                    font=dict(size=10, color="white"),
                    bgcolor="rgba(0,0,0,0.5)",
                    borderpad=2,
                    yshift=10
                )
            
            responsive_plotly_chart(fig, use_container_width=True)
            
            # Add an explanation for the chart
            st.markdown("""
            <div style="background-color: rgba(30, 30, 60, 0.7); padding: 10px; border-radius: 10px; margin-top: 15px; font-size: 0.9em;">
                <p><b>How to read this chart:</b></p>
                <ul>
                    <li>Each bubble represents a player who has played at least 10 matches</li>
                    <li>Bubble size indicates total runs scored</li>
                    <li>Color represents scoring efficiency (green = more efficient)</li>
                    <li>The dotted diagonal line represents a theoretical balance between dots and boundaries</li>
                    <li>Players are labeled based on exceptional boundary hitting or low dot ball percentage</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Top milestones chart follows below
            milestone_fig = go.Figure(data=[
                go.Bar(name='30s', x=filtered_stats.nlargest(10, 'runs')['batter'], 
                      y=filtered_stats.nlargest(10, 'runs')['thirties']),
                go.Bar(name='50s', x=filtered_stats.nlargest(10, 'runs')['batter'], 
                      y=filtered_stats.nlargest(10, 'runs')['fifties']),
                go.Bar(name='100s', x=filtered_stats.nlargest(10, 'runs')['batter'], 
                      y=filtered_stats.nlargest(10, 'runs')['hundreds'])
            ])
            milestone_fig.update_layout(
                barmode='group',
                title='Batting Milestones (Top 10 Run Scorers)',
                xaxis_tickangle=-45,
                xaxis=dict(
                    tickmode='array',  # Use array tick mode to show all players
                    tickvals=list(range(len(filtered_stats.nlargest(10, 'runs')))),  # Ensure a tick for each player
                    ticktext=filtered_stats.nlargest(10, 'runs')['batter'],
                    automargin=True  # Ensure labels don't get cut off
                )
            )
            responsive_plotly_chart(milestone_fig, use_container_width=True)
    
    # Phase Analysis Tab
    with analysis_tabs[1]:
        st.subheader("Performance by Match Phase")
        selected_player = st.selectbox(
            "Select player for phase-wise analysis",
            options=filtered_stats.nlargest(50, 'runs')['batter'].tolist(),
            key='phase_player'
        )
        
        player_phase_stats = phase_stats[phase_stats['batter'] == selected_player]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Strike rate by phase
            fig = px.bar(
                player_phase_stats,
                x='phase',
                y='batting_strike_rate',
                title=f'Strike Rate by Match Phase - {selected_player}',
                color='phase'
            )
            responsive_plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average by phase
            fig = px.bar(
                player_phase_stats,
                x='phase',
                y='batting_average',
                title=f'Average by Match Phase - {selected_player}',
                color='phase'
            )
            responsive_plotly_chart(fig, use_container_width=True)
    
    # Position Analysis Tab
    with analysis_tabs[2]:
        st.subheader("Performance by Batting Position")
        selected_player = st.selectbox(
            "Select player for position-wise analysis",
            options=filtered_stats.nlargest(50, 'runs')['batter'].tolist(),
            key='position_player'
        )
        
        player_position_stats = position_stats[position_stats['batter'] == selected_player]
        player_position_stats = player_position_stats[player_position_stats['innings'] >= 2]  # At least 2 innings
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Runs by position
            fig = px.bar(
                player_position_stats,
                x='position',
                y='runs',
                title=f'Runs by Batting Position - {selected_player}',
                color='position',
                hover_data=['innings', 'batting_average']
            )
            responsive_plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Strike rate by position
            fig = px.bar(
                player_position_stats,
                x='position',
                y='batting_strike_rate',
                title=f'Strike Rate by Batting Position - {selected_player}',
                color='position',
                hover_data=['innings', 'batting_average']
            )
            responsive_plotly_chart(fig, use_container_width=True)
        
        # Position-wise detailed stats
        st.subheader("Detailed Position-wise Statistics")
        detailed_stats = player_position_stats[['position', 'innings', 'runs', 'batting_average', 'batting_strike_rate']]
        detailed_stats = detailed_stats.sort_values('position')
        st.dataframe(
            detailed_stats.style.format({
                'batting_average': '{:.2f}',
                'batting_strike_rate': '{:.2f}'
            }),
            use_container_width=True
        )

def calculate_bowling_stats(deliveries_df=None):
    """Calculate comprehensive bowling statistics using pre-computed data when available."""
    # Try to load pre-computed data first
    try:
        data = load_player_analysis_data()
        return data['bowling_stats']
    except (FileNotFoundError, KeyError):
        # Fall back to calculating from scratch if pre-computed data is not available
        if deliveries_df is None:
            raise ValueError("Deliveries DataFrame is required when pre-computed data is not available")
            
        # Group by bowler for basic stats
        bowling_stats = deliveries_df.groupby('bowler').agg({
            'total_runs': 'sum',  # runs conceded
            'ball': 'count',  # balls bowled
            'is_wicket': 'sum',  # wickets taken
            'match_id': 'nunique'  # matches played
        }).reset_index()
        
        # Rename matches column to be consistent with merge suffixes
        bowling_stats = bowling_stats.rename(columns={'match_id': 'matches_bowling'})
        
        # Calculate derived metrics
        bowling_stats['overs'] = bowling_stats['ball'] / 6
        bowling_stats['bowling_economy'] = bowling_stats['total_runs'] / bowling_stats['overs']
        bowling_stats['bowling_average'] = bowling_stats['total_runs'] / bowling_stats['is_wicket'].replace(0, 1)  # runs per wicket
        bowling_stats['bowling_strike_rate'] = bowling_stats['ball'] / bowling_stats['is_wicket'].replace(0, 1)  # balls per wicket
        bowling_stats['wickets_per_match'] = bowling_stats['is_wicket'] / bowling_stats['matches_bowling']
        
        # Calculate dot balls
        dot_balls = deliveries_df[deliveries_df['total_runs'] == 0].groupby('bowler').size().reset_index(name='dot_balls')
        bowling_stats = pd.merge(bowling_stats, dot_balls, on='bowler', how='left')
        bowling_stats['dot_ball_percentage'] = (bowling_stats['dot_balls'] / bowling_stats['ball']) * 100
        
        return bowling_stats

def calculate_wicket_types(deliveries_df=None):
    """Calculate wicket types for each bowler using pre-computed data when available."""
    # Try to load pre-computed data first
    try:
        data = load_player_analysis_data()
        return data['wicket_types']
    except (FileNotFoundError, KeyError):
        # Fall back to calculating from scratch if pre-computed data is not available
        if deliveries_df is None:
            raise ValueError("Deliveries DataFrame is required when pre-computed data is not available")
            
        # Filter only wicket deliveries
        wickets = deliveries_df[deliveries_df['is_wicket'] == 1]
        
        # Group by bowler and wicket type
        wicket_types = wickets.groupby(['bowler', 'wicket_kind']).size().reset_index(name='count')
        
        # Pivot table for wicket types
        wicket_types = wicket_types.pivot(
            index='bowler',
            columns='wicket_kind',
            values='count'
        ).fillna(0).reset_index()
        
        return wicket_types

def calculate_bowling_phase_stats(deliveries_df=None):
    """Calculate bowling stats by match phase using pre-computed data when available."""
    # Try to load pre-computed data first
    try:
        data = load_player_analysis_data()
        return data['bowling_phase_stats']
    except (FileNotFoundError, KeyError):
        # Fall back to calculating from scratch if pre-computed data is not available
        if deliveries_df is None:
            raise ValueError("Deliveries DataFrame is required when pre-computed data is not available")
            
        # Define match phases
        deliveries_df['phase'] = pd.cut(
            deliveries_df['over'],
            bins=[-1, 5, 15, 20],
            labels=['Powerplay', 'Middle Overs', 'Death Overs']
        )
        
        # Calculate stats by phase
        phase_stats = deliveries_df.groupby(['bowler', 'phase']).agg({
            'total_runs': 'sum',
            'ball': 'count',
            'is_wicket': 'sum',
            'match_id': 'nunique'
        }).reset_index()
        
        # Calculate metrics
        phase_stats['overs'] = phase_stats['ball'] / 6
        phase_stats['bowling_economy'] = phase_stats['total_runs'] / phase_stats['overs']
        phase_stats['bowling_average'] = phase_stats['total_runs'] / phase_stats['is_wicket'].replace(0, 1)
        phase_stats['bowling_strike_rate'] = phase_stats['ball'] / phase_stats['is_wicket'].replace(0, 1)
        
        return phase_stats

def display_bowling_analysis(deliveries_df=None):
    """Display bowling analysis dashboard"""
    st.subheader("Bowling Analysis")
    
    # Add custom CSS to fix tab styling
    st.markdown("""
    <style>
    /* Fix for red underline in tabs */
    [data-testid="stTabs"] [role="tab"][aria-selected="true"]::before,
    [data-testid="stTabs"] [role="tab"][aria-selected="true"]::after,
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] > div::before,
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] > div::after,
    .st-emotion-cache-1inwz65,
    .st-emotion-cache-1y4pk3h,
    .st-emotion-cache-16idsys {
        display: none !important;
        border: none !important;
        border-bottom: none !important;
        background: none !important;
    }
    
    /* Fix for button hover */
    [data-testid="stTabs"] [role="tab"] {
        position: relative !important;
        z-index: 10 !important;
        margin-top: 5px !important;
    }
    
    [data-testid="stTabs"] [role="tab"]:hover {
        z-index: 100 !important;
        transform: translateY(-2px) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Load data
    if deliveries_df is None:
        st.error("No data available for analysis")
        return
    
    # Calculate stats
    bowling_stats = calculate_bowling_stats(deliveries_df)
    wicket_types = calculate_wicket_types(deliveries_df)
    phase_stats = calculate_bowling_phase_stats(deliveries_df)
    
    # Filters
    st.markdown("### Filter Players")
    col1, col2 = st.columns(2)
    with col1:
        min_overs = st.slider("Minimum overs bowled", 10, 500, 50)
    with col2:
        min_wickets = st.slider("Minimum wickets taken", 1, 100, 20)
    
    # Filter data based on user selection
    filtered_stats = bowling_stats[
        (bowling_stats['overs'] >= min_overs) & 
        (bowling_stats['is_wicket'] >= min_wickets)
    ]
    
    # Create tabs for different analyses
    analysis_tabs = st.tabs([
        "Overall Stats",
        "Wicket Analysis",
        "Phase Analysis"
    ])
    
    # Overall Stats Tab
    with analysis_tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top wicket takers
            fig = px.bar(
                filtered_stats.nlargest(10, 'is_wicket'),
                x='bowler',
                y='is_wicket',
                title='Top 10 Wicket Takers',
                hover_data=['bowling_economy', 'bowling_average', 'bowling_strike_rate']
            )
            fig.update_layout(xaxis_tickangle=-45)
            responsive_plotly_chart(fig, use_container_width=True)
            
            # Economy vs Strike Rate
            fig = px.scatter(
                filtered_stats,
                x='bowling_economy',
                y='bowling_strike_rate',
                text='bowler',
                title='Economy Rate vs Strike Rate',
                hover_data=['is_wicket', 'bowling_average']
            )
            fig.update_traces(textposition='top center')
            responsive_plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Dot ball percentage
            fig = px.bar(
                filtered_stats.nlargest(10, 'dot_ball_percentage'),
                x='bowler',
                y='dot_ball_percentage',
                title='Top 10 Dot Ball Percentages',
                hover_data=['bowling_economy', 'is_wicket']
            )
            fig.update_layout(xaxis_tickangle=-45)
            responsive_plotly_chart(fig, use_container_width=True)
            
            # Wickets per match
            fig = px.bar(
                filtered_stats.nlargest(10, 'wickets_per_match'),
                x='bowler',
                y='wickets_per_match',
                title='Top 10 Wickets per Match',
                hover_data=['is_wicket', 'matches_bowling']
            )
            fig.update_layout(xaxis_tickangle=-45)
            responsive_plotly_chart(fig, use_container_width=True)
    
    # Wicket Analysis Tab
    with analysis_tabs[1]:
        st.subheader("Wicket Type Analysis")
        selected_bowler = st.selectbox(
            "Select bowler for wicket analysis",
            options=filtered_stats.nlargest(50, 'is_wicket')['bowler'].tolist(),
            key='wicket_bowler'
        )
        
        # Get wicket types for selected bowler
        bowler_wickets = wicket_types[wicket_types['bowler'] == selected_bowler].melt(
            id_vars=['bowler'],
            var_name='wicket_type',
            value_name='count'
        )
        bowler_wickets = bowler_wickets[bowler_wickets['count'] > 0]
        
        # Plot wicket types
        fig = px.pie(
            bowler_wickets,
            values='count',
            names='wicket_type',
            title=f'Wicket Types - {selected_bowler}'
        )
        responsive_plotly_chart(fig, use_container_width=True)
    
    # Phase Analysis Tab
    with analysis_tabs[2]:
        st.subheader("Performance by Match Phase")
        selected_bowler = st.selectbox(
            "Select bowler for phase-wise analysis",
            options=filtered_stats.nlargest(50, 'is_wicket')['bowler'].tolist(),
            key='phase_bowler'
        )
        
        bowler_phase_stats = phase_stats[phase_stats['bowler'] == selected_bowler]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Economy by phase
            fig = px.bar(
                bowler_phase_stats,
                x='phase',
                y='bowling_economy',
                title=f'Economy Rate by Match Phase - {selected_bowler}',
                color='phase'
            )
            responsive_plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Wickets by phase
            fig = px.bar(
                bowler_phase_stats,
                x='phase',
                y='is_wicket',
                title=f'Wickets by Match Phase - {selected_bowler}',
                color='phase'
            )
            responsive_plotly_chart(fig, use_container_width=True)
        
        # Phase-wise detailed stats
        st.subheader("Detailed Phase-wise Statistics")
        detailed_stats = bowler_phase_stats[['phase', 'overs', 'is_wicket', 'bowling_economy', 'bowling_average', 'bowling_strike_rate']]
        detailed_stats = detailed_stats.sort_values('phase')
        st.dataframe(
            detailed_stats.style.format({
                'overs': '{:.1f}',
                'bowling_economy': '{:.2f}',
                'bowling_average': '{:.2f}',
                'bowling_strike_rate': '{:.2f}'
            }),
            use_container_width=True
        )

def calculate_allrounder_stats(deliveries_df=None):
    """Calculate comprehensive all-rounder statistics using pre-computed data when available."""
    # Try to load pre-computed data first
    try:
        data = load_player_analysis_data()
        return data['allrounder_stats']
    except (FileNotFoundError, KeyError):
        # Fall back to calculating from scratch if pre-computed data is not available
        if deliveries_df is None:
            raise ValueError("Deliveries DataFrame is required when pre-computed data is not available")
            
        # Get batting and bowling stats
        batting_stats = calculate_batting_stats(deliveries_df)
        bowling_stats = calculate_bowling_stats(deliveries_df)
        
        # Identify players who both bat and bowl
        allrounders = pd.merge(
            batting_stats,
            bowling_stats,
            left_on='batter',
            right_on='bowler',
            how='inner',
            suffixes=('_batting', '_bowling')
        )
        
        # Rename columns for clarity
        allrounders = allrounders.rename(columns={
            'batter': 'player',
            'matches_batting': 'batting_matches',
            'matches_bowling': 'bowling_matches',
            'runs': 'batting_runs',
            'total_runs': 'runs_conceded',
            'is_wicket': 'wickets'
        })
        
        # Calculate composite scores
        # Normalize batting and bowling stats to a 0-1 scale
        allrounders['batting_score'] = (
            (allrounders['batting_runs'] / allrounders['batting_runs'].max()) * 0.4 +
            (allrounders['batting_average'] / allrounders['batting_average'].max()) * 0.3 +
            (allrounders['batting_strike_rate'] / allrounders['batting_strike_rate'].max()) * 0.3
        )
        
        allrounders['bowling_score'] = (
            (allrounders['wickets'] / allrounders['wickets'].max()) * 0.4 +
            (1 - allrounders['bowling_economy'] / allrounders['bowling_economy'].max()) * 0.3 +
            (1 - allrounders['bowling_average'] / allrounders['bowling_average'].max()) * 0.3
        )
        
        # Calculate overall all-rounder score
        allrounders['allrounder_score'] = (allrounders['batting_score'] + allrounders['bowling_score']) / 2

def display_allrounder_analysis(deliveries_df=None):
    """Display all-rounder analysis dashboard"""
    st.subheader("All-Rounder Analysis")
    
    # Add custom CSS to fix tab styling
    st.markdown("""
    <style>
    /* Fix for red underline in tabs */
    [data-testid="stTabs"] [role="tab"][aria-selected="true"]::before,
    [data-testid="stTabs"] [role="tab"][aria-selected="true"]::after,
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] > div::before,
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] > div::after,
    .st-emotion-cache-1inwz65,
    .st-emotion-cache-1y4pk3h,
    .st-emotion-cache-16idsys {
        display: none !important;
        border: none !important;
        border-bottom: none !important;
        background: none !important;
    }
    
    /* Fix for button hover */
    [data-testid="stTabs"] [role="tab"] {
        position: relative !important;
        z-index: 10 !important;
        margin-top: 5px !important;
    }
    
    [data-testid="stTabs"] [role="tab"]:hover {
        z-index: 100 !important;
        transform: translateY(-2px) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Load data
    if deliveries_df is None:
        st.error("No data available for analysis")
        return
    
    # Calculate stats
    allrounder_stats = calculate_allrounder_stats(deliveries_df)
    
    # Filters
    st.markdown("### Filter Players")
    col1, col2 = st.columns(2)
    with col1:
        min_batting_runs = st.slider("Minimum batting runs", 100, 2000, 500)
    with col2:
        min_wickets = st.slider("Minimum wickets", 10, 100, 20)
    
    # Filter data based on user selection
    filtered_stats = allrounder_stats[
        (allrounder_stats['batting_runs'] >= min_batting_runs) & 
        (allrounder_stats['wickets'] >= min_wickets)
    ]
    
    # Create tabs for different analyses
    analysis_tabs = st.tabs([
        "Overall Rankings",
        "Performance Matrix",
        "Detailed Stats"
    ])
    
    # Overall Rankings Tab
    with analysis_tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top all-rounders by composite score
            fig = px.bar(
                filtered_stats.nlargest(10, 'allrounder_score'),
                x='player',
                y='allrounder_score',
                title='Top 10 All-Rounders (Composite Score)',
                hover_data=['batting_runs', 'wickets']
            )
            fig.update_layout(xaxis_tickangle=-45)
            responsive_plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Batting vs Bowling score scatter
            fig = px.scatter(
                filtered_stats,
                x='batting_score',
                y='bowling_score',
                text='player',
                title='Batting vs Bowling Performance',
                hover_data=['batting_runs', 'wickets', 'allrounder_score']
            )
            fig.update_traces(textposition='top center')
            responsive_plotly_chart(fig, use_container_width=True)
    
    # Performance Matrix Tab
    with analysis_tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Runs vs Wickets scatter
            fig = px.scatter(
                filtered_stats,
                x='batting_runs',
                y='wickets',
                text='player',
                title='Runs vs Wickets Matrix',
                hover_data=['batting_average', 'bowling_economy']
            )
            fig.update_traces(textposition='top center')
            responsive_plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Strike Rate vs Economy scatter
            fig = px.scatter(
                filtered_stats,
                x='batting_strike_rate',
                y='bowling_economy',
                text='player',
                title='Strike Rate vs Economy Rate',
                hover_data=['batting_runs', 'wickets']
            )
            fig.update_traces(textposition='top center')
            responsive_plotly_chart(fig, use_container_width=True)
    
    # Detailed Stats Tab
    with analysis_tabs[2]:
        st.subheader("Detailed All-Rounder Statistics")
        
        # Select player for detailed analysis
        selected_player = st.selectbox(
            "Select player for detailed analysis",
            options=filtered_stats.nlargest(50, 'allrounder_score')['player'].tolist()
        )
        
        player_stats = filtered_stats[filtered_stats['player'] == selected_player]
        
        # Check if player_stats is empty
        if player_stats.empty:
            st.warning(f"No data available for {selected_player} with the current filters. Please adjust the filters.")
            return
        
        # Display detailed stats in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Batting Stats")
            batting_metrics = {
                "Runs": int(player_stats['batting_runs'].iloc[0]),
                "Average": f"{player_stats['batting_average'].iloc[0]:.2f}",
                "Strike Rate": f"{player_stats['batting_strike_rate'].iloc[0]:.2f}",
                "Matches": int(player_stats['batting_matches'].iloc[0]),
                "Batting Score": f"{player_stats['batting_score'].iloc[0]:.3f}"
            }
            
            for metric, value in batting_metrics.items():
                st.metric(metric, value)
        
        with col2:
            st.subheader("Bowling Stats")
            bowling_metrics = {
                "Wickets": int(player_stats['wickets'].iloc[0]),
                "Economy": f"{player_stats['bowling_economy'].iloc[0]:.2f}",
                "Average": f"{player_stats['bowling_average'].iloc[0]:.2f}",
                "Matches": int(player_stats['bowling_matches'].iloc[0]),
                "Bowling Score": f"{player_stats['bowling_score'].iloc[0]:.3f}"
            }
            
            for metric, value in bowling_metrics.items():
                st.metric(metric, value)
        
        # Display overall all-rounder score
        st.metric(
            "Overall All-Rounder Score",
            f"{player_stats['allrounder_score'].iloc[0]:.3f}"
        )

def calculate_head_to_head_stats(deliveries_df=None):
    """Calculate head-to-head statistics between batsmen and bowlers using pre-computed data when available."""
    # Try to load pre-computed data first
    try:
        data = load_player_analysis_data()
        return data['h2h_stats']
    except (FileNotFoundError, KeyError):
        # Fall back to calculating from scratch if pre-computed data is not available
        if deliveries_df is None:
            raise ValueError("Deliveries DataFrame is required when pre-computed data is not available")
            
        # Filter out extras and calculate basic stats
        h2h_stats = deliveries_df.groupby(['batter', 'bowler']).agg({
            'batsman_runs': ['sum', 'count'],
            'is_wicket': 'sum',
            'match_id': 'nunique'
        }).reset_index()
        
        # Flatten column names
        h2h_stats.columns = ['batsman', 'bowler', 'runs', 'balls', 'dismissals', 'matches']
        
        # Calculate derived metrics
        h2h_stats['average'] = h2h_stats['runs'] / h2h_stats['dismissals'].replace(0, 1)
        h2h_stats['strike_rate'] = (h2h_stats['runs'] / h2h_stats['balls']) * 100
        h2h_stats['dominance_ratio'] = h2h_stats['strike_rate'] / (h2h_stats['dismissals'].replace(0, 0.5) * 10)
        
        return h2h_stats

def display_head_to_head_analysis(deliveries_df=None):
    """Display head-to-head analysis dashboard"""
    st.subheader("Head-to-Head Analysis")
    
    # Add custom CSS to fix tab styling
    st.markdown("""
    <style>
    /* Fix for red underline in tabs */
    [data-testid="stTabs"] [role="tab"][aria-selected="true"]::before,
    [data-testid="stTabs"] [role="tab"][aria-selected="true"]::after,
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] > div::before,
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] > div::after,
    .st-emotion-cache-1inwz65,
    .st-emotion-cache-1y4pk3h,
    .st-emotion-cache-16idsys {
        display: none !important;
        border: none !important;
        border-bottom: none !important;
        background: none !important;
    }
    
    /* Fix for button hover */
    [data-testid="stTabs"] [role="tab"] {
        position: relative !important;
        z-index: 10 !important;
        margin-top: 5px !important;
    }
    
    [data-testid="stTabs"] [role="tab"]:hover {
        z-index: 100 !important;
        transform: translateY(-2px) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Load data
    if deliveries_df is None:
        st.error("No data available for analysis")
        return
    
    # Calculate stats
    h2h_stats = calculate_head_to_head_stats(deliveries_df)
    
    # Filters
    st.markdown("### Filter Players")
    col1, col2 = st.columns(2)
    with col1:
        min_balls = st.slider("Minimum balls faced", 6, 100, 12)  # At least 2 overs
    with col2:
        min_runs = st.slider("Minimum runs scored", 0, 5000, 200)
    
    # Filter data based on user selection
    filtered_stats = h2h_stats[
        (h2h_stats['balls'] >= min_balls) & 
        (h2h_stats['runs'] >= min_runs)
    ]
    
    # Create tabs for different analyses
    analysis_tabs = st.tabs([
        "Batsman vs Bowler",
        "Top Matchups",
        "Matchup Matrix"
    ])
    
    # Batsman vs Bowler Tab
    with analysis_tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Select batsman
            selected_batsman = st.selectbox(
                "Select batsman",
                options=sorted(filtered_stats['batsman'].unique())
            )
        
        with col2:
            # Select bowler
            # Filter bowlers who have faced the selected batsman
            bowlers_vs_batsman = filtered_stats[filtered_stats['batsman'] == selected_batsman]['bowler'].unique()
            if len(bowlers_vs_batsman) == 0:
                st.warning(f"No bowlers found for {selected_batsman} with the current filters.")
                return
                
            selected_bowler = st.selectbox(
                "Select bowler",
                options=sorted(bowlers_vs_batsman)
            )
        
        # Get specific matchup stats
        matchup_stats = filtered_stats[
            (filtered_stats['batsman'] == selected_batsman) & 
            (filtered_stats['bowler'] == selected_bowler)
        ]
        
        if not matchup_stats.empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Runs / Balls", f"{matchup_stats['runs'].iloc[0]} / {matchup_stats['balls'].iloc[0]}")
                st.metric("Strike Rate", f"{matchup_stats['strike_rate'].iloc[0]:.2f}")
            
            with col2:
                st.metric("Dismissals", matchup_stats['dismissals'].iloc[0])
                st.metric("Average", f"{matchup_stats['average'].iloc[0]:.2f}")
            
            with col3:
                st.metric("Matches", matchup_stats['matches'].iloc[0])
                st.metric("Dominance Ratio", f"{matchup_stats['dominance_ratio'].iloc[0]:.2f}")
        else:
            st.info(f"No data available for the matchup between {selected_batsman} and {selected_bowler}")
    
    # Top Matchups Tab
    with analysis_tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Most dominant batting performances
            top_batting = filtered_stats.nlargest(10, 'dominance_ratio')
            if not top_batting.empty:
                fig = px.bar(
                    top_batting,
                    x='batsman',
                    y='dominance_ratio',
                    color='bowler',
                    title='Top 10 Dominant Batting Performances',
                    hover_data=['runs', 'balls', 'dismissals']
                )
                fig.update_layout(xaxis_tickangle=-45)
                responsive_plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for dominant batting performances")
        
        with col2:
            # Most dominant bowling performances
            top_bowling = filtered_stats.nsmallest(10, 'strike_rate')
            if not top_bowling.empty:
                fig = px.bar(
                    top_bowling,
                    x='bowler',
                    y='strike_rate',
                    color='batsman',
                    title='Top 10 Dominant Bowling Performances',
                    hover_data=['runs', 'balls', 'dismissals']
                )
                fig.update_layout(xaxis_tickangle=-45)
                responsive_plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for dominant bowling performances")
    
    # Matchup Matrix Tab
    with analysis_tabs[2]:
        st.subheader("Matchup Matrix Analysis")
        
        # Select players for matrix
        selected_batsmen = st.multiselect(
            "Select batsmen",
            options=sorted(filtered_stats['batsman'].unique()),
            default=sorted(filtered_stats['batsman'].unique())[:min(5, len(filtered_stats['batsman'].unique()))]
        )
        
        selected_bowlers = st.multiselect(
            "Select bowlers",
            options=sorted(filtered_stats['bowler'].unique()),
            default=sorted(filtered_stats['bowler'].unique())[:min(5, len(filtered_stats['bowler'].unique()))]
        )
        
        if selected_batsmen and selected_bowlers:
            # Filter data for selected players
            matrix_data = filtered_stats[
                (filtered_stats['batsman'].isin(selected_batsmen)) & 
                (filtered_stats['bowler'].isin(selected_bowlers))
            ]
            
            if not matrix_data.empty:
                # Create matrix for strike rate
                strike_rate_matrix = matrix_data.pivot(
                    index='batsman',
                    columns='bowler',
                    values='strike_rate'
                ).fillna(0)
                
                # Create heatmap
                fig = px.imshow(
                    strike_rate_matrix,
                    title='Strike Rate Matrix',
                    labels=dict(x="Bowler", y="Batsman", color="Strike Rate"),
                    aspect="auto"
                )
                responsive_plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for the selected players")
        else:
            st.info("Please select at least one batsman and one bowler") 