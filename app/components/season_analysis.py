'''\
Module for analyzing season data.
This module loads precomputed season statistics and provides functions to display
these analyses using Streamlit and Plotly.\
'''

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import os
from pathlib import Path
from utils.chart_utils import responsive_plotly_chart, update_chart_for_responsive_layout

# ---------------------------
# Helper Functions
# ---------------------------

def load_precomputed_data(filename):
    """
    Load precomputed data from Parquet files.
    
    Args:
        filename (str): Name of the file to load.
        
    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    data_dir = Path(__file__).resolve().parent.parent / "data"
    
    # Convert legacy filename to parquet if needed
    if filename.endswith('.csv'):
        filename = filename.replace('.csv', '.parquet')
    
    file_path = data_dir / filename
    
    if not os.path.exists(file_path):
        st.error(f"Precomputed data file not found: {filename}")
        return pd.DataFrame()
    
    # All parquet files can be loaded with a single method, no need to handle index separately
    # as parquet preserves the index information
    return pd.read_parquet(file_path)

# ---------------------------
# Season Stats Loading Functions
# ---------------------------

def load_season_stats(season):
    """
    Load precomputed season statistics.
    
    Args:
        season (int): Season year to load data for.
        
    Returns:
        dict: Dictionary containing various season statistics.
    """
    # Load season stats
    season_stats = load_precomputed_data(f'season_{season}_stats.parquet')
    
    # Load standings
    standings = load_precomputed_data(f'season_{season}_standings.parquet')
    
    # Load key matches
    key_matches = load_precomputed_data(f'season_{season}_key_matches.parquet')
    
    # Load batting stats
    batting_stats = load_precomputed_data(f'season_{season}_batting_stats.parquet')
    
    # Load bowling stats
    bowling_stats = load_precomputed_data(f'season_{season}_bowling_stats.parquet')
    
    # Load fielding stats
    fielding_stats = load_precomputed_data(f'season_{season}_fielding_stats.parquet')
    
    # Load all-round stats
    all_round_stats = load_precomputed_data(f'season_{season}_all_round_stats.parquet')
    
    # Load points progression
    points_progression = load_precomputed_data(f'season_{season}_points_progression.parquet')
    
    return {
        'season_stats': season_stats,
        'standings': standings,
        'key_matches': key_matches,
        'batting_stats': batting_stats,
        'bowling_stats': bowling_stats,
        'fielding_stats': fielding_stats,
        'all_round_stats': all_round_stats,
        'points_progression': points_progression
    }

# ---------------------------
# Display Functions
# ---------------------------

def display_season_highlights(matches_df, deliveries_df, season):
    """Display season highlights and key statistics."""
    st.header(f"Season {season} Highlights")
    
    # Load precomputed season statistics
    stats = load_season_stats(season)
    
    # Display basic statistics in columns
    col1, col2, col3 = st.columns(3)
    
    # Extract basic stats from precomputed data
    if not stats['season_stats'].empty:
        season_data = stats['season_stats'].iloc[0]
        
        with col1:
            st.metric("Total Matches", season_data.get('total_matches', 0))
            try:
                avg_score = float(season_data.get('avg_match_score', 0))
                st.metric("Average Match Score", f"{avg_score:.1f}")
            except (TypeError, ValueError):
                st.metric("Average Match Score", "N/A")
        
        with col2:
            st.metric("Total Runs", season_data.get('total_runs', 0))
            st.metric("Total Wickets", season_data.get('total_wickets', 0))
        
        with col3:
            try:
                sixes_per_match = float(season_data.get('sixes_per_match', 0))
                st.metric("Sixes per Match", f"{sixes_per_match:.1f}")
            except (TypeError, ValueError):
                st.metric("Sixes per Match", "N/A")
            
            try:
                fours_per_match = float(season_data.get('fours_per_match', 0))
                st.metric("Fours per Match", f"{fours_per_match:.1f}")
            except (TypeError, ValueError):
                st.metric("Fours per Match", "N/A")
        
        # Display season winner
        if 'winner' in season_data:
            st.success(f"🏆 Season Winner: {season_data['winner']}")
    else:
        st.warning("Season statistics not available")
    
    # Display team standings
    st.subheader("Team Standings")
    if not stats['standings'].empty:
        st.dataframe(
            stats['standings'].style.format({
                'nrr': '{:.3f}',
                'runs': '{:.0f}',
                'wickets': '{:.0f}'
            }),
            use_container_width=True
        )
    else:
        st.warning("Team standings not available")

def display_season_standings(matches_df, deliveries_df, season):
    """Display detailed team standings and points progression."""
    st.subheader("Points Table Progression")
    
    # Load precomputed season statistics
    stats = load_season_stats(season)
    
    # Create cumulative points progression plot
    if not stats['points_progression'].empty:
        points_df = stats['points_progression'].reset_index()
        points_df.columns = ['index'] + list(points_df.columns[1:])
        
        # Melt the DataFrame to get it in the right format for plotting
        points_melt = points_df.melt(
            id_vars='index', 
            var_name='Team', 
            value_name='Points'
        )
        points_melt.rename(columns={'index': 'Match Number'}, inplace=True)
        
        # Create a new line chart with markers instead of area chart
        fig = px.line(
            points_melt,
            x='Match Number',
            y='Points',
            color='Team',
            markers=True,  # Add markers to make the progression clearer
            line_shape='linear',  # Use linear lines for clearer changes
            title='Team Points Progression Over the Season',
            labels={'Points': 'Points', 'Match Number': 'Match Number'},
            template='plotly_dark',
            color_discrete_sequence=['#00ff88', '#ff0088', '#00ffff', '#ff00ff', '#ffff00', '#ff8800', '#88ff00', '#0088ff']  # Neon colors
        )
        
        # Add ranking annotations at selected intervals
        # Calculate rankings at each match number
        match_numbers = sorted(points_melt['Match Number'].unique())
        selected_matches = [match_numbers[0]] + match_numbers[::max(1, len(match_numbers)//5)][-5:]  # Start, then ~5 evenly spaced points
        
        annotations = []
        for match_num in selected_matches:
            # Get points for this match
            match_data = points_melt[points_melt['Match Number'] == match_num]
            # Sort by points descending
            match_data = match_data.sort_values('Points', ascending=False)
            
            # Add rank number next to each team at this match
            for i, (_, row) in enumerate(match_data.iterrows(), 1):
                annotations.append(dict(
                    x=match_num,
                    y=row['Points'],
                    text=f"#{i}",
                    showarrow=False,
                    font=dict(color="white", size=9),
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="white",
                    borderwidth=1,
                    borderpad=2,
                    xanchor='left'
                ))
        
        fig.update_layout(
            xaxis_title='Match Number',
            yaxis_title='Points',
            legend_title='Teams',
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
            xaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
            annotations=annotations,
            height=500  # Make chart taller for better readability
        )
        
        # Add a range slider to zoom into specific parts of the season
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type='linear'
            )
        )
        
        # Use the responsive chart function instead of st.plotly_chart
        responsive_plotly_chart(fig, use_container_width=True)
        
        # Add a radio button to switch between different visualizations
        viz_option = st.radio(
            "Choose visualization type:",
            ["Line Chart", "Bar Chart Race", "Heatmap View"],
            horizontal=True
        )
        
        if viz_option == "Bar Chart Race":
            # Create a bar chart race effect
            # This creates static frames that give the impression of a race
            st.subheader("Points Race Throughout the Season")
            
            # Allow user to select specific matches to view
            step = max(1, len(match_numbers) // 10)  # Divide season into ~10 steps
            selected_match = st.select_slider(
                "Select match number to view standings:",
                options=match_numbers[::step],
                value=match_numbers[-1]  # Default to final match
            )
            
            # Get data for selected match
            match_data = points_melt[points_melt['Match Number'] == selected_match]
            # Sort by points descending
            match_data = match_data.sort_values('Points', ascending=False)
            
            # Create horizontal bar chart
            bar_fig = px.bar(
                match_data,
                y='Team',
                x='Points',
                orientation='h',
                color='Team',
                title=f'Team Standings After Match {selected_match}',
                color_discrete_sequence=['#00ff88', '#ff0088', '#00ffff', '#ff00ff', '#ffff00', '#ff8800', '#88ff00', '#0088ff']
            )
            
            # Add rank numbers
            for i, (_, row) in enumerate(match_data.iterrows(), 1):
                bar_fig.add_annotation(
                    x=0,
                    y=row['Team'],
                    text=f"#{i}",
                    showarrow=False,
                    font=dict(color="white", size=12),
                    xanchor='right',
                    yanchor='middle',
                    xshift=-10
                )
            
            bar_fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(gridcolor='rgba(128,128,128,0.1)', categoryorder='total ascending'),
                xaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                height=400
            )
            
            responsive_plotly_chart(bar_fig, use_container_width=True)
            
        elif viz_option == "Heatmap View":
            # Create a heatmap showing team positions throughout the season
            st.subheader("Team Positions Throughout the Season")
            
            # Create a DataFrame to store team rankings at each match
            rankings = []
            for match_num in match_numbers:
                # Get points for this match
                match_data = points_melt[points_melt['Match Number'] == match_num]
                # Sort by points descending and get team rankings
                match_data = match_data.sort_values('Points', ascending=False)
                match_data['Rank'] = range(1, len(match_data) + 1)
                
                # Store ranking data
                for _, row in match_data.iterrows():
                    rankings.append({
                        'Match Number': match_num,
                        'Team': row['Team'],
                        'Rank': row['Rank']
                    })
            
            rankings_df = pd.DataFrame(rankings)
            
            # Create heatmap
            heat_fig = px.imshow(
                rankings_df.pivot(index='Team', columns='Match Number', values='Rank'),
                labels=dict(x="Match Number", y="Team", color="Rank"),
                x=match_numbers,
                color_continuous_scale=px.colors.sequential.Plasma_r,  # Reversed so 1st is bright
                title="Team Rankings Heatmap (Brighter = Higher Rank)",
                template='plotly_dark'
            )
            
            heat_fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            
            responsive_plotly_chart(heat_fig, use_container_width=True)
    else:
        st.warning("Points progression data not available")
    
    # Display final standings
    st.subheader("Final Standings")
    if not stats['standings'].empty:
        final_standings = stats['standings'].reset_index()
        final_standings.columns = ['Team'] + list(final_standings.columns[1:])
        st.dataframe(final_standings, use_container_width=True)
    else:
        st.warning("Final standings data not available")

def display_top_performers(matches_df, deliveries_df, season):
    """Display detailed analysis of top performers."""
    st.subheader(f"Season {season} Top Performers")
    
    # Load precomputed season statistics
    stats = load_season_stats(season)
    
    # Create tabs for different categories
    tabs = st.tabs([
        "Batting",
        "Bowling",
        "Fielding",
        "All-Round"
    ])
    
    # Batting Tab
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top run scorers
            st.subheader("Top Run Scorers")
            if not stats['batting_stats'].empty:
                top_batters = stats['batting_stats'].nlargest(10, 'runs')
                fig = px.bar(
                    top_batters.reset_index(),
                    x='batter',
                    y='runs',
                    title='Top 10 Run Scorers',
                    hover_data=['average', 'strike_rate'],
                    template='plotly_dark',
                    color_discrete_sequence=['#00ff88']  # Neon green
                )
                fig.update_layout(
                    xaxis_tickangle=-45,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                    xaxis=dict(gridcolor='rgba(128,128,128,0.1)')
                )
                responsive_plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Batting statistics not available")
        
        with col2:
            # Best strike rates (min 100 balls)
            st.subheader("Best Strike Rates")
            if not stats['batting_stats'].empty:
                qualified_batters = stats['batting_stats'][stats['batting_stats']['balls'] >= 100]
                top_sr = qualified_batters.nlargest(10, 'strike_rate')
                fig = px.bar(
                    top_sr.reset_index(),
                    x='batter',
                    y='strike_rate',
                    title='Top 10 Strike Rates (min. 100 balls)',
                    hover_data=['runs', 'balls'],
                    template='plotly_dark',
                    color_discrete_sequence=['#ff0088']  # Neon pink
                )
                fig.update_layout(
                    xaxis_tickangle=-45,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                    xaxis=dict(gridcolor='rgba(128,128,128,0.1)')
                )
                responsive_plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Batting statistics not available")
    
    # Bowling Tab
    with tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top wicket takers
            st.subheader("Top Wicket Takers")
            if not stats['bowling_stats'].empty:
                top_bowlers = stats['bowling_stats'].nlargest(10, 'wickets')
                fig = px.bar(
                    top_bowlers.reset_index(),
                    x='bowler',
                    y='wickets',
                    title='Top 10 Wicket Takers',
                    hover_data=['economy', 'average'],
                    template='plotly_dark',
                    color_discrete_sequence=['#00ffff']  # Neon cyan
                )
                fig.update_layout(
                    xaxis_tickangle=-45,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                    xaxis=dict(gridcolor='rgba(128,128,128,0.1)')
                )
                responsive_plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Bowling statistics not available")
        
        with col2:
            # Best economy rates (min 20 overs)
            st.subheader("Best Economy Rates")
            if not stats['bowling_stats'].empty:
                qualified_bowlers = stats['bowling_stats'][stats['bowling_stats']['overs'] >= 20]
                top_economy = qualified_bowlers.nsmallest(10, 'economy')
                fig = px.bar(
                    top_economy.reset_index(),
                    x='bowler',
                    y='economy',
                    title='Top 10 Economy Rates (min. 20 overs)',
                    hover_data=['wickets', 'overs'],
                    template='plotly_dark',
                    color_discrete_sequence=['#ff00ff']  # Neon magenta
                )
                fig.update_layout(
                    xaxis_tickangle=-45,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                    xaxis=dict(gridcolor='rgba(128,128,128,0.1)')
                )
                responsive_plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Bowling statistics not available")
    
    # Fielding Tab
    with tabs[2]:
        if not stats['fielding_stats'].empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Most catches
                st.subheader("Most Catches")
                top_catchers = stats['fielding_stats'].nlargest(10, 'catches')
                fig = px.bar(
                    top_catchers.reset_index(),
                    x='player',
                    y='catches',
                    title='Top 10 Catchers',
                    template='plotly_dark',
                    color_discrete_sequence=['#ffff00']  # Neon yellow
                )
                fig.update_layout(
                    xaxis_tickangle=-45,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                    xaxis=dict(gridcolor='rgba(128,128,128,0.1)')
                )
                responsive_plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Most run outs
                st.subheader("Most Run Outs")
                top_run_outs = stats['fielding_stats'].nlargest(10, 'run_outs')
                fig = px.bar(
                    top_run_outs.reset_index(),
                    x='player',
                    y='run_outs',
                    title='Top 10 Run Outs',
                    template='plotly_dark',
                    color_discrete_sequence=['#ff8800']  # Neon orange
                )
                fig.update_layout(
                    xaxis_tickangle=-45,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                    xaxis=dict(gridcolor='rgba(128,128,128,0.1)')
                )
                responsive_plotly_chart(fig, use_container_width=True)
        else:
            st.info("Fielding statistics are not available for this season")
    
    # All-Round Tab
    with tabs[3]:
        if not stats['all_round_stats'].empty:
            all_round_df = stats['all_round_stats'].reset_index()
            st.dataframe(
                all_round_df.style.format({
                    'runs': '{:.0f}',
                    'wickets': '{:.0f}',
                    'batting_sr': '{:.2f}',
                    'bowling_economy': '{:.2f}'
                }),
                use_container_width=True
            )
        else:
            st.info("No qualifying all-rounders found for this season")

def display_season_analysis(matches_df, deliveries_df):
    """Display comprehensive season analysis."""
    st.title("Season Analysis")
    
    # Get unique seasons and sort them
    seasons = sorted(matches_df['season'].unique())
    
    # Season selector
    selected_season = st.selectbox(
        "Select Season",
        seasons,
        index=len(seasons)-1  # Default to latest season
    )
    
    # Create tabs for different aspects of season analysis
    tabs = st.tabs([
        "Highlights",
        "Standings",
        "Top Performers",
        "Key Matches"
    ])
    
    # Highlights Tab
    with tabs[0]:
        display_season_highlights(matches_df, deliveries_df, selected_season)
    
    # Standings Tab
    with tabs[1]:
        display_season_standings(matches_df, deliveries_df, selected_season)
    
    # Top Performers Tab
    with tabs[2]:
        display_top_performers(matches_df, deliveries_df, selected_season)
    
    # Key Matches Tab
    with tabs[3]:
        st.subheader("Key Matches")
        # Load precomputed season statistics
        stats = load_season_stats(selected_season)
        
        if not stats['key_matches'].empty:
            for _, match in stats['key_matches'].iterrows():
                st.markdown(f"""
                **{match['type']}**  
                {match['description']}  
                Winner: {match['winner']} ({match['margin']})
                """)
        else:
            st.info("No key matches data available for this season") 