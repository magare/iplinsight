import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
from utils.chart_utils import responsive_plotly_chart, update_chart_for_responsive_layout

def load_team_analysis_data():
    """Load pre-computed data for team analysis section."""
    base_path = Path(__file__).resolve().parent.parent / "data"
    
    # Load pre-computed datasets
    team_stats = pd.read_parquet(base_path / "team_stats.parquet")
    batting_first_stats = pd.read_parquet(base_path / "batting_first_stats.parquet")
    nrr_data = pd.read_parquet(base_path / "nrr_data.parquet")
    avg_nrr_by_season = pd.read_parquet(base_path / "avg_nrr_by_season.parquet")
    playoff_stats = pd.read_parquet(base_path / "playoff_stats.parquet")
    
    # Special handling for the head-to-head matrix which has teams as index
    head_to_head_matrix = pd.read_parquet(base_path / "head_to_head_matrix.parquet")
    
    venue_team_stats = pd.read_parquet(base_path / "venue_team_stats.parquet")
    team_phase_stats = pd.read_parquet(base_path / "team_phase_stats.parquet")
    
    return {
        'team_stats': team_stats,
        'batting_first_stats': batting_first_stats,
        'nrr_data': nrr_data,
        'avg_nrr_by_season': avg_nrr_by_season,
        'playoff_stats': playoff_stats,
        'head_to_head_matrix': head_to_head_matrix,
        'venue_team_stats': venue_team_stats,
        'team_phase_stats': team_phase_stats
    }

def calculate_team_metrics(matches_df=None, deliveries_df=None) -> Dict:
    """Get pre-computed team metrics."""
    data = load_team_analysis_data()
    metrics = {'team_stats': data['team_stats']}
    return metrics

def plot_overall_performance(matches_df=None, deliveries_df=None):
    """Plot overall team performance metrics using pre-computed data."""
    st.subheader("Overall Team Performance")
    
    # Load pre-computed data
    data = load_team_analysis_data()
    team_stats = data['team_stats']
    batting_first_stats = data['batting_first_stats']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Calculate win percentage for the hover text
        team_stats['Win_Percentage'] = (team_stats['Wins'] / team_stats['Matches'] * 100).round(1)
        
        # Win/Loss Ratio Plot
        fig = px.bar(
            team_stats.sort_values('Win_Loss_Ratio', ascending=False),
            x='Team',
            y='Win_Loss_Ratio',
            title='Team Win/Loss Ratio',
            text=team_stats['Win_Loss_Ratio'].round(2).apply(lambda x: f"{x:.2f}x"),  # Format as 1.50x
            template='plotly_dark',
            color_discrete_sequence=['#00ff88'],  # Neon green
            hover_data={
                'Win_Percentage': True,
                'Wins': True, 
                'Losses': True,
                'Win_Loss_Ratio': False  # Hide the raw ratio in the hover
            }
        )
        
        # Add hover template for better information
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>Win/Loss Ratio: %{text}<br>Win Rate: %{customdata[0]}%<br>Wins: %{customdata[1]}<br>Losses: %{customdata[2]}"
        )
        
        fig.update_layout(
            height=400,
            xaxis_title="Team",
            yaxis_title="Win/Loss Ratio",
            xaxis_tickangle=-45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=60, b=80),
            yaxis=dict(
                gridcolor='rgba(128,128,128,0.1)',
                tickmode='linear',  # Use linear tick mode to show all ticks
                dtick=0.25,         # Set tick interval to 0.25
                automargin=True     # Ensure labels don't get cut off
            ),
            xaxis=dict(
                gridcolor='rgba(128,128,128,0.1)',
                tickmode='array',   # Use array tick mode to show all team names
                tickvals=list(range(len(team_stats))),  # Ensure a tick for each team 
                ticktext=team_stats.sort_values('Win_Loss_Ratio', ascending=False)['Team'],
                automargin=True     # Ensure labels don't get cut off
            )
        )
        responsive_plotly_chart(fig, use_container_width=True)
        
        # Add explanation text below the chart
        st.markdown("""
        <div style="font-size: 0.9rem; margin-top: -15px; margin-bottom: 15px; opacity: 0.9;">
            <p><b>Win/Loss Ratio</b>: Values represent how many matches a team wins for each loss (e.g., 2.00x means 2 wins per loss, or a 67% win rate).</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Batting First vs Chasing Success
        # Create a melted dataframe for plotting
        batting_first_melted = pd.melt(
            batting_first_stats,
            id_vars=['Team'],
            value_vars=['Batting_First_Win_Pct', 'Chasing_Win_Pct'],
            var_name='Scenario',
            value_name='Win Percentage'
        )
        
        # Rename values for display
        batting_first_melted['Scenario'] = batting_first_melted['Scenario'].replace({
            'Batting_First_Win_Pct': 'Batting First Win %',
            'Chasing_Win_Pct': 'Chasing Win %'
        })
        
        fig = px.bar(
            batting_first_melted,
            x='Team',
            y='Win Percentage',
            color='Scenario',
            title='Win Percentage: Batting First vs Chasing',
            barmode='group',
            template='plotly_dark',
            color_discrete_sequence=['#00ff88', '#ff0088']  # Neon green, pink
        )
        
        fig.update_layout(
            height=400,
            xaxis_title="Team",
            yaxis_title="Win Percentage",
            xaxis_tickangle=-45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=60, b=80),
            yaxis=dict(
                gridcolor='rgba(128,128,128,0.1)',
                tickmode='linear',  # Use linear tick mode to show all ticks
                dtick=10,           # Set tick interval to 10%
                range=[0, 100],     # Ensure consistent y-axis range
                automargin=True     # Ensure labels don't get cut off
            ),
            xaxis=dict(
                gridcolor='rgba(128,128,128,0.1)',
                tickmode='array',   # Use array tick mode to show all team names
                tickvals=list(range(len(batting_first_melted['Team'].unique()))),  # Ensure a tick for each team
                ticktext=batting_first_melted['Team'].unique(),
                automargin=True     # Ensure labels don't get cut off
            )
        )
        responsive_plotly_chart(fig, use_container_width=True)

def calculate_nrr(innings1_score: int, innings1_overs: float, innings2_score: int, innings2_overs: float) -> float:
    """Calculate Net Run Rate for a match."""
    if innings1_overs == 0 or innings2_overs == 0:
        return 0
    
    team1_rr = innings1_score / innings1_overs
    team2_rr = innings2_score / innings2_overs
    return team1_rr - team2_rr

def plot_nrr_analysis(matches_df=None, deliveries_df=None):
    """Plot Net Run Rate analysis using pre-computed data."""
    st.subheader("Net Run Rate Analysis")
    
    # Add explanation of what NRR is
    st.markdown("""
    <div style="background-color: rgba(30, 30, 60, 0.7); padding: 15px; border-radius: 10px; margin-bottom: 20px; border: 1px solid rgba(80, 80, 255, 0.3);">
        <p><b>What is Net Run Rate (NRR)?</b> Net Run Rate is a cricket statistic that measures a team's scoring rate compared to their opponent's. 
        A positive NRR means a team scores runs faster than they concede them, while a negative NRR indicates the opposite.</p>
        <p style="margin-bottom: 0">NRR = (Runs scored ÷ Overs faced) - (Runs conceded ÷ Overs bowled)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load pre-computed data
    data = load_team_analysis_data()
    nrr_df = data['nrr_data']
    avg_nrr_by_season = data['avg_nrr_by_season']
    
    # Create a team selector to filter data
    teams = sorted(nrr_df['team1'].unique())
    selected_teams = st.multiselect(
        "Select teams to compare (max 5 recommended):", 
        teams,
        default=teams[:5]  # Default to 5 teams
    )
    
    if not selected_teams:
        st.warning("Please select at least one team to view the Net Run Rate analysis.")
        return
    
    # Define team_colors for use in all charts
    # Use a consistent color palette with fixed hex colors (instead of relying on plotly names)
    distinct_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"
    ]
    team_colors = {team: distinct_colors[i % len(distinct_colors)] for i, team in enumerate(teams)}
    
    # Full-width container for first chart
    st.markdown("### Win Percentage by Teams")
    
    # Use the original team stats dataframe from the data
    team_stats_df = data.get('team_stats', pd.DataFrame())
    
    if not team_stats_df.empty:
        # Check if we have the right columns for this analysis
        if all(col in team_stats_df.columns for col in ['Team', 'Matches', 'Wins', 'Losses']):
            # Filter for selected teams
            filtered_team_stats = team_stats_df[team_stats_df['Team'].isin(selected_teams)]
            
            if filtered_team_stats.empty:
                st.warning("No data available for the selected teams.")
            else:
                # Calculate win percentage
                filtered_team_stats['Win_Percentage'] = (filtered_team_stats['Wins'] / filtered_team_stats['Matches'] * 100).round(1)
                
                # Sort by win percentage (descending)
                filtered_team_stats = filtered_team_stats.sort_values('Win_Percentage', ascending=False)
                
                # Create chart for win percentage
                fig = go.Figure()
                
                # Add main bar chart for win percentage
                fig.add_trace(go.Bar(
                    x=filtered_team_stats['Team'],
                    y=filtered_team_stats['Win_Percentage'],
                    name='Win Percentage',
                    marker_color=[team_colors.get(team, "#1f77b4") for team in filtered_team_stats['Team']],
                    text=filtered_team_stats['Win_Percentage'].round(1).astype(str) + '%',
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Win Rate: %{y:.1f}%<br>Matches: %{customdata[0]}<br>Wins: %{customdata[1]}<br>Losses: %{customdata[2]}<extra></extra>',
                    customdata=filtered_team_stats[['Matches', 'Wins', 'Losses']].values
                ))
                
                # Update layout
                fig.update_layout(
                    title={
                        'text': 'Team Win Percentage Comparison',
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 18}
                    },
                    height=500,
                    xaxis_title="Team",
                    yaxis_title="Win Percentage (%)",
                    yaxis=dict(
                        gridcolor='rgba(128,128,128,0.1)',
                        range=[0, 100]  # Fix y-axis from 0 to 100%
                    ),
                    template='plotly_dark',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=60, b=80, l=80, r=40),
                    hovermode='closest'
                )
                
                # Add a reference line at 50% win rate
                fig.add_shape(
                    type="line",
                    x0=-0.5,
                    y0=50,
                    x1=len(filtered_team_stats) - 0.5,
                    y1=50,
                    line=dict(color="rgba(255, 255, 255, 0.5)", width=1, dash="dash")
                )
                
                # Add annotation to explain the 50% line
                fig.add_annotation(
                    x=len(filtered_team_stats) - 1,
                    y=52,
                    text="50% Win Rate",
                    showarrow=False,
                    font=dict(size=10, color="white"),
                    align="right",
                    xanchor="right",
                    yanchor="bottom"
                )
                
                # Use try-except to handle any rendering errors
                try:
                    responsive_plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error rendering chart: {str(e)}")
                    st.write("Please try selecting different teams or refresh the page.")
                
                # Add a data table below the chart for more detailed information
                st.markdown("#### Detailed Team Performance Stats")
                
                # Prepare a more readable dataframe for display
                display_df = filtered_team_stats[['Team', 'Matches', 'Wins', 'Losses', 'Win_Percentage']].copy()
                display_df.columns = ['Team', 'Matches Played', 'Wins', 'Losses', 'Win Percentage (%)']
                
                # Display the table
                st.dataframe(display_df.set_index('Team'), use_container_width=True)
        else:
            st.warning("Required data columns for Win Percentage analysis are not available.")
    else:
        st.warning("No team statistics data available. Please select at least one team.")
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Second chart - NRR Performance
    st.markdown("### Net Run Rate Performance")
    
    # Filter data for selected teams
    filtered_nrr = nrr_df[nrr_df['team1'].isin(selected_teams)]
    
    # Create a more understandable visualization - Violin plot with points overlaid
    fig2 = go.Figure()
    
    # Add a horizontal line at NRR = 0
    fig2.add_shape(
        type="line",
        x0=-0.5,
        y0=0,
        x1=len(selected_teams) - 0.5,
        y1=0,
        line=dict(color="rgba(255, 255, 255, 0.5)", width=1, dash="dash")
    )
    
    # Add annotation to explain NRR values
    fig2.add_annotation(
        x=len(selected_teams)/2,
        y=max(filtered_nrr['nrr'].max(), 2) * 0.9,
        text="Higher NRR indicates stronger performance",
        showarrow=False,
        font=dict(size=12, color="white"),
        align="center",
        xanchor="center",
        yanchor="top",
        bgcolor="rgba(0,0,0,0.5)",
        borderpad=4
    )
    
    # Add violin plots for each team
    for i, team in enumerate(selected_teams):
        team_nrr = filtered_nrr[filtered_nrr['team1'] == team]['nrr']
        
        # Calculate statistics
        mean_nrr = team_nrr.mean()
        median_nrr = team_nrr.median()
        positive_pct = (team_nrr > 0).mean() * 100
        
        # Get RGB values safely for the fillcolor
        color = team_colors[team]
        # Get hex color components or use a default rgba if there's an issue
        try:
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            fillcolor = f"rgba({r},{g},{b},0.5)"
        except (ValueError, IndexError):
            # Fallback to a safe default with 50% opacity
            fillcolor = f"{color}80"  # 80 in hex = 50% opacity
        
        # Add violin plot
        fig2.add_trace(go.Violin(
            x=[team] * len(team_nrr),
            y=team_nrr,
            name=team,
            box_visible=True,
            meanline_visible=True,
            line_color=team_colors[team],
            fillcolor=fillcolor,
            points="all",
            pointpos=0,
            jitter=0.3,
            marker=dict(
                color=team_colors[team],
                size=4,
                opacity=0.6
            ),
            hoverinfo="skip"
        ))
        
        # Add mean indicator and label
        fig2.add_trace(go.Scatter(
            x=[team],
            y=[mean_nrr],
            mode="markers+text",
            marker=dict(
                symbol="diamond",
                size=12,
                color=team_colors[team],
                line=dict(color="white", width=1),
            ),
            text=["Mean"],
            textposition="top center",
            name=f"{team} Mean",
            customdata=[[team, mean_nrr, median_nrr, positive_pct]],
            hovertemplate="<b>%{customdata[0]}</b><br>Mean NRR: %{customdata[1]:.3f}<br>Median NRR: %{customdata[2]:.3f}<br>Positive NRR: %{customdata[3]:.1f}%<extra></extra>"
        ))
    
    fig2.update_layout(
        title={
            'text': 'Distribution of Match Net Run Rates',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        height=500,
        xaxis_title="Team",
        yaxis_title="Net Run Rate (NRR)",
        template='plotly_dark',
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.1)',
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.5)'
        ),
        xaxis=dict(
            tickangle=-45,
            gridcolor='rgba(128,128,128,0.1)'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=60, b=80, l=80, r=40),
        showlegend=False
    )
    
    responsive_plotly_chart(fig2, use_container_width=True)
    
    # Add an explanation for how to interpret the violin plots
    st.markdown("""
    <div style="background-color: rgba(30, 30, 60, 0.7); padding: 10px; border-radius: 10px; margin-top: 15px; border: 1px solid rgba(80, 80, 255, 0.3);">
        <p><b>How to read this chart:</b> 
        <ul>
            <li>Each "violin" shows the distribution of match NRRs for a team</li>
            <li>Wider sections indicate more matches with that NRR value</li>
            <li>The diamond marker shows the team's mean NRR</li>
            <li>The box inside shows the median and interquartile range</li>
            <li>Each dot represents an individual match</li>
        </ul>
        </p>
    </div>
    """, unsafe_allow_html=True)

def plot_playoff_analysis(matches_df=None):
    """Plot playoff performance analysis using pre-computed data."""
    st.subheader("Playoff Performance")
    
    # Load pre-computed data
    data = load_team_analysis_data()
    playoff_df = data['playoff_stats']
    
    # Rename columns for display
    playoff_df = playoff_df.rename(columns={
        'Playoff_Appearances': 'Playoff Appearances',
        'Final_Appearances': 'Final Appearances',
        'Championships': 'Championships'
    })
    
    # Create grouped bar chart
    playoff_melted = pd.melt(
        playoff_df,
        id_vars=['Team'],
        value_vars=['Playoff Appearances', 'Final Appearances', 'Championships'],
        var_name='Stage',
        value_name='Count'
    )
    
    fig = px.bar(
        playoff_melted,
        x='Team',
        y='Count',
        color='Stage',
        title='Playoff Performance by Team',
        barmode='group',
        template='plotly_dark',
        color_discrete_sequence=['#00ff88', '#ff0088', '#00ffff']  # Neon green, pink, cyan
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="Team",
        yaxis_title="Number of Appearances/Wins",
        xaxis_tickangle=-45,
        yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.1)',
            tickmode='array',  # Use array tick mode to show all team names
            tickvals=list(range(len(playoff_df['Team'].unique()))),  # Ensure a tick for each team
            ticktext=playoff_df['Team'].unique(),
            automargin=True  # Ensure labels don't get cut off
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=60, b=80)
    )
    responsive_plotly_chart(fig, use_container_width=True)

def plot_head_to_head_analysis(matches_df=None, deliveries_df=None):
    """Plot comprehensive head-to-head analysis using pre-computed data."""
    st.subheader("Head-to-Head Win Matrix")
    
    # Add explanation for non-technical users
    st.markdown("""
    <div style="background-color: rgba(0,255,136,0.1); padding: 10px; border-radius: 5px; margin-bottom: 15px;">
        <p><strong>How to read this chart:</strong> This matrix shows the number of matches won by teams listed on the left (rows) against teams listed at the bottom (columns).</p>
        <p>For example, if you see a "2" where row "RR" meets column "KKR", it means Rajasthan Royals (RR) has won 2 matches against Kolkata Knight Riders (KKR).</p>
        <p>Darker green indicates more wins. Hover over any cell for details.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load pre-computed data
    data = load_team_analysis_data()
    win_matrix = data['head_to_head_matrix']
    
    # Create heatmap using Plotly
    fig = px.imshow(
        win_matrix,
        text_auto=True,
        aspect='auto',
        color_continuous_scale=[[0, '#000000'], [0.01, '#001a09'], [1, '#00ff88']],  # Black to neon green with better gradient
        labels=dict(x="Opponent Team", y="Team", color="Wins"),
        title='Head-to-Head Win Matrix',
        template='plotly_dark'
    )
    
    # Improve hover information
    hovertemplate = (
        "<b>%{y}</b> has won <b>%{z}</b> matches<br>" +
        "against <b>%{x}</b><extra></extra>"
    )
    fig.update_traces(hovertemplate=hovertemplate)
    
    # Add annotations to explain the axes
    fig.add_annotation(
        x=0.5,
        y=-0.15,
        xref="paper",
        yref="paper",
        text="Teams as Opponents (Lost Against)",
        showarrow=False,
        font=dict(size=12, color="#00ff88")
    )
    
    fig.add_annotation(
        x=-0.15,
        y=0.5,
        xref="paper",
        yref="paper",
        text="Teams (Won)",
        showarrow=False,
        textangle=-90,
        font=dict(size=12, color="#00ff88")
    )
    
    fig.update_layout(
        margin=dict(t=60, b=100, l=100, r=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        coloraxis_colorbar=dict(
            title="Number of Wins",
            tickvals=[0, 1, 2],
            ticktext=["0 wins", "1 win", "2+ wins"],
        )
    )
    responsive_plotly_chart(fig, use_container_width=True)

def plot_venue_performance(matches_df=None, deliveries_df=None):
    """Plot comprehensive venue performance analysis using both pre-computed data and on-the-fly calculations."""
    st.subheader("Venue Performance Analysis")
    
    # Load pre-computed data
    data = load_team_analysis_data()
    venue_team_stats = data['venue_team_stats']
    
    # We still need the original dataframes for some venue-specific calculations
    if matches_df is None or deliveries_df is None:
        st.error("This analysis requires the original match and delivery data")
        return
    
    # Get unique venues
    venues = sorted(matches_df['venue'].unique())
    selected_venue = st.selectbox("Select Venue", venues, key="venue_performance_venue")
    
    # Filter matches for selected venue
    venue_matches = matches_df[matches_df['venue'] == selected_venue]
    
    # Display basic venue statistics
    total_matches = len(venue_matches)
    avg_first_innings = deliveries_df[
        deliveries_df['match_id'].isin(venue_matches['match_id']) & 
        (deliveries_df['inning'] == 1)
    ].groupby('match_id')['total_runs'].sum().mean()
    
    chasing_wins = len(venue_matches[
        (venue_matches['win_by_wickets'] > 0)
    ])
    defending_wins = len(venue_matches[
        (venue_matches['win_by_runs'] > 0)
    ])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Matches", f"{total_matches:,}")
    with col2:
        try:
            avg_first_innings_value = float(avg_first_innings)
            st.metric("Avg. First Innings Score", f"{avg_first_innings_value:.1f}")
        except (TypeError, ValueError):
            st.metric("Avg. First Innings Score", "N/A")
    with col3:
        st.metric("Chasing Wins", f"{chasing_wins:,}")
    with col4:
        st.metric("Defending Wins", f"{defending_wins:,}")
    
    # Create tabs for detailed analysis
    venue_tabs = st.tabs([
        "Team Performance",
        "Batting Analysis",
        "Bowling Analysis",
        "Season Trends"
    ])
    
    # Team Performance Tab
    with venue_tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Team win percentages at this venue (use pre-computed data)
            venue_team_data = venue_team_stats[venue_team_stats['Venue'] == selected_venue]
            
            # Rename Win_Rate column for better readability
            if 'Win_Rate' in venue_team_data.columns:
                venue_team_data = venue_team_data.rename(columns={'Win_Rate': 'Win Rate (%)'})
            
            fig = px.bar(
                venue_team_data.sort_values('Win Rate (%)', ascending=False),
                x='Team',
                y='Win Rate (%)',
                title=f'Team Win Rates at {selected_venue}',
                text=venue_team_data['Win Rate (%)'].round(1),
                template='plotly_dark',
                color_discrete_sequence=['#00ff88']
            )
            fig.update_layout(
                height=400,
                xaxis_tickangle=-45,
                yaxis=dict(
                    title="Win Rate (%)",
                    gridcolor='rgba(128,128,128,0.1)',
                    range=[0, 100]
                ),
                xaxis=dict(
                    gridcolor='rgba(128,128,128,0.1)',
                    tickmode='array',  # Use array tick mode to show all teams
                    tickvals=list(range(len(venue_team_data))),  # Ensure a tick for each team
                    ticktext=venue_team_data.sort_values('Win Rate (%)', ascending=False)['Team'],
                    automargin=True  # Ensure labels don't get cut off
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            responsive_plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Toss impact at this venue (calculate on-the-fly as it's venue-specific)
            toss_stats = []
            for decision in ['bat', 'field']:
                matches = venue_matches[venue_matches['toss_decision'] == decision]
                toss_winner_wins = len(matches[matches['toss_winner'] == matches['winner']])
                total = len(matches)
                
                if total > 0:
                    win_rate = (toss_winner_wins / total) * 100
                    toss_stats.append({
                        'Decision': decision.capitalize(),
                        'Win Rate': win_rate,
                        'Total Matches': total
                    })
            
            toss_df = pd.DataFrame(toss_stats)
            
            fig = px.bar(
                toss_df,
                x='Decision',
                y='Win Rate',
                title=f'Toss Impact at {selected_venue}',
                text=toss_df['Win Rate'].round(1),
                template='plotly_dark',
                color_discrete_sequence=['#ff0088']
            )
            fig.update_layout(
                height=400,
                yaxis=dict(
                    title="Win Rate After Winning Toss (%)",
                    gridcolor='rgba(128,128,128,0.1)',
                    range=[0, 100]
                ),
                xaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            responsive_plotly_chart(fig, use_container_width=True)
    
    # The remaining tabs will continue to use on-the-fly calculations as they're venue-specific
    # Batting Analysis Tab
    with venue_tabs[1]:
        venue_deliveries = deliveries_df[
            deliveries_df['match_id'].isin(venue_matches['match_id'])
        ]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Average runs by over
            runs_by_over = venue_deliveries.groupby(['inning', 'over'])['total_runs'].mean().reset_index()
            
            fig = px.line(
                runs_by_over,
                x='over',
                y='total_runs',
                color='inning',
                title=f'Average Runs per Over at {selected_venue}',
                template='plotly_dark',
                labels={'total_runs': 'Average Runs', 'over': 'Over', 'inning': 'Innings'}
            )
            fig.update_layout(
                height=400,
                yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                xaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            responsive_plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Boundary percentage
            boundaries = venue_deliveries[venue_deliveries['batsman_runs'].isin([4, 6])]
            boundary_stats = boundaries.groupby('batsman_runs').size()
            
            if not boundary_stats.empty:
                fig = px.pie(
                    values=boundary_stats.values,
                    names=['Fours', 'Sixes'],
                    title=f'Boundary Distribution at {selected_venue}',
                    template='plotly_dark',
                    color_discrete_sequence=['#00ff88', '#ff0088']
                )
                fig.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                responsive_plotly_chart(fig, use_container_width=True)
    
    # Bowling Analysis Tab
    with venue_tabs[2]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Wicket types distribution
            wickets = venue_deliveries[venue_deliveries['is_wicket'] == 1]
            wicket_types = wickets['wicket_kind'].value_counts()
            
            fig = px.pie(
                values=wicket_types.values,
                names=wicket_types.index,
                title=f'Wicket Types at {selected_venue}',
                template='plotly_dark'
            )
            fig.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            responsive_plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Wickets by over
            wickets_by_over = wickets.groupby(['inning', 'over']).size().reset_index(name='count')
            
            fig = px.line(
                wickets_by_over,
                x='over',
                y='count',
                color='inning',
                title=f'Wickets by Over at {selected_venue}',
                template='plotly_dark',
                labels={'count': 'Number of Wickets', 'over': 'Over', 'inning': 'Innings'}
            )
            fig.update_layout(
                height=400,
                yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                xaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            responsive_plotly_chart(fig, use_container_width=True)
    
    # Season Trends Tab
    with venue_tabs[3]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Average scores by season
            season_scores = venue_deliveries.groupby(['match_id', 'inning'])['total_runs'].sum().reset_index()
            season_scores = pd.merge(
                season_scores,
                venue_matches[['match_id', 'season']],
                on='match_id'
            )
            avg_scores = season_scores.groupby(['season', 'inning'])['total_runs'].mean().reset_index()
            
            fig = px.line(
                avg_scores,
                x='season',
                y='total_runs',
                color='inning',
                title=f'Average Scores by Season at {selected_venue}',
                template='plotly_dark',
                labels={'total_runs': 'Average Score', 'season': 'Season', 'inning': 'Innings'}
            )
            fig.update_layout(
                height=400,
                xaxis_tickangle=-45,
                yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                xaxis=dict(
                    gridcolor='rgba(128,128,128,0.1)',
                    tickmode='array',    # Use array tick mode to show all seasons
                    tickvals=avg_scores['season'].unique(),  # Ensure a tick for each season
                    ticktext=avg_scores['season'].unique(),
                    automargin=True      # Ensure labels don't get cut off
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            responsive_plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Matches per season
            matches_per_season = venue_matches.groupby('season').size().reset_index(name='matches')
            
            fig = px.bar(
                matches_per_season,
                x='season',
                y='matches',
                title=f'Matches per Season at {selected_venue}',
                template='plotly_dark',
                color_discrete_sequence=['#00ff88']
            )
            fig.update_layout(
                height=400,
                xaxis_tickangle=-45,
                yaxis=dict(
                    title="Number of Matches",
                    gridcolor='rgba(128,128,128,0.1)'
                ),
                xaxis=dict(
                    title="Season",
                    gridcolor='rgba(128,128,128,0.1)'
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            responsive_plotly_chart(fig, use_container_width=True)

def plot_phase_analysis(matches_df=None, deliveries_df=None):
    """Plot phase-wise analysis of team performance using pre-computed data."""
    st.subheader("Phase Analysis")
    
    # Load pre-computed data
    data = load_team_analysis_data()
    phase_stats = data['team_phase_stats']
    
    # Define match phases
    phases = {
        'Powerplay': (0, 6),
        'Middle Overs': (7, 15),
        'Death Overs': (16, 20)
    }
    
    # Allow user to select team for analysis
    teams = sorted(phase_stats['Team'].unique())
    selected_team = st.selectbox("Select Team", teams, key="phase_analysis_team")
    
    # Filter phase stats for selected team
    team_phase_stats = phase_stats[phase_stats['Team'] == selected_team]
    
    # Create tabs for different phase analyses
    phase_tabs = st.tabs([
        "Batting Phases",
        "Bowling Phases",
        "Phase Comparison",
        "Historical Trends"
    ])
    
    # Batting Phases Tab
    with phase_tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Run rate in different phases
            batting_run_rate = team_phase_stats[['Phase', 'Batting_Run_Rate']]
            
            fig = px.bar(
                batting_run_rate,
                x='Phase',
                y='Batting_Run_Rate',
                title=f'Run Rate by Phase - {selected_team} (Batting)',
                text=batting_run_rate['Batting_Run_Rate'].round(2),
                template='plotly_dark',
                color_discrete_sequence=['#00ff88']
            )
            fig.update_layout(
                height=400,
                yaxis=dict(
                    title="Run Rate",
                    gridcolor='rgba(128,128,128,0.1)'
                ),
                xaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            responsive_plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Boundary percentage in different phases
            boundary_pct = team_phase_stats[['Phase', 'Batting_Boundary_Pct']]
            
            fig = px.bar(
                boundary_pct,
                x='Phase',
                y='Batting_Boundary_Pct',
                title=f'Boundary Percentage by Phase - {selected_team} (Batting)',
                text=boundary_pct['Batting_Boundary_Pct'].round(1),
                template='plotly_dark',
                color_discrete_sequence=['#ff0088']
            )
            fig.update_layout(
                height=400,
                yaxis=dict(
                    title="Boundary %",
                    gridcolor='rgba(128,128,128,0.1)',
                    range=[0, 100]
                ),
                xaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            responsive_plotly_chart(fig, use_container_width=True)
    
    # Bowling Phases Tab
    with phase_tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Economy rate in different phases
            economy_rate = team_phase_stats[['Phase', 'Bowling_Economy']]
            
            fig = px.bar(
                economy_rate,
                x='Phase',
                y='Bowling_Economy',
                title=f'Economy Rate by Phase - {selected_team} (Bowling)',
                text=economy_rate['Bowling_Economy'].round(2),
                template='plotly_dark',
                color_discrete_sequence=['#00ff88']
            )
            fig.update_layout(
                height=400,
                yaxis=dict(
                    title="Economy Rate",
                    gridcolor='rgba(128,128,128,0.1)'
                ),
                xaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            responsive_plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Wicket rate in different phases
            wicket_rate = team_phase_stats[['Phase', 'Bowling_Wicket_Rate']]
            
            fig = px.bar(
                wicket_rate,
                x='Phase',
                y='Bowling_Wicket_Rate',
                title=f'Wickets per Over by Phase - {selected_team} (Bowling)',
                text=wicket_rate['Bowling_Wicket_Rate'].round(2),
                template='plotly_dark',
                color_discrete_sequence=['#ff0088']
            )
            fig.update_layout(
                height=400,
                yaxis=dict(
                    title="Wickets per Over",
                    gridcolor='rgba(128,128,128,0.1)'
                ),
                xaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            responsive_plotly_chart(fig, use_container_width=True)
    
    # Phase Comparison Tab
    with phase_tabs[2]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Compare batting and bowling performance in each phase
            phase_comparison = []
            for phase in team_phase_stats['Phase'].unique():
                phase_data = team_phase_stats[team_phase_stats['Phase'] == phase]
                
                batting_rr = phase_data['Batting_Run_Rate'].values[0]
                bowling_rr = phase_data['Bowling_Economy'].values[0]
                
                phase_comparison.append({
                    'Phase': phase,
                    'Role': 'Batting',
                    'Run Rate': batting_rr
                })
                phase_comparison.append({
                    'Phase': phase,
                    'Role': 'Bowling',
                    'Run Rate': bowling_rr
                })
            
            comparison_df = pd.DataFrame(phase_comparison)
            
            fig = px.bar(
                comparison_df,
                x='Phase',
                y='Run Rate',
                color='Role',
                title=f'Phase Performance Comparison - {selected_team}',
                barmode='group',
                template='plotly_dark',
                color_discrete_sequence=['#00ff88', '#ff0088']
            )
            fig.update_layout(
                height=400,
                yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                xaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            responsive_plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Net Run Rate by Phase
            net_rr = []
            for phase in team_phase_stats['Phase'].unique():
                phase_data = team_phase_stats[team_phase_stats['Phase'] == phase]
                
                batting_rr = phase_data['Batting_Run_Rate'].values[0]
                bowling_rr = phase_data['Bowling_Economy'].values[0]
                
                net_rr.append({
                    'Phase': phase,
                    'Net Run Rate': batting_rr - bowling_rr
                })
            
            net_rr_df = pd.DataFrame(net_rr)
            
            # Add a column for color based on positive/negative values
            net_rr_df['Color'] = net_rr_df['Net Run Rate'].apply(
                lambda x: '#2ecc71' if x >= 0 else '#e74c3c'
            )
            
            # Sort phases in a meaningful order
            phase_order = ['Powerplay', 'Middle Overs', 'Death Overs']
            net_rr_df['Phase'] = pd.Categorical(
                net_rr_df['Phase'], 
                categories=phase_order, 
                ordered=True
            )
            net_rr_df = net_rr_df.sort_values('Phase')
            
            fig = go.Figure()
            
            # Add horizontal line at y=0
            fig.add_shape(
                type="line",
                x0=-0.5,
                y0=0,
                x1=len(net_rr_df) - 0.5,
                y1=0,
                line=dict(color="rgba(255, 255, 255, 0.5)", width=1, dash="dash")
            )
            
            # Add bars
            fig.add_trace(go.Bar(
                x=net_rr_df['Phase'],
                y=net_rr_df['Net Run Rate'],
                text=net_rr_df['Net Run Rate'].round(2),
                textposition='outside',
                marker_color=net_rr_df['Color'],
                hovertemplate='<b>%{x}</b><br>Net Run Rate: %{y:.2f}<extra></extra>'
            ))
            
            # Add annotations to explain what positive and negative values mean
            if len(net_rr_df) > 0:
                max_y = max(abs(net_rr_df['Net Run Rate'].max()), abs(net_rr_df['Net Run Rate'].min())) * 1.1
                
                fig.add_annotation(
                    x=net_rr_df['Phase'].iloc[-1],
                    y=max_y/2,
                    text="↑ Batting faster than bowling",
                    showarrow=False,
                    font=dict(size=10, color="#2ecc71"),
                    xanchor="right",
                    yanchor="bottom"
                )
                
                fig.add_annotation(
                    x=net_rr_df['Phase'].iloc[-1],
                    y=-max_y/2,
                    text="↓ Bowling better than batting",
                    showarrow=False,
                    font=dict(size=10, color="#e74c3c"),
                    xanchor="right",
                    yanchor="top"
                )
            
            fig.update_layout(
                title={
                    'text': f'Net Run Rate by Phase - {selected_team}',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16}
                },
                template='plotly_dark',
                xaxis_title="Match Phase",
                yaxis_title="Net Run Rate (Runs per Over)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=60, b=40, l=60, r=60),
                height=400,
                yaxis=dict(
                    gridcolor='rgba(128,128,128,0.1)',
                    zeroline=False
                ),
                xaxis=dict(gridcolor='rgba(128,128,128,0.1)')
            )
            
            responsive_plotly_chart(fig, use_container_width=True)
            
            # Add explanation of what the chart means
            st.markdown("""
            <div style="background-color: rgba(30, 30, 60, 0.7); padding: 10px; border-radius: 10px; margin-top: 10px; border: 1px solid rgba(80, 80, 255, 0.3); font-size: 0.9em;">
                <p style="margin-bottom: 5px;"><b>Understanding Net Run Rate by Phase:</b></p>
                <ul style="margin-top: 0; padding-left: 20px;">
                    <li><span style="color: #2ecc71;">Positive values</span>: Team scores faster than they concede runs</li>
                    <li><span style="color: #e74c3c;">Negative values</span>: Team concedes runs faster than they score</li>
                    <li><b>Powerplay</b>: Overs 1-6</li>
                    <li><b>Middle Overs</b>: Overs 7-15</li>
                    <li><b>Death Overs</b>: Overs 16-20</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # For the Historical Trends Tab, we still need the original dataframes
    # Historical Trends Tab
    with phase_tabs[3]:
        # Get season-wise phase performance
        if matches_df is None or deliveries_df is None:
            st.error("This analysis requires the original match and delivery data")
            return
            
        seasons = sorted(matches_df['season'].unique())
        phase_trends = []
        
        for season in seasons:
            season_deliveries = deliveries_df[
                deliveries_df['match_id'].isin(
                    matches_df[matches_df['season'] == season]['match_id']
                )
            ]
            
            for phase, (start, end) in phases.items():
                # Batting
                batting_phase = season_deliveries[
                    (season_deliveries['batting_team'] == selected_team) &
                    (season_deliveries['over'] >= start) &
                    (season_deliveries['over'] <= end)
                ]
                
                # Bowling
                bowling_phase = season_deliveries[
                    (season_deliveries['bowling_team'] == selected_team) &
                    (season_deliveries['over'] >= start) &
                    (season_deliveries['over'] <= end)
                ]
                
                batting_rr = (batting_phase['total_runs'].sum() / (len(batting_phase)/6)) if len(batting_phase) > 0 else 0
                bowling_rr = (bowling_phase['total_runs'].sum() / (len(bowling_phase)/6)) if len(bowling_phase) > 0 else 0
                
                phase_trends.append({
                    'Season': season,
                    'Phase': phase,
                    'Batting RR': batting_rr,
                    'Bowling Economy': bowling_rr
                })
        
        trends_df = pd.DataFrame(phase_trends)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Batting Run Rate Trends
            fig = px.line(
                trends_df,
                x='Season',
                y='Batting RR',
                color='Phase',
                title=f'Batting Run Rate Trends - {selected_team}',
                template='plotly_dark',
                markers=True
            )
            fig.update_layout(
                height=400,
                xaxis_tickangle=-45,
                yaxis=dict(
                    title="Run Rate",
                    gridcolor='rgba(128,128,128,0.1)'
                ),
                xaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            responsive_plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bowling Economy Trends
            fig = px.line(
                trends_df,
                x='Season',
                y='Bowling Economy',
                color='Phase',
                title=f'Bowling Economy Trends - {selected_team}',
                template='plotly_dark',
                markers=True
            )
            fig.update_layout(
                height=400,
                xaxis_tickangle=-45,
                yaxis=dict(
                    title="Economy Rate",
                    gridcolor='rgba(128,128,128,0.1)'
                ),
                xaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            responsive_plotly_chart(fig, use_container_width=True)

def display_team_analysis(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame):
    """Main function to display team analysis."""
    st.header("Team Analysis")
    
    # Get device type to determine layout
    device_type = st.session_state.get('device_type', 'mobile')
    
    # Define analysis options
    analysis_options = [
        "Overall Performance",
        "Net Run Rate Analysis",
        "Playoff Performance",
        "Head-to-Head Analysis",
        "Venue Performance",
        "Phase Analysis"
    ]
    
    if device_type == 'mobile':
        # Use a selectbox for mobile view
        selected_analysis = st.selectbox("Select Analysis", analysis_options)
        
        # Display the selected analysis
        if selected_analysis == "Overall Performance":
            plot_overall_performance(matches_df, deliveries_df)
        elif selected_analysis == "Net Run Rate Analysis":
            plot_nrr_analysis(matches_df, deliveries_df)
        elif selected_analysis == "Playoff Performance":
            plot_playoff_analysis(matches_df)
        elif selected_analysis == "Head-to-Head Analysis":
            plot_head_to_head_analysis(matches_df, deliveries_df)
        elif selected_analysis == "Venue Performance":
            plot_venue_performance(matches_df, deliveries_df)
        elif selected_analysis == "Phase Analysis":
            plot_phase_analysis(matches_df, deliveries_df)
    else:
        # Use tabs for desktop view
        tabs = st.tabs(analysis_options)
        
        # Overall Performance Tab
        with tabs[0]:
            plot_overall_performance(matches_df, deliveries_df)
        
        # Net Run Rate Analysis Tab
        with tabs[1]:
            plot_nrr_analysis(matches_df, deliveries_df)
        
        # Playoff Performance Tab
        with tabs[2]:
            plot_playoff_analysis(matches_df)
        
        # Head-to-Head Analysis Tab
        with tabs[3]:
            plot_head_to_head_analysis(matches_df, deliveries_df)
        
        # Venue Performance Tab
        with tabs[4]:
            plot_venue_performance(matches_df, deliveries_df)
        
        # Phase Analysis Tab
        with tabs[5]:
            plot_phase_analysis(matches_df, deliveries_df) 