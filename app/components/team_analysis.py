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
            yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
            xaxis=dict(gridcolor='rgba(128,128,128,0.1)')
        )
        st.plotly_chart(fig, use_container_width=True)
        
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
            yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
            xaxis=dict(gridcolor='rgba(128,128,128,0.1)')
        )
        st.plotly_chart(fig, use_container_width=True)

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
        default=teams[:5]  # Default to 5 most successful teams
    )
    
    if not selected_teams:
        st.warning("Please select at least one team to view the Net Run Rate analysis.")
        return
    
    # Full-width container for first chart
    st.markdown("### Average Net Run Rate by Season")
    
    # Filter data for selected teams
    filtered_avg_nrr = avg_nrr_by_season[avg_nrr_by_season['team1'].isin(selected_teams)]
    
    # Use a consistent color palette for the teams
    neon_colors = ['#00ff88', '#ff0088', '#00ffff', '#ff00ff', '#ffff00', '#ff8800', '#88ff00', '#0088ff']
    
    # Create team-to-color mapping for consistency across charts
    team_colors = {team: neon_colors[i % len(neon_colors)] for i, team in enumerate(teams)}
    
    # Average NRR by Season for each team (line chart instead of bar)
    fig = px.line(
        filtered_avg_nrr,
        x='season',
        y='nrr',
        color='team1',
        markers=True,
        template='plotly_dark',
        color_discrete_map=team_colors
    )
    
    # Add horizontal line at NRR = 0
    fig.add_shape(
        type="line",
        x0=filtered_avg_nrr['season'].min(),
        y0=0,
        x1=filtered_avg_nrr['season'].max(),
        y1=0,
        line=dict(color="rgba(255, 255, 255, 0.5)", width=1, dash="dash")
    )
    
    fig.update_layout(
        height=500,  # Taller chart
        xaxis_title="Season",
        yaxis_title="Net Run Rate",
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=80, l=80, r=40),
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.1)',
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.5)'
        ),
        xaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    # Update hover template to show more information
    fig.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>Season: %{x}<br>NRR: %{y:.3f}"
    )
    
    responsive_plotly_chart(fig, use_container_width=True)
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Distribution of NRR for each team (full width)
    st.markdown("### Net Run Rate Distribution")
    
    # Filter data for selected teams
    filtered_nrr = nrr_df[nrr_df['team1'].isin(selected_teams)]
    
    fig2 = go.Figure()
    
    # Add box plot for each selected team
    for team in selected_teams:
        team_nrr = filtered_nrr[filtered_nrr['team1'] == team]['nrr']
        
        # Calculate some statistics
        mean_nrr = team_nrr.mean()
        median_nrr = team_nrr.median()
        
        fig2.add_trace(go.Box(
            y=team_nrr,
            name=team,
            boxpoints='outliers',
            marker_color=team_colors[team],
            line=dict(color=team_colors[team]),
            customdata=[[team, mean_nrr, median_nrr] for _ in range(len(team_nrr))],
            hovertemplate="<b>%{customdata[0]}</b><br>Match NRR: %{y:.3f}<br>Mean NRR: %{customdata[1]:.3f}<br>Median NRR: %{customdata[2]:.3f}"
        ))
    
    # Add horizontal line at NRR = 0
    fig2.add_shape(
        type="line",
        x0=-0.5,
        y0=0,
        x1=len(selected_teams) - 0.5,
        y1=0,
        line=dict(color="rgba(255, 255, 255, 0.5)", width=1, dash="dash")
    )
    
    fig2.update_layout(
        height=500,  # Taller chart
        yaxis_title="Net Run Rate",
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
        margin=dict(t=40, b=80, l=80, r=40),
        boxmode='group',
        showlegend=False
    )
    
    responsive_plotly_chart(fig2, use_container_width=True)
    
    # Add some interpretative text about NRR
    st.markdown("""
    <div style="background-color: rgba(30, 30, 60, 0.7); padding: 15px; border-radius: 10px; margin-top: 20px; border: 1px solid rgba(80, 80, 255, 0.3);">
        <h4 style="margin-top: 0;">Understanding the Charts</h4>
        <ul>
            <li><b>Line Chart:</b> Shows how each team's average NRR has changed over different IPL seasons. Teams consistently above 0 tend to be more dominant.</li>
            <li><b>Box Plot:</b> Shows the distribution of NRR across all matches for each team:
                <ul>
                    <li>The box represents the middle 50% of a team's NRR values</li>
                    <li>The line inside the box is the median NRR</li>
                    <li>Points outside the whiskers are outlier performances (exceptionally good or bad)</li>
                </ul>
            </li>
        </ul>
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
        xaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=60, b=80)
    )
    # Use responsive chart rendering
    responsive_plotly_chart(fig, use_container_width=True)

def plot_head_to_head_analysis(matches_df=None, deliveries_df=None):
    """Plot comprehensive head-to-head analysis using pre-computed data."""
    st.subheader("Head-to-Head Win Matrix")
    
    # Load pre-computed data
    data = load_team_analysis_data()
    win_matrix = data['head_to_head_matrix']
    
    # Create heatmap using Plotly
    fig = px.imshow(
        win_matrix,
        text_auto=True,
        aspect='auto',
        color_continuous_scale=[[0, '#000000'], [1, '#00ff88']],  # Black to neon green
        title='Head-to-Head Win Matrix',
        template='plotly_dark'
    )
    fig.update_layout(
        margin=dict(t=60, b=80),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    # Use responsive chart rendering
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
        st.metric("Avg. First Innings Score", f"{avg_first_innings:.1f}")
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
                xaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
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
            st.plotly_chart(fig, use_container_width=True)
    
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
            st.plotly_chart(fig, use_container_width=True)
        
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
                st.plotly_chart(fig, use_container_width=True)
    
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
            st.plotly_chart(fig, use_container_width=True)
        
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
            st.plotly_chart(fig, use_container_width=True)
    
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
                xaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
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
            st.plotly_chart(fig, use_container_width=True)

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
            st.plotly_chart(fig, use_container_width=True)
        
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
            st.plotly_chart(fig, use_container_width=True)
    
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
            st.plotly_chart(fig, use_container_width=True)
        
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
            st.plotly_chart(fig, use_container_width=True)
    
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
            st.plotly_chart(fig, use_container_width=True)
        
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
            
            fig = px.bar(
                net_rr_df,
                x='Phase',
                y='Net Run Rate',
                title=f'Net Run Rate by Phase - {selected_team}',
                text=net_rr_df['Net Run Rate'].round(2),
                template='plotly_dark',
                color_discrete_sequence=['#00ff88']
            )
            fig.update_layout(
                height=400,
                yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                xaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
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
            st.plotly_chart(fig, use_container_width=True)
        
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
            st.plotly_chart(fig, use_container_width=True)

def display_team_analysis(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame):
    """Main function to display team analysis."""
    st.header("Team Analysis")
    
    # Create tabs for different analyses
    tabs = st.tabs([
        "Overall Performance",
        "Net Run Rate Analysis",
        "Playoff Performance",
        "Head-to-Head Analysis",
        "Venue Performance",
        "Phase Analysis"
    ])
    
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