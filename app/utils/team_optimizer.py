import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from datetime import datetime

# Setup logging
log_dir = Path("app/logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"team_optimizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class TeamOptimizer:
    def __init__(self):
        """Initialize the team optimizer."""
        logger.info("Initializing TeamOptimizer")
        
    def optimize_team(self,
                     players_df: pd.DataFrame,
                     budget: float = 100.0,
                     team_size: int = 11,
                     max_per_team: int = 4,
                     must_include: Optional[List[str]] = None,
                     must_exclude: Optional[List[str]] = None,
                     min_batsmen: int = 3,
                     min_bowlers: int = 3,
                     min_all_rounders: int = 1,
                     min_wk: int = 1) -> Dict:
        """
        Optimize team selection based on predicted performance and constraints.
        
        Args:
            players_df (pd.DataFrame): DataFrame with player information and predictions
            budget (float): Total budget constraint
            team_size (int): Required team size
            max_per_team (int): Maximum players allowed from one team
            must_include (List[str]): List of player IDs that must be included
            must_exclude (List[str]): List of player IDs that must be excluded
            min_batsmen (int): Minimum number of batsmen required
            min_bowlers (int): Minimum number of bowlers required
            min_all_rounders (int): Minimum number of all-rounders required
            min_wk (int): Minimum number of wicket-keepers required
            
        Returns:
            Dict: Dictionary containing selected team and team statistics
        """
        try:
            logger.info("Starting team optimization")
            logger.info(f"Budget: {budget}, Team size: {team_size}")
            
            # Initialize constraints
            must_include = set(must_include) if must_include else set()
            must_exclude = set(must_exclude) if must_exclude else set()
            
            # Validate input data
            self._validate_input_data(players_df, must_include, must_exclude)
            
            # Pre-process player data
            players = self._preprocess_players(players_df)
            
            # Initialize selected team
            selected_team = self._initialize_team(players, must_include)
            
            # Optimize remaining slots
            remaining_slots = team_size - len(selected_team)
            remaining_budget = budget - sum(players.loc[list(selected_team), 'price'])
            
            # Get valid candidates for remaining slots
            candidates = self._get_valid_candidates(
                players,
                selected_team,
                must_exclude,
                max_per_team
            )
            
            # Fill remaining slots using greedy approach with constraints
            selected_team = self._fill_remaining_slots(
                players,
                candidates,
                selected_team,
                remaining_slots,
                remaining_budget,
                min_batsmen,
                min_bowlers,
                min_all_rounders,
                min_wk
            )
            
            # Select captain and vice-captain
            captain, vice_captain = self._select_captain_vice_captain(players, selected_team)
            
            # Prepare team statistics
            team_stats = self._calculate_team_stats(players, selected_team, captain, vice_captain)
            
            logger.info("Team optimization completed successfully")
            return team_stats
            
        except Exception as e:
            logger.error(f"Error in optimize_team: {str(e)}")
            raise
            
    def _validate_input_data(self, players_df: pd.DataFrame,
                            must_include: Set[str],
                            must_exclude: Set[str]):
        """Validate input data and constraints."""
        try:
            required_columns = ['player_id', 'name', 'team', 'role', 'price', 'predicted_points']
            missing_cols = [col for col in required_columns if col not in players_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Check for overlapping must_include and must_exclude
            overlap = must_include.intersection(must_exclude)
            if overlap:
                raise ValueError(f"Players {overlap} appear in both must_include and must_exclude")
                
            logger.info("Input data validation passed")
            
        except Exception as e:
            logger.error(f"Error in _validate_input_data: {str(e)}")
            raise
            
    def _preprocess_players(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess player data for optimization."""
        try:
            players = players_df.copy()
            
            # Calculate value for money (points per unit cost)
            players['value'] = players['predicted_points'] / players['price']
            
            # Sort by value for greedy selection
            players.sort_values('value', ascending=False, inplace=True)
            
            logger.info("Player data preprocessing completed")
            return players
            
        except Exception as e:
            logger.error(f"Error in _preprocess_players: {str(e)}")
            raise
            
    def _initialize_team(self, players: pd.DataFrame, must_include: Set[str]) -> Set[str]:
        """Initialize team with must-include players."""
        try:
            selected_team = must_include.copy()
            logger.info(f"Team initialized with {len(selected_team)} must-include players")
            return selected_team
            
        except Exception as e:
            logger.error(f"Error in _initialize_team: {str(e)}")
            raise
            
    def _get_valid_candidates(self, players: pd.DataFrame,
                            selected_team: Set[str],
                            must_exclude: Set[str],
                            max_per_team: int) -> pd.DataFrame:
        """Get valid candidates for remaining slots."""
        try:
            # Remove already selected and excluded players
            candidates = players[~players.index.isin(selected_team | must_exclude)].copy()
            
            # Check team constraints
            team_counts = players.loc[list(selected_team), 'team'].value_counts()
            for team, count in team_counts.items():
                if count >= max_per_team:
                    candidates = candidates[candidates['team'] != team]
                    
            logger.info(f"Found {len(candidates)} valid candidates")
            return candidates
            
        except Exception as e:
            logger.error(f"Error in _get_valid_candidates: {str(e)}")
            raise
            
    def _fill_remaining_slots(self, players: pd.DataFrame,
                            candidates: pd.DataFrame,
                            selected_team: Set[str],
                            remaining_slots: int,
                            remaining_budget: float,
                            min_batsmen: int,
                            min_bowlers: int,
                            min_all_rounders: int,
                            min_wk: int) -> Set[str]:
        """Fill remaining slots while satisfying all constraints."""
        try:
            # Get current team composition
            roles = players.loc[list(selected_team), 'role'].value_counts()
            
            # Calculate required players for each role
            required = {
                'Batsman': max(0, min_batsmen - roles.get('Batsman', 0)),
                'Bowler': max(0, min_bowlers - roles.get('Bowler', 0)),
                'All-Rounder': max(0, min_all_rounders - roles.get('All-Rounder', 0)),
                'Wicket-Keeper': max(0, min_wk - roles.get('Wicket-Keeper', 0))
            }
            
            # First fill required roles
            for role, count in required.items():
                role_candidates = candidates[candidates['role'] == role]
                role_candidates = role_candidates[role_candidates['price'] <= remaining_budget]
                
                for _ in range(count):
                    if len(role_candidates) > 0:
                        selected_player = role_candidates.index[0]
                        selected_team.add(selected_player)
                        remaining_budget -= players.loc[selected_player, 'price']
                        remaining_slots -= 1
                        
                        # Update candidates
                        candidates = self._get_valid_candidates(
                            players,
                            selected_team,
                            set(),
                            4
                        )
                        role_candidates = candidates[candidates['role'] == role]
                        role_candidates = role_candidates[role_candidates['price'] <= remaining_budget]
                    
            # Fill remaining slots with best available players
            while remaining_slots > 0 and len(candidates) > 0:
                affordable = candidates[candidates['price'] <= remaining_budget]
                if len(affordable) == 0:
                    break
                    
                selected_player = affordable.index[0]
                selected_team.add(selected_player)
                remaining_budget -= players.loc[selected_player, 'price']
                remaining_slots -= 1
                
                # Update candidates
                candidates = self._get_valid_candidates(
                    players,
                    selected_team,
                    set(),
                    4
                )
                
            logger.info(f"Team filled with {len(selected_team)} players")
            return selected_team
            
        except Exception as e:
            logger.error(f"Error in _fill_remaining_slots: {str(e)}")
            raise
            
    def _select_captain_vice_captain(self, players: pd.DataFrame,
                                   selected_team: Set[str]) -> Tuple[str, str]:
        """Select captain and vice-captain based on predicted points."""
        try:
            team_players = players.loc[list(selected_team)].sort_values('predicted_points', ascending=False)
            captain = team_players.index[0]
            vice_captain = team_players.index[1]
            
            logger.info(f"Selected captain: {players.loc[captain, 'name']}")
            logger.info(f"Selected vice-captain: {players.loc[vice_captain, 'name']}")
            
            return captain, vice_captain
            
        except Exception as e:
            logger.error(f"Error in _select_captain_vice_captain: {str(e)}")
            raise
            
    def _calculate_team_stats(self, players: pd.DataFrame,
                            selected_team: Set[str],
                            captain: str,
                            vice_captain: str) -> Dict:
        """Calculate and return team statistics."""
        try:
            team_players = players.loc[list(selected_team)]
            
            stats = {
                'team': {
                    'players': team_players[['name', 'team', 'role', 'price', 'predicted_points']].to_dict('records'),
                    'captain': players.loc[captain, 'name'],
                    'vice_captain': players.loc[vice_captain, 'name'],
                    'total_cost': team_players['price'].sum(),
                    'predicted_points': (
                        team_players['predicted_points'].sum() +
                        players.loc[captain, 'predicted_points'] +
                        0.5 * players.loc[vice_captain, 'predicted_points']
                    ),
                    'composition': team_players['role'].value_counts().to_dict(),
                    'teams_distribution': team_players['team'].value_counts().to_dict()
                }
            }
            
            logger.info("Team statistics calculated successfully")
            return stats
            
        except Exception as e:
            logger.error(f"Error in _calculate_team_stats: {str(e)}")
            raise 