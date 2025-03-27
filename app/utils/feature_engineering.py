import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Setup logging
log_dir = Path("app/logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"feature_engineering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class FeatureEngineering:
    def __init__(self):
        """Initialize the feature engineering class."""
        logger.info("Initializing FeatureEngineering")
        
    def process_player_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process player statistics and create features.
        
        Args:
            df (pd.DataFrame): Raw player statistics dataframe
            
        Returns:
            pd.DataFrame: Processed dataframe with engineered features
        """
        try:
            logger.info("Starting player stats processing")
            logger.info(f"Input dataframe shape: {df.shape}")
            
            processed_df = df.copy()
            
            # Calculate moving averages for key metrics
            metrics = ['runs', 'wickets', 'catches', 'stumpings', 'run_outs']
            windows = [3, 5, 10]
            
            for metric in metrics:
                if metric in df.columns:
                    for window in windows:
                        try:
                            col_name = f"{metric}_ma_{window}"
                            processed_df[col_name] = df.groupby('player_id')[metric].transform(
                                lambda x: x.rolling(window=window, min_periods=1).mean()
                            )
                            logger.debug(f"Created moving average feature: {col_name}")
                        except Exception as e:
                            logger.error(f"Error creating moving average for {metric}, window {window}: {str(e)}")
            
            # Calculate recent form (last 5 matches)
            try:
                processed_df['recent_form'] = df.groupby('player_id')['runs'].transform(
                    lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
                )
                logger.debug("Created recent form feature")
            except Exception as e:
                logger.error(f"Error calculating recent form: {str(e)}")
            
            # Add venue-specific statistics
            try:
                venue_stats = df.groupby(['player_id', 'venue']).agg({
                    'runs': 'mean',
                    'wickets': 'mean'
                }).reset_index()
                
                processed_df = processed_df.merge(
                    venue_stats,
                    on=['player_id', 'venue'],
                    suffixes=('', '_venue_avg')
                )
                logger.debug("Added venue-specific statistics")
            except Exception as e:
                logger.error(f"Error calculating venue statistics: {str(e)}")
            
            logger.info(f"Completed feature engineering. Output shape: {processed_df.shape}")
            return processed_df
            
        except Exception as e:
            logger.error(f"Error in process_player_stats: {str(e)}")
            raise
            
    def create_team_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create team-level features.
        
        Args:
            df (pd.DataFrame): Player statistics dataframe
            
        Returns:
            pd.DataFrame: Dataframe with team features
        """
        try:
            logger.info("Starting team features creation")
            
            team_features = df.copy()
            
            # Calculate team performance metrics
            team_metrics = df.groupby(['match_id', 'team']).agg({
                'runs': 'sum',
                'wickets': 'sum',
                'catches': 'sum'
            }).reset_index()
            
            # Calculate team form (last 5 matches)
            team_form = team_metrics.groupby('team')['runs'].transform(
                lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
            )
            
            team_features = team_features.merge(
                team_metrics,
                on=['match_id', 'team'],
                suffixes=('', '_team_total')
            )
            
            logger.info("Completed team features creation")
            return team_features
            
        except Exception as e:
            logger.error(f"Error in create_team_features: {str(e)}")
            raise
            
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with handled missing values
        """
        try:
            logger.info("Starting missing value handling")
            logger.info(f"Missing values before handling: {df.isnull().sum().sum()}")
            
            # Fill numeric columns with 0 or mean
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    if col.endswith(('_ma_', 'recent_form')):
                        df[col].fillna(0, inplace=True)
                    else:
                        df[col].fillna(df[col].mean(), inplace=True)
                    logger.debug(f"Filled missing values in column: {col}")
            
            # Fill categorical columns with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].mode()[0], inplace=True)
                    logger.debug(f"Filled missing values in column: {col}")
            
            logger.info(f"Missing values after handling: {df.isnull().sum().sum()}")
            return df
            
        except Exception as e:
            logger.error(f"Error in handle_missing_values: {str(e)}")
            raise

    def validate_features(self, df: pd.DataFrame) -> bool:
        """
        Validate the engineered features.
        
        Args:
            df (pd.DataFrame): Processed dataframe
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            logger.info("Starting feature validation")
            
            # Check for infinite values
            inf_check = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
            if inf_check > 0:
                logger.error(f"Found {inf_check} infinite values in the dataset")
                return False
            
            # Check for missing values
            missing_check = df.isnull().sum().sum()
            if missing_check > 0:
                logger.error(f"Found {missing_check} missing values in the dataset")
                return False
            
            # Check for expected columns
            required_columns = ['player_id', 'match_id', 'team', 'runs', 'wickets']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            logger.info("Feature validation passed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in validate_features: {str(e)}")
            return False 