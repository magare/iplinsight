import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

# Setup logging
log_dir = Path("app/logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"model_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, output_dir: str = "app/models"):
        """
        Initialize the model trainer.
        
        Args:
            output_dir (str): Directory to save models and visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.models = {}
        logger.info(f"Initialized ModelTrainer with output directory: {output_dir}")
        
    def prepare_time_series_validation(self, X: pd.DataFrame, y: pd.Series) -> TimeSeriesSplit:
        """
        Prepare time series cross-validation splits.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            
        Returns:
            TimeSeriesSplit: Time series cross-validation splitter
        """
        try:
            tscv = TimeSeriesSplit(n_splits=10)
            logger.info("Created TimeSeriesSplit with 10 folds")
            return tscv
        except Exception as e:
            logger.error(f"Error in prepare_time_series_validation: {str(e)}")
            raise
            
    def train_model(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'rf') -> Dict:
        """
        Train a model with time series cross-validation.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            model_type (str): Type of model ('rf' for Random Forest, 'lr' for Linear Regression)
            
        Returns:
            Dict: Dictionary containing model and evaluation metrics
        """
        try:
            logger.info(f"Starting model training for {model_type}")
            logger.info(f"Feature matrix shape: {X.shape}")
            
            # Initialize model
            if model_type == 'rf':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = LinearRegression()
            
            # Prepare cross-validation
            tscv = self.prepare_time_series_validation(X, y)
            
            # Initialize metrics storage
            metrics = {
                'mae': [],
                'rmse': [],
                'r2': []
            }
            
            # Perform cross-validation
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_val)
                
                # Calculate metrics
                metrics['mae'].append(mean_absolute_error(y_val, y_pred))
                metrics['rmse'].append(np.sqrt(mean_squared_error(y_val, y_pred)))
                metrics['r2'].append(r2_score(y_val, y_pred))
                
                logger.info(f"Fold {fold + 1} - MAE: {metrics['mae'][-1]:.4f}, "
                          f"RMSE: {metrics['rmse'][-1]:.4f}, R²: {metrics['r2'][-1]:.4f}")
            
            # Calculate average metrics
            avg_metrics = {
                'avg_mae': np.mean(metrics['mae']),
                'avg_rmse': np.mean(metrics['rmse']),
                'avg_r2': np.mean(metrics['r2']),
                'std_mae': np.std(metrics['mae']),
                'std_rmse': np.std(metrics['rmse']),
                'std_r2': np.std(metrics['r2'])
            }
            
            logger.info(f"Average metrics - MAE: {avg_metrics['avg_mae']:.4f} ± {avg_metrics['std_mae']:.4f}, "
                      f"RMSE: {avg_metrics['avg_rmse']:.4f} ± {avg_metrics['std_rmse']:.4f}, "
                      f"R²: {avg_metrics['avg_r2']:.4f} ± {avg_metrics['std_r2']:.4f}")
            
            # Save model and metrics
            model_path = self.output_dir / f"{model_type}_model.joblib"
            metrics_path = self.output_dir / f"{model_type}_metrics.json"
            
            joblib.dump(model, model_path)
            pd.Series(avg_metrics).to_json(metrics_path)
            
            self.models[model_type] = {
                'model': model,
                'metrics': avg_metrics
            }
            
            # Generate feature importance visualization
            self.visualize_feature_importance(model, X.columns, model_type)
            
            return self.models[model_type]
            
        except Exception as e:
            logger.error(f"Error in train_model: {str(e)}")
            raise
            
    def visualize_feature_importance(self, model, feature_names: List[str], model_type: str):
        """
        Visualize feature importance or coefficients.
        
        Args:
            model: Trained model
            feature_names (List[str]): List of feature names
            model_type (str): Type of model ('rf' or 'lr')
        """
        try:
            plt.figure(figsize=(12, 6))
            
            if model_type == 'rf':
                importances = model.feature_importances_
                title = 'Random Forest Feature Importance'
            else:
                importances = model.coef_
                title = 'Linear Regression Coefficients'
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=True)
            
            # Plot feature importance
            plt.barh(range(len(importance_df)), importance_df['importance'])
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.xlabel('Importance' if model_type == 'rf' else 'Coefficient')
            plt.title(title)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(self.output_dir / f"{model_type}_feature_importance.png")
            plt.close()
            
            logger.info(f"Saved feature importance visualization for {model_type}")
            
        except Exception as e:
            logger.error(f"Error in visualize_feature_importance: {str(e)}")
            raise
            
    def evaluate_model(self, model_type: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate model performance on test set.
        
        Args:
            model_type (str): Type of model to evaluate
            X_test (pd.DataFrame): Test feature matrix
            y_test (pd.Series): Test target variable
            
        Returns:
            Dict: Dictionary containing evaluation metrics
        """
        try:
            logger.info(f"Evaluating {model_type} model on test set")
            
            if model_type not in self.models:
                raise ValueError(f"Model {model_type} not found. Train the model first.")
            
            model = self.models[model_type]['model']
            y_pred = model.predict(X_test)
            
            metrics = {
                'test_mae': mean_absolute_error(y_test, y_pred),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'test_r2': r2_score(y_test, y_pred)
            }
            
            logger.info(f"Test metrics - MAE: {metrics['test_mae']:.4f}, "
                      f"RMSE: {metrics['test_rmse']:.4f}, R²: {metrics['test_r2']:.4f}")
            
            # Save test metrics
            metrics_path = self.output_dir / f"{model_type}_test_metrics.json"
            pd.Series(metrics).to_json(metrics_path)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in evaluate_model: {str(e)}")
            raise 