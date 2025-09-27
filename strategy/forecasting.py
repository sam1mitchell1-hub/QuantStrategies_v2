"""
GBT Forecasting Models

Machine learning forecasting models for the trading strategy including:
- LightGBM and CatBoost implementations
- Cross-validation with purged/walk-forward splits
- Probability calibration (Platt scaling, Isotonic regression)
- Model evaluation and monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Try to import ML libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available. Install with: pip install catboost")

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available. Install with: pip install scikit-learn")


@dataclass
class ModelResults:
    """Container for model results."""
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    feature_importance: Optional[pd.DataFrame] = None
    metrics: Optional[Dict[str, float]] = None
    calibration_metrics: Optional[Dict[str, float]] = None


class GBTForecaster:
    """GBT-based forecasting model for options trading strategy."""
    
    def __init__(self, config):
        """Initialize forecaster with configuration."""
        self.config = config
        self.model = None
        self.calibrator = None
        self.feature_names = None
        self.is_trained = False
        
        # Validate model availability
        if config.model.model_type == 'lightgbm' and not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available")
        if config.model.model_type == 'catboost' and not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not available")
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn not available")
    
    def train(self, 
              X: pd.DataFrame, 
              y: pd.Series,
              sample_weight: Optional[pd.Series] = None) -> ModelResults:
        """
        Train the GBT model.
        
        Args:
            X: Feature matrix
            y: Target variable
            sample_weight: Optional sample weights
            
        Returns:
            ModelResults with training metrics
        """
        print(f"Training {self.config.model.model_type} model...")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Prepare data
        X_clean, y_clean, sample_weight_clean = self._prepare_data(X, y, sample_weight)
        
        # Create model
        self.model = self._create_model()
        
        # Train model
        if self.config.model.model_type == 'lightgbm':
            self.model.fit(
                X_clean, y_clean,
                sample_weight=sample_weight_clean,
                eval_set=[(X_clean, y_clean)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
        elif self.config.model.model_type == 'catboost':
            self.model.fit(
                X_clean, y_clean,
                sample_weight=sample_weight_clean,
                eval_set=(X_clean, y_clean),
                early_stopping_rounds=50,
                verbose=False
            )
        
        # Get predictions
        if self.config.model.objective == 'classification':
            predictions = self.model.predict(X_clean)
            probabilities = self.model.predict_proba(X_clean)[:, 1]
        else:
            predictions = self.model.predict(X_clean)
            probabilities = None
        
        # Calibrate probabilities if classification
        if self.config.model.objective == 'classification':
            self.calibrator = self._create_calibrator()
            self.calibrator.fit(X_clean, y_clean)
            calibrated_probs = self.calibrator.predict_proba(X_clean)[:, 1]
        else:
            calibrated_probs = None
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_clean, predictions, probabilities)
        
        # Get feature importance
        feature_importance = self._get_feature_importance()
        
        self.is_trained = True
        
        return ModelResults(
            predictions=predictions,
            probabilities=calibrated_probs,
            feature_importance=feature_importance,
            metrics=metrics
        )
    
    def predict(self, X: pd.DataFrame) -> ModelResults:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            ModelResults with predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare data
        X_clean = self._prepare_features(X)
        
        # Make predictions
        if self.config.model.objective == 'classification':
            predictions = self.model.predict(X_clean)
            probabilities = self.model.predict_proba(X_clean)[:, 1]
            
            # Apply calibration if available
            if self.calibrator is not None:
                probabilities = self.calibrator.predict_proba(X_clean)[:, 1]
        else:
            predictions = self.model.predict(X_clean)
            probabilities = None
        
        return ModelResults(
            predictions=predictions,
            probabilities=probabilities
        )
    
    def cross_validate(self, 
                      X: pd.DataFrame, 
                      y: pd.Series,
                      sample_weight: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Perform cross-validation with purged/walk-forward splits.
        
        Args:
            X: Feature matrix
            y: Target variable
            sample_weight: Optional sample weights
            
        Returns:
            Cross-validation results
        """
        print("Performing cross-validation...")
        
        # Prepare data
        X_clean, y_clean, sample_weight_clean = self._prepare_data(X, y, sample_weight)
        
        # Create time series splits with embargo
        embargo_days = self.config.model.cv_embargo_days
        horizon_days = self.config.market.trading_horizon_days
        
        # Calculate embargo in terms of data points (assuming daily data)
        embargo_points = embargo_days
        horizon_points = horizon_days
        
        # Create purged time series split
        cv_splits = self._create_purged_splits(
            len(X_clean), 
            n_splits=self.config.model.cv_folds,
            embargo=embargo_points,
            horizon=horizon_points
        )
        
        cv_results = {
            'scores': [],
            'predictions': [],
            'probabilities': [],
            'feature_importance': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            print(f"  Fold {fold + 1}/{self.config.model.cv_folds}")
            
            # Split data
            X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
            y_train, y_val = y_clean.iloc[train_idx], y_clean.iloc[val_idx]
            
            if sample_weight_clean is not None:
                sw_train = sample_weight_clean.iloc[train_idx]
                sw_val = sample_weight_clean.iloc[val_idx]
            else:
                sw_train = sw_val = None
            
            # Train model
            model = self._create_model()
            
            if self.config.model.model_type == 'lightgbm':
                model.fit(
                    X_train, y_train,
                    sample_weight=sw_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )
            elif self.config.model.model_type == 'catboost':
                model.fit(
                    X_train, y_train,
                    sample_weight=sw_train,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=50,
                    verbose=False
                )
            
            # Make predictions
            if self.config.model.objective == 'classification':
                val_pred = model.predict(X_val)
                val_probs = model.predict_proba(X_val)[:, 1]
            else:
                val_pred = model.predict(X_val)
                val_probs = None
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_val, val_pred, val_probs)
            cv_results['scores'].append(metrics)
            cv_results['predictions'].extend(val_pred)
            cv_results['probabilities'].extend(val_probs if val_probs is not None else [])
            
            # Store feature importance
            importance = self._get_feature_importance_from_model(model)
            cv_results['feature_importance'].append(importance)
        
        # Calculate average metrics
        avg_metrics = {}
        for metric in cv_results['scores'][0].keys():
            avg_metrics[f'cv_{metric}'] = np.mean([scores[metric] for scores in cv_results['scores']])
            avg_metrics[f'cv_{metric}_std'] = np.std([scores[metric] for scores in cv_results['scores']])
        
        cv_results['average_metrics'] = avg_metrics
        
        return cv_results
    
    def _prepare_data(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
        """Prepare data for training."""
        # Remove rows with NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        
        X_clean = X[mask]
        y_clean = y[mask]
        sample_weight_clean = sample_weight[mask] if sample_weight is not None else None
        
        return X_clean, y_clean, sample_weight_clean
    
    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction."""
        # Ensure same columns as training
        if self.feature_names is not None:
            X_clean = X[self.feature_names]
        else:
            X_clean = X
        
        # Fill NaN values with median
        X_clean = X_clean.fillna(X_clean.median())
        
        return X_clean
    
    def _create_model(self):
        """Create model instance."""
        if self.config.model.model_type == 'lightgbm':
            return lgb.LGBMClassifier(**self.config.model.lightgbm_params) if self.config.model.objective == 'classification' else lgb.LGBMRegressor(**self.config.model.lightgbm_params)
        elif self.config.model.model_type == 'catboost':
            if self.config.model.objective == 'classification':
                return cb.CatBoostClassifier(verbose=False, random_seed=42)
            else:
                return cb.CatBoostRegressor(verbose=False, random_seed=42)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model.model_type}")
    
    def _create_calibrator(self):
        """Create probability calibrator."""
        if self.config.model.calibration_method == 'platt':
            return CalibratedClassifierCV(
                LogisticRegression(),
                method='sigmoid',
                cv=self.config.model.calibration_cv_folds
            )
        elif self.config.model.calibration_method == 'isotonic':
            return CalibratedClassifierCV(
                LogisticRegression(),
                method='isotonic',
                cv=self.config.model.calibration_cv_folds
            )
        else:
            raise ValueError(f"Unsupported calibration method: {self.config.model.calibration_method}")
    
    def _create_purged_splits(self, n_samples: int, n_splits: int, embargo: int, horizon: int):
        """Create purged time series splits."""
        # Calculate split size
        split_size = n_samples // n_splits
        
        splits = []
        for i in range(n_splits):
            # Calculate split boundaries
            start_idx = i * split_size
            end_idx = min((i + 1) * split_size, n_samples)
            
            # Create train and validation indices
            train_end = start_idx + int(split_size * 0.8)  # 80% for training
            val_start = train_end + embargo  # Add embargo
            val_end = end_idx - horizon  # Remove horizon
            
            if val_start < val_end:
                train_idx = list(range(start_idx, train_end))
                val_idx = list(range(val_start, val_end))
                splits.append((train_idx, val_idx))
        
        return splits
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_probs: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {}
        
        if self.config.model.objective == 'classification':
            # Classification metrics
            metrics['accuracy'] = np.mean(y_true == y_pred)
            metrics['hit_rate'] = metrics['accuracy']  # Alias for hit rate
            
            if y_probs is not None:
                metrics['auc'] = roc_auc_score(y_true, y_probs)
                metrics['log_loss'] = log_loss(y_true, y_probs)
                metrics['brier_score'] = brier_score_loss(y_true, y_probs)
                
                # Information Ratio (simplified)
                positive_predictions = y_probs[y_true == 1]
                negative_predictions = y_probs[y_true == 0]
                
                if len(positive_predictions) > 0 and len(negative_predictions) > 0:
                    hit_rate = np.mean(y_true == (y_probs > 0.5))
                    avg_positive_prob = np.mean(positive_predictions)
                    avg_negative_prob = np.mean(negative_predictions)
                    
                    # Simplified IR calculation
                    metrics['information_ratio'] = (hit_rate - 0.5) / np.std(y_probs) if np.std(y_probs) > 0 else 0
        else:
            # Regression metrics
            mse = np.mean((y_true - y_pred) ** 2)
            mae = np.mean(np.abs(y_true - y_pred))
            
            metrics['mse'] = mse
            metrics['rmse'] = np.sqrt(mse)
            metrics['mae'] = mae
            
            # R-squared
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Information Ratio (for returns)
            if np.std(y_pred) > 0:
                metrics['information_ratio'] = np.mean(y_pred) / np.std(y_pred)
            else:
                metrics['information_ratio'] = 0
        
        return metrics
    
    def _get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model."""
        if self.model is None:
            return pd.DataFrame()
        
        return self._get_feature_importance_from_model(self.model)
    
    def _get_feature_importance_from_model(self, model) -> pd.DataFrame:
        """Get feature importance from a specific model."""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance()
        else:
            return pd.DataFrame()
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save_model(self, filepath: str):
        """Save trained model to file."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        import joblib
        
        model_data = {
            'model': self.model,
            'calibrator': self.calibrator,
            'feature_names': self.feature_names,
            'config': self.config
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file."""
        import joblib
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.calibrator = model_data['calibrator']
        self.feature_names = model_data['feature_names']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")
    
    def get_forecast(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get forecast with confidence intervals.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with forecasts and confidence intervals
        """
        results = self.predict(X)
        
        forecast = {
            'predictions': results.predictions,
            'probabilities': results.probabilities
        }
        
        if self.config.model.objective == 'classification':
            # Convert probabilities to expected returns
            # This is a simplified mapping - in practice, you'd use historical conditional returns
            if results.probabilities is not None:
                # Simple mapping: prob > 0.5 -> positive return, prob < 0.5 -> negative return
                # Scale by confidence (distance from 0.5)
                confidence = np.abs(results.probabilities - 0.5) * 2  # Scale to [0, 1]
                expected_returns = (results.probabilities - 0.5) * confidence * 0.02  # Max 2% return
                forecast['expected_returns'] = expected_returns
            else:
                forecast['expected_returns'] = np.zeros(len(results.predictions))
        else:
            # For regression, predictions are already expected returns
            forecast['expected_returns'] = results.predictions
        
        return forecast
