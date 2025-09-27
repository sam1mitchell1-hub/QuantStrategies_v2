"""
Feature Engineering Pipeline

Builds comprehensive features for the trading strategy including:
- Realized volatility and bipower variation
- IV surface features (level, skew, term structure)
- Cross-asset features (VIX, credit, rates)
- Volume and liquidity features
- Regime indicators
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class FeatureSet:
    """Container for feature sets."""
    features: pd.DataFrame
    feature_names: List[str]
    target_returns: Optional[pd.Series] = None
    target_volatility: Optional[pd.Series] = None
    target_direction: Optional[pd.Series] = None
    metadata: Dict[str, Any] = None


class FeatureBuilder:
    """Feature engineering pipeline for options trading strategy."""
    
    def __init__(self, config):
        """Initialize feature builder with configuration."""
        self.config = config
        self.feature_cache = {}
        
    def build_features(self, 
                      price_data: pd.DataFrame,
                      options_data: pd.DataFrame,
                      vix_data: Optional[pd.DataFrame] = None,
                      rates_data: Optional[pd.DataFrame] = None,
                      credit_data: Optional[pd.DataFrame] = None) -> FeatureSet:
        """
        Build comprehensive feature set from market data.
        
        Args:
            price_data: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
            options_data: Options data with IV surface information
            vix_data: VIX data (optional)
            rates_data: Interest rates data (optional)
            credit_data: Credit spreads data (optional)
            
        Returns:
            FeatureSet with engineered features and targets
        """
        print("Building features...")
        
        # Ensure data is sorted by timestamp
        price_data = price_data.sort_index()
        options_data = options_data.sort_index()
        
        # Build base features from price data
        base_features = self._build_price_features(price_data)
        
        # Build volatility features
        vol_features = self._build_volatility_features(price_data)
        
        # Build IV surface features
        iv_features = self._build_iv_features(options_data, price_data)
        
        # Build cross-asset features
        cross_asset_features = self._build_cross_asset_features(
            vix_data, rates_data, credit_data, price_data.index
        )
        
        # Build volume and liquidity features
        liquidity_features = self._build_liquidity_features(price_data, options_data)
        
        # Build regime features
        regime_features = self._build_regime_features(price_data)
        
        # Combine all features
        all_features = pd.concat([
            base_features,
            vol_features,
            iv_features,
            cross_asset_features,
            liquidity_features,
            regime_features
        ], axis=1)
        
        # Remove any rows with all NaN values
        all_features = all_features.dropna(how='all')
        
        # Build targets
        targets = self._build_targets(price_data, all_features.index)
        
        # Create feature set
        feature_set = FeatureSet(
            features=all_features,
            feature_names=list(all_features.columns),
            target_returns=targets['returns'],
            target_volatility=targets['volatility'],
            target_direction=targets['direction'],
            metadata={
                'n_features': len(all_features.columns),
                'n_observations': len(all_features),
                'feature_groups': {
                    'base': len(base_features.columns),
                    'volatility': len(vol_features.columns),
                    'iv_surface': len(iv_features.columns),
                    'cross_asset': len(cross_asset_features.columns),
                    'liquidity': len(liquidity_features.columns),
                    'regime': len(regime_features.columns)
                }
            }
        )
        
        print(f"Built {len(all_features.columns)} features for {len(all_features)} observations")
        return feature_set
    
    def _build_price_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Build basic price-based features."""
        features = pd.DataFrame(index=price_data.index)
        
        # Returns
        features['return_1d'] = price_data['close'].pct_change()
        features['return_5d'] = price_data['close'].pct_change(5)
        features['return_20d'] = price_data['close'].pct_change(20)
        
        # Overnight gap
        features['overnight_gap'] = (price_data['open'] - price_data['close'].shift(1)) / price_data['close'].shift(1)
        
        # Last hour return (approximated as close - open)
        features['last_hour_return'] = (price_data['close'] - price_data['open']) / price_data['open']
        
        # Price levels
        features['price_level'] = price_data['close']
        features['log_price'] = np.log(price_data['close'])
        
        # High-low range
        features['hl_range'] = (price_data['high'] - price_data['low']) / price_data['close']
        features['hl_range_5d'] = features['hl_range'].rolling(5).mean()
        
        # Price momentum
        features['momentum_5d'] = price_data['close'] / price_data['close'].shift(5) - 1
        features['momentum_20d'] = price_data['close'] / price_data['close'].shift(20) - 1
        
        return features
    
    def _build_volatility_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Build realized volatility and related features."""
        features = pd.DataFrame(index=price_data.index)
        
        # Calculate returns for volatility estimation
        returns = price_data['close'].pct_change().dropna()
        
        # Realized volatility for different windows
        for window in self.config.features.realized_vol_windows:
            rv = returns.rolling(window).std() * np.sqrt(252)  # Annualized
            features[f'rv_{window}d'] = rv
            
            # RV percentiles
            rv_percentile = rv.rolling(252).rank(pct=True)
            features[f'rv_{window}d_percentile'] = rv_percentile
        
        # Bipower variation (more robust to jumps)
        bpv_window = self.config.features.bipower_variation_window
        abs_returns = returns.abs()
        bpv = (abs_returns * abs_returns.shift(1)).rolling(bpv_window).sum() * 252
        features[f'bpv_{bpv_window}d'] = bpv
        
        # Jump detection
        jump_threshold = self.config.features.jump_detection_threshold
        rv_5d = returns.rolling(5).std() * np.sqrt(252)
        rv_20d = returns.rolling(20).std() * np.sqrt(252)
        
        # Jump flag: large return relative to recent volatility
        features['jump_flag'] = (abs_returns > jump_threshold * rv_5d.shift(1)).astype(int)
        features['jump_intensity'] = features['jump_flag'].rolling(20).sum()
        
        # Volatility of volatility
        features['vol_of_vol'] = rv_5d.rolling(20).std()
        
        # Volatility regime
        rv_20d_percentile = rv_20d.rolling(252).rank(pct=True)
        features['low_vol_regime'] = (rv_20d_percentile < self.config.features.regime_thresholds['low_vol']).astype(int)
        features['high_vol_regime'] = (rv_20d_percentile > self.config.features.regime_thresholds['high_vol']).astype(int)
        features['extreme_vol_regime'] = (rv_20d_percentile > self.config.features.regime_thresholds['extreme_vol']).astype(int)
        
        return features
    
    def _build_iv_features(self, options_data: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """Build IV surface features."""
        features = pd.DataFrame(index=price_data.index)
        
        if options_data.empty:
            # Return empty features if no options data
            return features
        
        # Align options data with price data
        options_aligned = options_data.reindex(price_data.index, method='ffill')
        
        # IV level features
        if 'iv_atm' in options_aligned.columns:
            features['iv_atm'] = options_aligned['iv_atm']
            features['iv_atm_5d_ma'] = features['iv_atm'].rolling(5).mean()
            features['iv_atm_20d_ma'] = features['iv_atm'].rolling(20).mean()
            
            # IV percentiles
            features['iv_atm_percentile'] = features['iv_atm'].rolling(252).rank(pct=True)
        
        # IV skew features
        for strike_pct in self.config.features.iv_skew_strikes:
            col_name = f'iv_skew_{strike_pct:.0%}'
            if col_name in options_aligned.columns:
                features[col_name] = options_aligned[col_name]
                
                # Skew relative to ATM
                if 'iv_atm' in features.columns:
                    features[f'skew_{strike_pct:.0%}'] = features[col_name] - features['iv_atm']
        
        # IV term structure features
        for tenor in self.config.features.iv_term_structure_tenors:
            col_name = f'iv_{tenor}d'
            if col_name in options_aligned.columns:
                features[col_name] = options_aligned[col_name]
        
        # IV-RV gap (Volatility Risk Premium)
        if 'iv_atm' in features.columns and 'rv_20d' in features.columns:
            features['vrp'] = features['iv_atm'] - features['rv_20d']
            features['vrp_percentile'] = features['vrp'].rolling(252).rank(pct=True)
        
        # IV surface curvature (approximated)
        if all(f'skew_{strike:.0%}' in features.columns for strike in [0.95, 1.0, 1.05]):
            features['iv_curvature'] = (features['skew_95%'] - 2 * features['skew_100%'] + features['skew_105%'])
        
        return features
    
    def _build_cross_asset_features(self, 
                                  vix_data: Optional[pd.DataFrame],
                                  rates_data: Optional[pd.DataFrame],
                                  credit_data: Optional[pd.DataFrame],
                                  price_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Build cross-asset features."""
        features = pd.DataFrame(index=price_index)
        
        # VIX features
        if vix_data is not None and self.config.features.vix_features:
            vix_aligned = vix_data.reindex(price_index, method='ffill')
            
            if 'vix' in vix_aligned.columns:
                features['vix'] = vix_aligned['vix']
                features['vix_5d_ma'] = features['vix'].rolling(5).mean()
                features['vix_20d_ma'] = features['vix'].rolling(20).mean()
                features['vix_percentile'] = features['vix'].rolling(252).rank(pct=True)
                
                # VIX term structure (if available)
                if 'vix_9d' in vix_aligned.columns and 'vix_30d' in vix_aligned.columns:
                    features['vix_term_slope'] = vix_aligned['vix_30d'] - vix_aligned['vix_9d']
        
        # Rates features
        if rates_data is not None and self.config.features.rates_features:
            rates_aligned = rates_data.reindex(price_index, method='ffill')
            
            if '10y_rate' in rates_aligned.columns:
                features['10y_rate'] = rates_aligned['10y_rate']
                features['10y_rate_change'] = features['10y_rate'].diff()
                
                # Yield curve slope (if available)
                if '2y_rate' in rates_aligned.columns:
                    features['yield_curve_slope'] = features['10y_rate'] - rates_aligned['2y_rate']
        
        # Credit features
        if credit_data is not None and self.config.features.credit_features:
            credit_aligned = credit_data.reindex(price_index, method='ffill')
            
            if 'credit_spread' in credit_aligned.columns:
                features['credit_spread'] = credit_aligned['credit_spread']
                features['credit_spread_change'] = features['credit_spread'].diff()
                features['credit_spread_percentile'] = features['credit_spread'].rolling(252).rank(pct=True)
        
        return features
    
    def _build_liquidity_features(self, price_data: pd.DataFrame, options_data: pd.DataFrame) -> pd.DataFrame:
        """Build volume and liquidity features."""
        features = pd.DataFrame(index=price_data.index)
        
        # Volume features
        if 'volume' in price_data.columns:
            features['volume'] = price_data['volume']
            features['volume_5d_ma'] = features['volume'].rolling(5).mean()
            features['volume_20d_ma'] = features['volume'].rolling(20).mean()
            
            # Volume z-scores
            volume_window = self.config.features.volume_zscore_window
            volume_mean = features['volume'].rolling(volume_window).mean()
            volume_std = features['volume'].rolling(volume_window).std()
            features['volume_zscore'] = (features['volume'] - volume_mean) / volume_std
            
            # Volume percentiles
            volume_percentile_window = self.config.features.volume_percentile_window
            features['volume_percentile'] = features['volume'].rolling(volume_percentile_window).rank(pct=True)
        
        # Options liquidity features (if available)
        if not options_data.empty:
            options_aligned = options_data.reindex(price_data.index, method='ffill')
            
            if 'avg_bid_ask_spread' in options_aligned.columns:
                features['options_spread'] = options_aligned['avg_bid_ask_spread']
                features['options_spread_percentile'] = features['options_spread'].rolling(252).rank(pct=True)
            
            if 'total_oi' in options_aligned.columns:
                features['options_oi'] = options_aligned['total_oi']
                features['options_oi_percentile'] = features['options_oi'].rolling(252).rank(pct=True)
        
        return features
    
    def _build_regime_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Build regime and market state features."""
        features = pd.DataFrame(index=price_data.index)
        
        # Calculate returns for regime analysis
        returns = price_data['close'].pct_change()
        
        # RV percentiles for different windows
        for window in self.config.features.rv_percentile_windows:
            rv = returns.rolling(window).std() * np.sqrt(252)
            rv_percentile = rv.rolling(252).rank(pct=True)
            features[f'rv_{window}d_percentile'] = rv_percentile
        
        # Market regime dummies
        rv_20d = returns.rolling(20).std() * np.sqrt(252)
        rv_20d_percentile = rv_20d.rolling(252).rank(pct=True)
        
        features['low_vol_regime'] = (rv_20d_percentile < 0.25).astype(int)
        features['normal_vol_regime'] = ((rv_20d_percentile >= 0.25) & (rv_20d_percentile <= 0.75)).astype(int)
        features['high_vol_regime'] = (rv_20d_percentile > 0.75).astype(int)
        
        # Trend regime
        price_ma_20 = price_data['close'].rolling(20).mean()
        price_ma_50 = price_data['close'].rolling(50).mean()
        
        features['uptrend_regime'] = (price_ma_20 > price_ma_50).astype(int)
        features['downtrend_regime'] = (price_ma_20 < price_ma_50).astype(int)
        
        # Momentum regime
        momentum_5d = price_data['close'] / price_data['close'].shift(5) - 1
        features['strong_momentum'] = (momentum_5d > 0.02).astype(int)  # >2% in 5 days
        features['strong_reversal'] = (momentum_5d < -0.02).astype(int)  # <-2% in 5 days
        
        return features
    
    def _build_targets(self, price_data: pd.DataFrame, feature_index: pd.DatetimeIndex) -> Dict[str, pd.Series]:
        """Build target variables for training."""
        targets = {}
        
        # Calculate returns
        returns = price_data['close'].pct_change()
        
        # Target returns (h-day forward returns)
        horizon_days = self.config.market.trading_horizon_days
        targets['returns'] = returns.shift(-horizon_days)  # Forward returns
        
        # Target direction (binary classification)
        targets['direction'] = (targets['returns'] > 0).astype(int)
        
        # Target volatility (realized vol over horizon)
        # This is a simplified version - in practice, you'd use HAR-RV or similar
        targets['volatility'] = returns.rolling(horizon_days).std().shift(-horizon_days) * np.sqrt(252)
        
        # Align targets with feature index
        for key, target in targets.items():
            targets[key] = target.reindex(feature_index)
        
        return targets
    
    def get_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance from trained model."""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance()
        else:
            raise ValueError("Model does not support feature importance")
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 20):
        """Plot feature importance."""
        import matplotlib.pyplot as plt
        
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def save_features(self, feature_set: FeatureSet, filepath: str):
        """Save feature set to file."""
        feature_set.features.to_csv(filepath)
        print(f"Features saved to {filepath}")
    
    def load_features(self, filepath: str) -> FeatureSet:
        """Load feature set from file."""
        features = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return FeatureSet(
            features=features,
            feature_names=list(features.columns)
        )
