"""
Feature Engineering Service
Advanced technical indicators and feature creation for ML models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import ta
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import yaml

logger = logging.getLogger(__name__)


class AdvancedIndicators:
    """Advanced technical indicators"""
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period).mean()
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        return ta.momentum.RSIIndicator(data, window=period).rsi()
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD indicator"""
        macd_indicator = ta.trend.MACD(data, window_fast=fast, window_slow=slow, window_sign=signal)
        return macd_indicator.macd(), macd_indicator.macd_signal(), macd_indicator.macd_diff()
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        bb_indicator = ta.volatility.BollingerBands(data, window=period, window_dev=std_dev)
        return bb_indicator.bollinger_hband(), bb_indicator.bollinger_mavg(), bb_indicator.bollinger_lband()
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        return ta.volatility.AverageTrueRange(high, low, close, window=period).average_true_range()
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        stoch_indicator = ta.momentum.StochasticOscillator(high, low, close, window=k_period, smooth_window=d_period)
        return stoch_indicator.stoch(), stoch_indicator.stoch_signal()
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R"""
        return ta.momentum.WilliamsRIndicator(high, low, close, lbp=period).williams_r()
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        return ta.trend.CCIIndicator(high, low, close, window=period).cci()
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Average Directional Index"""
        adx_indicator = ta.trend.ADXIndicator(high, low, close, window=period)
        return adx_indicator.adx(), adx_indicator.adx_pos(), adx_indicator.adx_neg()
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        return ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price"""
        return ta.volume.VolumeSMAIndicator(close, volume).volume_sma()


class FeatureBuilder:
    """Feature engineering and creation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.indicators_config = config.get('indicators', {})
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features"""
        try:
            features_df = df.copy()
            
            # Price changes
            features_df['price_change'] = df['close'].pct_change()
            features_df['price_change_2'] = df['close'].pct_change(2)
            features_df['price_change_5'] = df['close'].pct_change(5)
            features_df['price_change_10'] = df['close'].pct_change(10)
            
            # Price ratios
            features_df['high_low_ratio'] = df['high'] / df['low']
            features_df['close_open_ratio'] = df['close'] / df['open']
            features_df['high_close_ratio'] = df['high'] / df['close']
            features_df['low_close_ratio'] = df['low'] / df['close']
            
            # Price positions
            features_df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            features_df['open_position'] = (df['open'] - df['low']) / (df['high'] - df['low'])
            
            # Volatility measures
            features_df['price_range'] = df['high'] - df['low']
            features_df['price_range_pct'] = features_df['price_range'] / df['close']
            features_df['body_size'] = abs(df['close'] - df['open'])
            features_df['body_size_pct'] = features_df['body_size'] / df['close']
            
            # Log returns
            features_df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            features_df['log_return_2'] = np.log(df['close'] / df['close'].shift(2))
            features_df['log_return_5'] = np.log(df['close'] / df['close'].shift(5))
            
            logger.info(f"Created {len([col for col in features_df.columns if col not in df.columns])} price features")
            return features_df
            
        except Exception as e:
            logger.error(f"Error creating price features: {e}")
            raise
    
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features"""
        try:
            features_df = df.copy()
            
            # Volume changes
            features_df['volume_change'] = df['volume'].pct_change()
            features_df['volume_change_2'] = df['volume'].pct_change(2)
            features_df['volume_change_5'] = df['volume'].pct_change(5)
            
            # Volume ratios
            features_df['volume_sma_20'] = df['volume'].rolling(20).mean()
            features_df['volume_ratio'] = df['volume'] / features_df['volume_sma_20']
            features_df['volume_sma_50'] = df['volume'].rolling(50).mean()
            features_df['volume_ratio_50'] = df['volume'] / features_df['volume_sma_50']
            
            # Price-volume features
            features_df['price_volume'] = df['close'] * df['volume']
            features_df['price_volume_change'] = features_df['price_volume'].pct_change()
            
            # Volume-weighted features
            features_df['vwap'] = AdvancedIndicators.vwap(df['high'], df['low'], df['close'], df['volume'])
            features_df['price_vwap_ratio'] = df['close'] / features_df['vwap']
            features_df['price_vwap_diff'] = df['close'] - features_df['vwap']
            
            # OBV
            features_df['obv'] = AdvancedIndicators.obv(df['close'], df['volume'])
            features_df['obv_change'] = features_df['obv'].pct_change()
            
            logger.info(f"Created {len([col for col in features_df.columns if col not in df.columns])} volume features")
            return features_df
            
        except Exception as e:
            logger.error(f"Error creating volume features: {e}")
            raise
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators"""
        try:
            features_df = df.copy()
            
            # Moving averages
            sma_periods = self.indicators_config.get('sma_periods', [5, 10, 20, 50])
            for period in sma_periods:
                features_df[f'sma_{period}'] = AdvancedIndicators.sma(df['close'], period)
                features_df[f'price_sma_{period}_ratio'] = df['close'] / features_df[f'sma_{period}']
            
            ema_periods = self.indicators_config.get('ema_periods', [8, 13, 21, 34])
            for period in ema_periods:
                features_df[f'ema_{period}'] = AdvancedIndicators.ema(df['close'], period)
                features_df[f'price_ema_{period}_ratio'] = df['close'] / features_df[f'ema_{period}']
            
            # RSI
            rsi_period = self.indicators_config.get('rsi_period', 14)
            features_df['rsi'] = AdvancedIndicators.rsi(df['close'], rsi_period)
            features_df['rsi_change'] = features_df['rsi'].diff()
            features_df['rsi_oversold'] = (features_df['rsi'] < 30).astype(int)
            features_df['rsi_overbought'] = (features_df['rsi'] > 70).astype(int)
            
            # MACD
            macd_fast = self.indicators_config.get('macd_fast', 12)
            macd_slow = self.indicators_config.get('macd_slow', 26)
            macd_signal = self.indicators_config.get('macd_signal', 9)
            
            macd_line, signal_line, histogram = AdvancedIndicators.macd(
                df['close'], macd_fast, macd_slow, macd_signal
            )
            features_df['macd'] = macd_line
            features_df['macd_signal'] = signal_line
            features_df['macd_histogram'] = histogram
            features_df['macd_signal_diff'] = macd_line - signal_line
            
            # Bollinger Bands
            bb_period = self.indicators_config.get('bb_period', 20)
            bb_std = self.indicators_config.get('bb_std', 2)
            
            bb_upper, bb_middle, bb_lower = AdvancedIndicators.bollinger_bands(
                df['close'], bb_period, bb_std
            )
            features_df['bb_upper'] = bb_upper
            features_df['bb_middle'] = bb_middle
            features_df['bb_lower'] = bb_lower
            features_df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            features_df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
            features_df['bb_squeeze'] = (features_df['bb_width'] < features_df['bb_width'].rolling(20).mean() * 0.8).astype(int)
            
            # ATR
            atr_period = self.indicators_config.get('atr_period', 14)
            features_df['atr'] = AdvancedIndicators.atr(df['high'], df['low'], df['close'], atr_period)
            features_df['atr_ratio'] = features_df['atr'] / df['close']
            features_df['atr_change'] = features_df['atr'].pct_change()
            
            # Stochastic
            stoch_k_period = self.indicators_config.get('stoch_k', 14)
            stoch_d_period = self.indicators_config.get('stoch_d', 3)
            
            stoch_k, stoch_d = AdvancedIndicators.stochastic(
                df['high'], df['low'], df['close'], stoch_k_period, stoch_d_period
            )
            features_df['stoch_k'] = stoch_k
            features_df['stoch_d'] = stoch_d
            features_df['stoch_kd_diff'] = stoch_k - stoch_d
            
            # Williams %R
            features_df['williams_r'] = AdvancedIndicators.williams_r(df['high'], df['low'], df['close'])
            
            # CCI
            features_df['cci'] = AdvancedIndicators.cci(df['high'], df['low'], df['close'])
            
            # ADX
            adx, adx_pos, adx_neg = AdvancedIndicators.adx(df['high'], df['low'], df['close'])
            features_df['adx'] = adx
            features_df['adx_pos'] = adx_pos
            features_df['adx_neg'] = adx_neg
            
            logger.info(f"Created {len([col for col in features_df.columns if col not in df.columns])} technical indicators")
            return features_df
            
        except Exception as e:
            logger.error(f"Error creating technical indicators: {e}")
            raise
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        try:
            features_df = df.copy()
            
            # Extract time components
            features_df['hour'] = df.index.hour
            features_df['day_of_week'] = df.index.dayofweek
            features_df['day_of_month'] = df.index.day
            features_df['month'] = df.index.month
            features_df['quarter'] = df.index.quarter
            
            # Cyclical encoding
            features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
            features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
            features_df['day_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
            features_df['day_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
            features_df['month_sin'] = np.sin(2 * np.pi * features_df['month'] / 12)
            features_df['month_cos'] = np.cos(2 * np.pi * features_df['month'] / 12)
            
            # Trading session indicators
            features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
            features_df['is_london_session'] = ((features_df['hour'] >= 8) & (features_df['hour'] < 16)).astype(int)
            features_df['is_ny_session'] = ((features_df['hour'] >= 13) & (features_df['hour'] < 21)).astype(int)
            features_df['is_asian_session'] = ((features_df['hour'] >= 0) & (features_df['hour'] < 8)).astype(int)
            
            logger.info(f"Created {len([col for col in features_df.columns if col not in df.columns])} time features")
            return features_df
            
        except Exception as e:
            logger.error(f"Error creating time features: {e}")
            raise
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features"""
        try:
            features_df = df.copy()
            
            # Rolling statistics
            windows = [5, 10, 20, 50]
            for window in windows:
                # Price statistics
                features_df[f'price_mean_{window}'] = df['close'].rolling(window).mean()
                features_df[f'price_std_{window}'] = df['close'].rolling(window).std()
                features_df[f'price_skew_{window}'] = df['close'].rolling(window).skew()
                features_df[f'price_kurt_{window}'] = df['close'].rolling(window).kurt()
                
                # Volume statistics
                features_df[f'volume_mean_{window}'] = df['volume'].rolling(window).mean()
                features_df[f'volume_std_{window}'] = df['volume'].rolling(window).std()
                
                # Price position in distribution
                features_df[f'price_percentile_{window}'] = df['close'].rolling(window).rank(pct=True)
            
            # Correlation features
            features_df['price_volume_corr'] = df['close'].rolling(20).corr(df['volume'])
            features_df['high_low_corr'] = df['high'].rolling(20).corr(df['low'])
            
            # Momentum features
            features_df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            features_df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            features_df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
            
            # Rate of change
            features_df['roc_5'] = df['close'].pct_change(5)
            features_df['roc_10'] = df['close'].pct_change(10)
            features_df['roc_20'] = df['close'].pct_change(20)
            
            logger.info(f"Created {len([col for col in features_df.columns if col not in df.columns])} statistical features")
            return features_df
            
        except Exception as e:
            logger.error(f"Error creating statistical features: {e}")
            raise
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features"""
        try:
            logger.info("Creating all features")
            
            # Create different types of features
            features_df = self.create_price_features(df)
            features_df = self.create_volume_features(features_df)
            features_df = self.create_technical_indicators(features_df)
            features_df = self.create_time_features(features_df)
            features_df = self.create_statistical_features(features_df)
            
            # Remove NaN values
            features_df = features_df.dropna()
            
            logger.info(f"Created {len(features_df.columns)} total features")
            return features_df
            
        except Exception as e:
            logger.error(f"Error creating all features: {e}")
            raise
    
    def normalize_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """Normalize features for ML models"""
        try:
            # Select numeric columns only
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if fit_scaler:
                # Fit scaler on training data
                normalized_data = self.feature_scaler.fit_transform(df[numeric_cols])
            else:
                # Transform using fitted scaler
                normalized_data = self.feature_scaler.transform(df[numeric_cols])
            
            # Create new DataFrame with normalized data
            normalized_df = df.copy()
            normalized_df[numeric_cols] = normalized_data
            
            logger.info(f"Normalized {len(numeric_cols)} features")
            return normalized_df
            
        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
            raise
    
    def create_target_variables(self, df: pd.DataFrame, prediction_horizon: int = 5) -> pd.DataFrame:
        """Create target variables for ML models"""
        try:
            features_df = df.copy()
            
            # Price direction prediction
            features_df['price_change_future'] = df['close'].shift(-prediction_horizon) / df['close'] - 1
            
            # Binary classification targets
            features_df['target_binary'] = (features_df['price_change_future'] > 0).astype(int)
            
            # Multi-class classification
            features_df['target_multiclass'] = pd.cut(
                features_df['price_change_future'],
                bins=[-np.inf, -0.02, 0, 0.02, np.inf],
                labels=[0, 1, 2, 3]  # strong down, down, up, strong up
            ).astype(int)
            
            # Regression target
            features_df['target_regression'] = features_df['price_change_future']
            
            # Volatility target
            features_df['volatility_future'] = df['close'].rolling(prediction_horizon).std().shift(-prediction_horizon)
            
            logger.info("Created target variables")
            return features_df
            
        except Exception as e:
            logger.error(f"Error creating target variables: {e}")
            raise

