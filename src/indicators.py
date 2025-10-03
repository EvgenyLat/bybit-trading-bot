"""
Technical Indicators Module for Trading Bot
Implements various technical analysis indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Collection of technical analysis indicators"""
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """
        Simple Moving Average
        
        Args:
            data: Price series
            period: SMA period
            
        Returns:
            SMA series
        """
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """
        Exponential Moving Average
        
        Args:
            data: Price series
            period: EMA period
            
        Returns:
            EMA series
        """
        return data.ewm(span=period).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index
        
        Args:
            data: Price series
            period: RSI period (default 14)
            
        Returns:
            RSI series
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, 
             signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence)
        
        Args:
            data: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, 
                       std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands
        
        Args:
            data: Price series
            period: SMA period
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        middle_band = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                  k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_period: %K period
            d_period: %D period
            
        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, 
            period: int = 14) -> pd.Series:
        """
        Average True Range
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ATR period
            
        Returns:
            ATR series
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, 
                   period: int = 14) -> pd.Series:
        """
        Williams %R
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Williams %R period
            
        Returns:
            Williams %R series
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """
        Calculate all configured indicators
        
        Args:
            df: OHLCV DataFrame
            config: Configuration dictionary
            
        Returns:
            DataFrame with all indicators
        """
        try:
            logger.info("Calculating technical indicators")
            
            result_df = df.copy()
            
            # SMA
            sma_periods = config['indicators']['sma_periods']
            for period in sma_periods:
                result_df[f'sma_{period}'] = TechnicalIndicators.sma(df['close'], period)
            
            # EMA
            ema_periods = config['indicators']['ema_periods']
            for period in ema_periods:
                result_df[f'ema_{period}'] = TechnicalIndicators.ema(df['close'], period)
            
            # RSI
            rsi_period = config['indicators']['rsi_period']
            result_df['rsi'] = TechnicalIndicators.rsi(df['close'], rsi_period)
            
            # MACD
            macd_fast = config['indicators']['macd_fast']
            macd_slow = config['indicators']['macd_slow']
            macd_signal = config['indicators']['macd_signal']
            
            macd_line, signal_line, histogram = TechnicalIndicators.macd(
                df['close'], macd_fast, macd_slow, macd_signal
            )
            result_df['macd'] = macd_line
            result_df['macd_signal'] = signal_line
            result_df['macd_histogram'] = histogram
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['close'])
            result_df['bb_upper'] = bb_upper
            result_df['bb_middle'] = bb_middle
            result_df['bb_lower'] = bb_lower
            
            # Stochastic
            stoch_k, stoch_d = TechnicalIndicators.stochastic(
                df['high'], df['low'], df['close']
            )
            result_df['stoch_k'] = stoch_k
            result_df['stoch_d'] = stoch_d
            
            # ATR
            result_df['atr'] = TechnicalIndicators.atr(
                df['high'], df['low'], df['close']
            )
            
            # Williams %R
            result_df['williams_r'] = TechnicalIndicators.williams_r(
                df['high'], df['low'], df['close']
            )
            
            logger.info("Technical indicators calculated successfully")
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            raise
    
    @staticmethod
    def get_signal_strength(df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        Calculate overall signal strength based on multiple indicators
        
        Args:
            df: DataFrame with indicators
            config: Configuration dictionary
            
        Returns:
            Signal strength series (-1 to 1)
        """
        try:
            signals = []
            
            # RSI signals
            rsi = df['rsi']
            rsi_signal = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))
            signals.append(rsi_signal)
            
            # MACD signals
            macd = df['macd']
            macd_signal_line = df['macd_signal']
            macd_signal = np.where(macd > macd_signal_line, 1, 
                                 np.where(macd < macd_signal_line, -1, 0))
            signals.append(macd_signal)
            
            # Moving average signals
            sma_20 = df['sma_20']
            sma_50 = df['sma_50']
            ma_signal = np.where(sma_20 > sma_50, 1, 
                               np.where(sma_20 < sma_50, -1, 0))
            signals.append(ma_signal)
            
            # Bollinger Bands signals
            bb_upper = df['bb_upper']
            bb_lower = df['bb_lower']
            close = df['close']
            bb_signal = np.where(close < bb_lower, 1, 
                                np.where(close > bb_upper, -1, 0))
            signals.append(bb_signal)
            
            # Combine signals
            signal_matrix = np.array(signals)
            combined_signal = np.mean(signal_matrix, axis=0)
            
            return pd.Series(combined_signal, index=df.index)
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            raise

