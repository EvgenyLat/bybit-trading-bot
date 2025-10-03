"""
Historical Data Fetcher
Downloads historical OHLCV data from Bybit API
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP
import yaml
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class HistoricalFetcher:
    """Fetches historical data from Bybit"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.api_config = config['api']
        self.session = HTTP(
            api_key=self.api_config['api_key'],
            api_secret=self.api_config['api_secret'],
            testnet=self.api_config['testnet']
        )
    
    async def fetch_historical_data(self, symbol: str, timeframe: str, 
                                  start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch historical data from Bybit"""
        try:
            logger.info(f"Fetching historical data for {symbol} {timeframe} "
                       f"from {start_date} to {end_date}")
            
            all_data = []
            current_date = start_date
            
            while current_date < end_date:
                # Calculate batch size (max 1000 candles per request)
                batch_end = min(current_date + timedelta(days=30), end_date)
                
                response = self.session.get_kline(
                    category=self.config['trading']['category'],
                    symbol=symbol,
                    interval=timeframe,
                    start=int(current_date.timestamp() * 1000),
                    end=int(batch_end.timestamp() * 1000),
                    limit=1000
                )
                
                if response['retCode'] != 0:
                    raise Exception(f"Bybit API error: {response['retMsg']}")
                
                klines = response['result']['list']
                
                for kline in klines:
                    all_data.append({
                        'timestamp': pd.to_datetime(int(kline[0]), unit='ms'),
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5]),
                        'turnover': float(kline[6])
                    })
                
                current_date = batch_end
                
                # Rate limiting
                await asyncio.sleep(0.1)
            
            df = pd.DataFrame(all_data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            logger.info(f"Fetched {len(df)} historical candles")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise
    
    def fetch_recent_data(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        """Fetch recent data (last N candles)"""
        try:
            logger.info(f"Fetching recent {limit} candles for {symbol} {timeframe}")
            
            response = self.session.get_kline(
                category=self.config['trading']['category'],
                symbol=symbol,
                interval=timeframe,
                limit=limit
            )
            
            if response['retCode'] != 0:
                raise Exception(f"Bybit API error: {response['retMsg']}")
            
            klines = response['result']['list']
            data = []
            
            for kline in reversed(klines):  # Bybit returns newest first
                data.append({
                    'timestamp': pd.to_datetime(int(kline[0]), unit='ms'),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5]),
                    'turnover': float(kline[6])
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            logger.info(f"Fetched {len(df)} recent candles")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching recent data: {e}")
            raise
    
    def validate_data_quality(self, df: pd.DataFrame) -> bool:
        """Validate data quality"""
        try:
            # Check for required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.error("Missing required columns")
                return False
            
            # Check for null values
            if df.isnull().any().any():
                logger.warning("Data contains null values")
                return False
            
            # Check for negative prices
            price_cols = ['open', 'high', 'low', 'close']
            if (df[price_cols] <= 0).any().any():
                logger.error("Data contains non-positive prices")
                return False
            
            # Check OHLC logic
            if not ((df['high'] >= df['low']) & 
                   (df['high'] >= df['open']) & 
                   (df['high'] >= df['close']) &
                   (df['low'] <= df['open']) & 
                   (df['low'] <= df['close'])).all():
                logger.error("OHLC data violates price logic")
                return False
            
            # Check for duplicate timestamps
            if df.index.duplicated().any():
                logger.warning("Data contains duplicate timestamps")
                return False
            
            logger.info("Data quality validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating data quality: {e}")
            return False
    
    def get_data_gaps(self, df: pd.DataFrame, expected_interval: str) -> List[Dict]:
        """Detect gaps in data"""
        try:
            gaps = []
            
            # Calculate expected interval in minutes
            interval_map = {
                '1': 1, '3': 3, '5': 5, '15': 15, '30': 30,
                '60': 60, '120': 120, '240': 240, '360': 360,
                '720': 720, 'D': 1440, 'W': 10080, 'M': 43200
            }
            
            expected_minutes = interval_map.get(expected_interval, 60)
            
            # Check for gaps
            timestamps = df.index
            for i in range(1, len(timestamps)):
                time_diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 60
                
                if time_diff > expected_minutes * 1.5:  # Allow 50% tolerance
                    gaps.append({
                        'start': timestamps[i-1],
                        'end': timestamps[i],
                        'duration_minutes': time_diff,
                        'expected_minutes': expected_minutes
                    })
            
            if gaps:
                logger.warning(f"Found {len(gaps)} data gaps")
            else:
                logger.info("No data gaps detected")
            
            return gaps
            
        except Exception as e:
            logger.error(f"Error detecting data gaps: {e}")
            return []

