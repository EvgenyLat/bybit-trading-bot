"""
Data Collector Service
Handles historical data fetching and real-time WebSocket data collection
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import redis
from pybit.unified_trading import HTTP, WebSocket
import yaml
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class TimescaleDBStorage:
    """TimescaleDB storage handler"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.engine = None
        self.session = None
        self._init_connection()
    
    def _init_connection(self):
        """Initialize TimescaleDB connection"""
        try:
            db_config = {
                'host': os.getenv('TIMESCALEDB_HOST', 'localhost'),
                'port': os.getenv('TIMESCALEDB_PORT', 5432),
                'dbname': os.getenv('TIMESCALEDB_DBNAME', 'trading_bot'),
                'user': os.getenv('TIMESCALEDB_USER', 'trading_user'),
                'password': os.getenv('TIMESCALEDB_PASSWORD', 'password')
            }
            
            connection_string = (
                f"postgresql://{db_config['user']}:{db_config['password']}"
                f"@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
            )
            
            self.engine = create_engine(connection_string)
            self.session = sessionmaker(bind=self.engine)()
            
            # Create tables if they don't exist
            self._create_tables()
            
            logger.info("TimescaleDB connection initialized")
            
        except Exception as e:
            logger.error(f"Error initializing TimescaleDB: {e}")
            raise
    
    def _create_tables(self):
        """Create necessary tables"""
        try:
            # Create OHLCV table
            create_ohlcv_table = """
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                timestamp TIMESTAMPTZ NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                open DECIMAL(20,8) NOT NULL,
                high DECIMAL(20,8) NOT NULL,
                low DECIMAL(20,8) NOT NULL,
                close DECIMAL(20,8) NOT NULL,
                volume DECIMAL(20,8) NOT NULL,
                turnover DECIMAL(20,8),
                PRIMARY KEY (timestamp, symbol, timeframe)
            );
            """
            
            # Create hypertable
            create_hypertable = """
            SELECT create_hypertable('ohlcv_data', 'timestamp', 
                                   if_not_exists => TRUE);
            """
            
            # Create indexes
            create_indexes = """
            CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe 
            ON ohlcv_data (symbol, timeframe, timestamp DESC);
            """
            
            self.session.execute(text(create_ohlcv_table))
            self.session.execute(text(create_hypertable))
            self.session.execute(text(create_indexes))
            self.session.commit()
            
            logger.info("TimescaleDB tables created successfully")
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            self.session.rollback()
            raise
    
    def store_ohlcv_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Store OHLCV data to TimescaleDB"""
        try:
            # Prepare data for insertion
            df_copy = df.copy()
            df_copy['symbol'] = symbol
            df_copy['timeframe'] = timeframe
            df_copy = df_copy.reset_index()
            
            # Rename timestamp column if needed
            if 'timestamp' not in df_copy.columns:
                df_copy = df_copy.rename(columns={df_copy.columns[0]: 'timestamp'})
            
            # Insert data
            df_copy.to_sql('ohlcv_data', self.engine, if_exists='append', 
                          index=False, method='multi')
            
            logger.info(f"Stored {len(df_copy)} OHLCV records for {symbol} {timeframe}")
            
        except Exception as e:
            logger.error(f"Error storing OHLCV data: {e}")
            raise
    
    def get_ohlcv_data(self, symbol: str, timeframe: str, 
                      start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Retrieve OHLCV data from TimescaleDB"""
        try:
            query = """
            SELECT timestamp, open, high, low, close, volume, turnover
            FROM ohlcv_data
            WHERE symbol = :symbol 
            AND timeframe = :timeframe
            AND timestamp BETWEEN :start_date AND :end_date
            ORDER BY timestamp ASC
            """
            
            df = pd.read_sql(
                query, 
                self.engine,
                params={
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'start_date': start_date,
                    'end_date': end_date
                },
                parse_dates=['timestamp']
            )
            
            df.set_index('timestamp', inplace=True)
            logger.info(f"Retrieved {len(df)} OHLCV records for {symbol} {timeframe}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving OHLCV data: {e}")
            raise
    
    def get_latest_data(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        """Get latest OHLCV data"""
        try:
            query = """
            SELECT timestamp, open, high, low, close, volume, turnover
            FROM ohlcv_data
            WHERE symbol = :symbol 
            AND timeframe = :timeframe
            ORDER BY timestamp DESC
            LIMIT :limit
            """
            
            df = pd.read_sql(
                query,
                self.engine,
                params={'symbol': symbol, 'timeframe': timeframe, 'limit': limit},
                parse_dates=['timestamp']
            )
            
            df = df.sort_values('timestamp').set_index('timestamp')
            logger.info(f"Retrieved latest {len(df)} OHLCV records for {symbol} {timeframe}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving latest data: {e}")
            raise


class RedisCache:
    """Redis cache handler for real-time data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.redis_client = None
        self._init_connection()
    
    def _init_connection(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                password=os.getenv('REDIS_PASSWORD'),
                db=int(os.getenv('REDIS_DB', 0)),
                decode_responses=True
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection initialized")
            
        except Exception as e:
            logger.error(f"Error initializing Redis: {e}")
            raise
    
    def store_latest_candle(self, symbol: str, timeframe: str, data: Dict):
        """Store latest candle data in Redis"""
        try:
            key = f"latest_candle:{symbol}:{timeframe}"
            self.redis_client.hset(key, mapping=data)
            self.redis_client.expire(key, 3600)  # Expire in 1 hour
            
        except Exception as e:
            logger.error(f"Error storing latest candle: {e}")
    
    def get_latest_candle(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get latest candle data from Redis"""
        try:
            key = f"latest_candle:{symbol}:{timeframe}"
            data = self.redis_client.hgetall(key)
            return data if data else None
            
        except Exception as e:
            logger.error(f"Error getting latest candle: {e}")
            return None


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
    
    def fetch_historical_data(self, symbol: str, timeframe: str, 
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


class WebSocketClient:
    """Real-time WebSocket client for Bybit"""
    
    def __init__(self, config: Dict, storage: TimescaleDBStorage, cache: RedisCache):
        self.config = config
        self.storage = storage
        self.cache = cache
        self.ws = None
        self.symbol = config['trading']['symbol']
        self.timeframe = config['trading']['timeframe']
        self.category = config['trading']['category']
        self.running = False
    
    def start(self):
        """Start WebSocket connection"""
        try:
            logger.info("Starting WebSocket client")
            
            self.ws = WebSocket(
                testnet=self.config['api']['testnet'],
                channel_type="public"
            )
            
            # Subscribe to kline data
            self.ws.kline_stream(
                symbol=self.symbol,
                interval=self.timeframe,
                callback=self._handle_kline_message
            )
            
            self.running = True
            logger.info("WebSocket client started")
            
        except Exception as e:
            logger.error(f"Error starting WebSocket client: {e}")
            raise
    
    def stop(self):
        """Stop WebSocket connection"""
        try:
            if self.ws:
                self.ws.exit()
                self.running = False
                logger.info("WebSocket client stopped")
        except Exception as e:
            logger.error(f"Error stopping WebSocket client: {e}")
    
    def _handle_kline_message(self, message):
        """Handle incoming kline messages"""
        try:
            if message.get('topic') == f'kline.{self.timeframe}.{self.symbol}':
                data = message['data']
                
                for kline in data:
                    # Process new candle
                    candle_data = {
                        'timestamp': pd.to_datetime(int(kline['start']), unit='ms'),
                        'open': float(kline['open']),
                        'high': float(kline['high']),
                        'low': float(kline['low']),
                        'close': float(kline['close']),
                        'volume': float(kline['volume']),
                        'turnover': float(kline['turnover'])
                    }
                    
                    # Store in Redis cache
                    self.cache.store_latest_candle(self.symbol, self.timeframe, candle_data)
                    
                    # Store completed candles in TimescaleDB
                    if kline['confirm']:  # Only store confirmed candles
                        df = pd.DataFrame([candle_data])
                        df.set_index('timestamp', inplace=True)
                        self.storage.store_ohlcv_data(df, self.symbol, self.timeframe)
                
        except Exception as e:
            logger.error(f"Error handling kline message: {e}")


class DataCollector:
    """Main data collector service"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.storage = TimescaleDBStorage(self.config)
        self.cache = RedisCache(self.config)
        self.historical_fetcher = HistoricalFetcher(self.config)
        self.ws_client = WebSocketClient(self.config, self.storage, self.cache)
        
        logger.info("Data Collector initialized")
    
    def _load_config(self) -> Dict:
        """Load configuration"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    async def download_historical_data(self, symbol: str, timeframe: str, 
                                     days: int = 365):
        """Download and store historical data"""
        try:
            logger.info(f"Downloading {days} days of historical data for {symbol} {timeframe}")
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Fetch historical data
            df = await self.historical_fetcher.fetch_historical_data(
                symbol, timeframe, start_date, end_date
            )
            
            # Store in TimescaleDB
            self.storage.store_ohlcv_data(df, symbol, timeframe)
            
            logger.info(f"Successfully downloaded and stored {len(df)} historical candles")
            
        except Exception as e:
            logger.error(f"Error downloading historical data: {e}")
            raise
    
    def start_real_time_collection(self):
        """Start real-time data collection"""
        try:
            logger.info("Starting real-time data collection")
            self.ws_client.start()
            
        except Exception as e:
            logger.error(f"Error starting real-time collection: {e}")
            raise
    
    def stop_real_time_collection(self):
        """Stop real-time data collection"""
        try:
            logger.info("Stopping real-time data collection")
            self.ws_client.stop()
            
        except Exception as e:
            logger.error(f"Error stopping real-time collection: {e}")
    
    def get_latest_data(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        """Get latest data from storage"""
        try:
            return self.storage.get_latest_data(symbol, timeframe, limit)
        except Exception as e:
            logger.error(f"Error getting latest data: {e}")
            raise
    
    def get_historical_data(self, symbol: str, timeframe: str, 
                          start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get historical data from storage"""
        try:
            return self.storage.get_ohlcv_data(symbol, timeframe, start_date, end_date)
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            raise

