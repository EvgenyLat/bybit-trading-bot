"""
Bybit Data Feed Module for Trading Bot
Handles market data retrieval and processing using Bybit API
"""

import pandas as pd
import numpy as np
import asyncio
import websocket
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
import logging
import yaml
from pybit.unified_trading import HTTP
from pybit.unified_trading import WebSocket

logger = logging.getLogger(__name__)


class BybitDataFeed:
    """Handles market data retrieval and processing from Bybit"""
    
    def __init__(self, config: Dict):
        """
        Initialize BybitDataFeed with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.symbol = config['trading']['symbol']
        self.timeframe = config['trading']['timeframe']
        self.category = config['trading']['category']
        
        # Initialize Bybit connections
        self._init_bybit_connections()
        
        # Data storage
        self.realtime_data = {}
        self.historical_data = pd.DataFrame()
        self.data_callbacks = []
        
        # WebSocket connection
        self.ws = None
        self.ws_thread = None
        self.ws_connected = False
        
    def _init_bybit_connections(self):
        """Initialize Bybit HTTP and WebSocket connections"""
        try:
            api_config = self.config['api']
            
            # Initialize HTTP session for REST API
            self.http_session = HTTP(
                api_key=api_config['api_key'],
                api_secret=api_config['api_secret'],
                testnet=api_config['testnet']
            )
            
            # Initialize WebSocket for real-time data
            self.ws_endpoint = "wss://stream-testnet.bybit.com/v5/public/linear" if api_config['testnet'] else "wss://stream.bybit.com/v5/public/linear"
            
            logger.info("Bybit connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Bybit connections: {e}")
            raise
    
    def get_historical_data(self, symbol: str, timeframe: str, 
                          limit: int = 1000) -> pd.DataFrame:
        """
        Get historical OHLCV data from Bybit
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
            limit: Number of candles to retrieve
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching historical data for {symbol} {timeframe} from Bybit")
            
            # Get kline data from Bybit
            response = self.http_session.get_kline(
                category=self.category,
                symbol=symbol,
                interval=timeframe,
                limit=min(limit, 1000)  # Bybit max limit is 1000
            )
            
            if response['retCode'] != 0:
                raise Exception(f"Bybit API error: {response['retMsg']}")
            
            # Convert to DataFrame
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
            
            logger.info(f"Retrieved {len(df)} candles from Bybit")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data from Bybit: {e}")
            # Fallback to mock data for testing
            return self._get_mock_data(limit)
    
    def _get_mock_data(self, limit: int) -> pd.DataFrame:
        """Generate mock data for testing"""
        try:
            logger.info("Generating mock data for testing")
            
            dates = pd.date_range(
                end=datetime.now(),
                periods=limit,
                freq='1T'  # 1 minute
            )
            
            # Generate realistic price data
            np.random.seed(42)
            base_price = 50000  # Base price for BTC
            
            data = []
            for i, date in enumerate(dates):
                # Random walk with trend
                price_change = np.random.normal(0, 0.01)  # 1% volatility
                base_price *= (1 + price_change)
                
                # Generate OHLC
                high = base_price * (1 + abs(np.random.normal(0, 0.005)))
                low = base_price * (1 - abs(np.random.normal(0, 0.005)))
                volume = np.random.uniform(100, 1000)
                
                data.append({
                    'timestamp': date,
                    'open': base_price,
                    'high': high,
                    'low': low,
                    'close': base_price,
                    'volume': volume,
                    'turnover': volume * base_price
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating mock data: {e}")
            raise
    
    def start_websocket(self):
        """Start WebSocket connection for real-time data"""
        try:
            logger.info("Starting Bybit WebSocket connection")
            
            # Create WebSocket instance
            self.ws = WebSocket(
                testnet=self.config['api']['testnet'],
                channel_type="public"
            )
            
            # Subscribe to kline data
            self.ws.kline_stream(
                symbol=self.symbol,
                interval=self.timeframe,
                callback=self._handle_websocket_message
            )
            
            # Subscribe to orderbook data
            self.ws.orderbook_stream(
                symbol=self.symbol,
                depth=25,
                callback=self._handle_orderbook_message
            )
            
            # Subscribe to trade data
            self.ws.trade_stream(
                symbol=self.symbol,
                callback=self._handle_trade_message
            )
            
            self.ws_connected = True
            logger.info("WebSocket connection established")
            
        except Exception as e:
            logger.error(f"Error starting WebSocket: {e}")
            self.ws_connected = False
    
    def stop_websocket(self):
        """Stop WebSocket connection"""
        try:
            if self.ws:
                self.ws.exit()
                self.ws_connected = False
                logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error stopping WebSocket: {e}")
    
    def _handle_websocket_message(self, message):
        """Handle WebSocket kline messages"""
        try:
            if message.get('topic') == f'kline.{self.timeframe}.{self.symbol}':
                data = message['data']
                
                for kline in data:
                    timestamp = pd.to_datetime(int(kline['start']), unit='ms')
                    
                    self.realtime_data.update({
                        'timestamp': timestamp,
                        'open': float(kline['open']),
                        'high': float(kline['high']),
                        'low': float(kline['low']),
                        'close': float(kline['close']),
                        'volume': float(kline['volume']),
                        'turnover': float(kline['turnover'])
                    })
                
                # Notify callbacks
                self._notify_callbacks(self.realtime_data)
                
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    def _handle_orderbook_message(self, message):
        """Handle WebSocket orderbook messages"""
        try:
            if message.get('topic') == f'orderbook.25.{self.symbol}':
                data = message['data']
                
                # Extract best bid/ask
                if data['b'] and data['a']:  # bids and asks exist
                    best_bid = float(data['b'][0][0])
                    best_ask = float(data['a'][0][0])
                    
                    self.realtime_data.update({
                        'best_bid': best_bid,
                        'best_ask': best_ask,
                        'spread': best_ask - best_bid,
                        'mid_price': (best_bid + best_ask) / 2
                    })
                
        except Exception as e:
            logger.error(f"Error handling orderbook message: {e}")
    
    def _handle_trade_message(self, message):
        """Handle WebSocket trade messages"""
        try:
            if message.get('topic') == f'publicTrade.{self.symbol}':
                data = message['data']
                
                for trade in data:
                    self.realtime_data.update({
                        'last_trade_price': float(trade['p']),
                        'last_trade_size': float(trade['v']),
                        'last_trade_time': pd.to_datetime(int(trade['T']), unit='ms'),
                        'trade_side': trade['S']  # Buy or Sell
                    })
                
        except Exception as e:
            logger.error(f"Error handling trade message: {e}")
    
    def add_data_callback(self, callback: Callable):
        """Add callback function for real-time data updates"""
        self.data_callbacks.append(callback)
    
    def _notify_callbacks(self, data: Dict):
        """Notify all registered callbacks with new data"""
        for callback in self.data_callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in data callback: {e}")
    
    def get_realtime_data(self, symbol: str) -> Dict:
        """
        Get current real-time market data
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with current market data
        """
        try:
            # If WebSocket is connected, return latest data
            if self.ws_connected and self.realtime_data:
                return self.realtime_data.copy()
            
            # Fallback to REST API
            response = self.http_session.get_tickers(
                category=self.category,
                symbol=symbol
            )
            
            if response['retCode'] != 0:
                raise Exception(f"Bybit API error: {response['retMsg']}")
            
            ticker = response['result']['list'][0]
            
            return {
                'symbol': symbol,
                'price': float(ticker['lastPrice']),
                'bid': float(ticker['bid1Price']),
                'ask': float(ticker['ask1Price']),
                'volume': float(ticker['volume24h']),
                'turnover': float(ticker['turnover24h']),
                'high': float(ticker['high24h']),
                'low': float(ticker['low24h']),
                'change': float(ticker['price24hPcnt']),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting real-time data: {e}")
            # Return mock data as fallback
            return self._get_mock_realtime_data(symbol)
    
    def _get_mock_realtime_data(self, symbol: str) -> Dict:
        """Generate mock real-time data"""
        import numpy as np
        
        base_price = 50000 + np.random.normal(0, 1000)
        
        return {
            'symbol': symbol,
            'price': base_price,
            'bid': base_price * 0.9999,
            'ask': base_price * 1.0001,
            'volume': np.random.uniform(100, 1000),
            'turnover': np.random.uniform(5000000, 10000000),
            'high': base_price * 1.02,
            'low': base_price * 0.98,
            'change': np.random.uniform(-0.05, 0.05),
            'timestamp': datetime.now()
        }
    
    def get_orderbook(self, symbol: str, depth: int = 25) -> Dict:
        """
        Get orderbook data
        
        Args:
            symbol: Trading pair symbol
            depth: Orderbook depth
            
        Returns:
            Dictionary with orderbook data
        """
        try:
            response = self.http_session.get_orderbook(
                category=self.category,
                symbol=symbol,
                limit=depth
            )
            
            if response['retCode'] != 0:
                raise Exception(f"Bybit API error: {response['retMsg']}")
            
            orderbook = response['result']
            
            return {
                'symbol': symbol,
                'bids': [[float(bid[0]), float(bid[1])] for bid in orderbook['b']],
                'asks': [[float(ask[0]), float(ask[1])] for ask in orderbook['a']],
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting orderbook: {e}")
            return {}
    
    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """
        Get recent trades
        
        Args:
            symbol: Trading pair symbol
            limit: Number of trades to retrieve
            
        Returns:
            List of recent trades
        """
        try:
            response = self.http_session.get_public_trade_history(
                category=self.category,
                symbol=symbol,
                limit=limit
            )
            
            if response['retCode'] != 0:
                raise Exception(f"Bybit API error: {response['retMsg']}")
            
            trades = []
            for trade in response['result']['list']:
                trades.append({
                    'price': float(trade['p']),
                    'size': float(trade['v']),
                    'side': trade['S'],
                    'timestamp': pd.to_datetime(int(trade['T']), unit='ms')
                })
            
            return trades
            
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
            return []
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate data quality
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if data is valid
        """
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
            
            logger.info("Data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return False
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            response = self.http_session.get_wallet_balance(
                accountType="UNIFIED"
            )
            
            if response['retCode'] != 0:
                raise Exception(f"Bybit API error: {response['retMsg']}")
            
            return response['result']
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        try:
            response = self.http_session.get_positions(
                category=self.category,
                symbol=self.symbol
            )
            
            if response['retCode'] != 0:
                raise Exception(f"Bybit API error: {response['retMsg']}")
            
            return response['result']['list']
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
