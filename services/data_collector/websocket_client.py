"""
WebSocket Client for Real-time Data
Handles real-time data streaming from Bybit WebSocket API
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable
import pandas as pd
import json
from pybit.unified_trading import WebSocket
import yaml
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class WebSocketClient:
    """Real-time WebSocket client for Bybit"""
    
    def __init__(self, config: Dict, data_callback: Optional[Callable] = None):
        self.config = config
        self.data_callback = data_callback
        self.ws = None
        self.symbol = config['trading']['symbol']
        self.timeframe = config['trading']['timeframe']
        self.category = config['trading']['category']
        self.running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5
        
        # Data buffers
        self.latest_candles = {}
        self.orderbook_data = {}
        self.trade_data = []
        
        logger.info("WebSocket client initialized")
    
    def start(self):
        """Start WebSocket connection"""
        try:
            logger.info("Starting WebSocket client")
            
            self.ws = WebSocket(
                testnet=self.config['api']['testnet'],
                channel_type="public"
            )
            
            # Subscribe to different data streams
            self._subscribe_to_streams()
            
            self.running = True
            self.reconnect_attempts = 0
            logger.info("WebSocket client started successfully")
            
        except Exception as e:
            logger.error(f"Error starting WebSocket client: {e}")
            self._handle_reconnect()
    
    def stop(self):
        """Stop WebSocket connection"""
        try:
            if self.ws:
                self.ws.exit()
                self.running = False
                logger.info("WebSocket client stopped")
        except Exception as e:
            logger.error(f"Error stopping WebSocket client: {e}")
    
    def _subscribe_to_streams(self):
        """Subscribe to various data streams"""
        try:
            # Subscribe to kline data
            self.ws.kline_stream(
                symbol=self.symbol,
                interval=self.timeframe,
                callback=self._handle_kline_message
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
            
            logger.info("Subscribed to all data streams")
            
        except Exception as e:
            logger.error(f"Error subscribing to streams: {e}")
            raise
    
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
                        'turnover': float(kline['turnover']),
                        'confirm': kline['confirm']
                    }
                    
                    # Store latest candle
                    self.latest_candles[self.timeframe] = candle_data
                    
                    # Call data callback if provided
                    if self.data_callback:
                        self.data_callback('kline', candle_data)
                    
                    # Log confirmed candles
                    if kline['confirm']:
                        logger.debug(f"Confirmed candle: {candle_data['timestamp']} "
                                   f"O:{candle_data['open']} H:{candle_data['high']} "
                                   f"L:{candle_data['low']} C:{candle_data['close']}")
                
        except Exception as e:
            logger.error(f"Error handling kline message: {e}")
    
    def _handle_orderbook_message(self, message):
        """Handle incoming orderbook messages"""
        try:
            if message.get('topic') == f'orderbook.25.{self.symbol}':
                data = message['data']
                
                orderbook_data = {
                    'timestamp': pd.to_datetime(int(data['ts']), unit='ms'),
                    'symbol': self.symbol,
                    'bids': [[float(bid[0]), float(bid[1])] for bid in data['b']],
                    'asks': [[float(ask[0]), float(ask[1])] for ask in data['a']],
                    'update_id': data['u']
                }
                
                # Store orderbook data
                self.orderbook_data = orderbook_data
                
                # Call data callback if provided
                if self.data_callback:
                    self.data_callback('orderbook', orderbook_data)
                
        except Exception as e:
            logger.error(f"Error handling orderbook message: {e}")
    
    def _handle_trade_message(self, message):
        """Handle incoming trade messages"""
        try:
            if message.get('topic') == f'publicTrade.{self.symbol}':
                data = message['data']
                
                for trade in data:
                    trade_data = {
                        'timestamp': pd.to_datetime(int(trade['T']), unit='ms'),
                        'symbol': self.symbol,
                        'price': float(trade['p']),
                        'size': float(trade['v']),
                        'side': trade['S'],
                        'trade_id': trade['i']
                    }
                    
                    # Store trade data (keep last 1000)
                    self.trade_data.append(trade_data)
                    if len(self.trade_data) > 1000:
                        self.trade_data = self.trade_data[-1000:]
                    
                    # Call data callback if provided
                    if self.data_callback:
                        self.data_callback('trade', trade_data)
                
        except Exception as e:
            logger.error(f"Error handling trade message: {e}")
    
    def _handle_reconnect(self):
        """Handle reconnection logic"""
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            logger.warning(f"Attempting to reconnect ({self.reconnect_attempts}/"
                         f"{self.max_reconnect_attempts}) in {self.reconnect_delay} seconds")
            
            asyncio.create_task(self._reconnect_after_delay())
        else:
            logger.error("Max reconnection attempts reached. Stopping WebSocket client.")
            self.running = False
    
    async def _reconnect_after_delay(self):
        """Reconnect after delay"""
        await asyncio.sleep(self.reconnect_delay)
        if self.running:
            self.start()
    
    def get_latest_candle(self, timeframe: str = None) -> Optional[Dict]:
        """Get latest candle data"""
        timeframe = timeframe or self.timeframe
        return self.latest_candles.get(timeframe)
    
    def get_latest_orderbook(self) -> Optional[Dict]:
        """Get latest orderbook data"""
        return self.orderbook_data
    
    def get_recent_trades(self, limit: int = 100) -> List[Dict]:
        """Get recent trades"""
        return self.trade_data[-limit:] if self.trade_data else []
    
    def get_market_depth(self) -> Dict:
        """Get market depth information"""
        try:
            orderbook = self.get_latest_orderbook()
            if not orderbook:
                return {}
            
            bids = orderbook['bids']
            asks = orderbook['asks']
            
            if not bids or not asks:
                return {}
            
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2
            
            # Calculate bid/ask volumes
            bid_volume = sum([bid[1] for bid in bids[:5]])  # Top 5 levels
            ask_volume = sum([ask[1] for ask in asks[:5]])   # Top 5 levels
            
            return {
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread,
                'spread_pct': (spread / mid_price) * 100,
                'mid_price': mid_price,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'volume_imbalance': (bid_volume - ask_volume) / (bid_volume + ask_volume),
                'timestamp': orderbook['timestamp']
            }
            
        except Exception as e:
            logger.error(f"Error calculating market depth: {e}")
            return {}
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected"""
        return self.running and self.ws is not None
    
    def get_connection_status(self) -> Dict:
        """Get connection status information"""
        return {
            'connected': self.is_connected(),
            'running': self.running,
            'reconnect_attempts': self.reconnect_attempts,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'latest_candle_timestamp': self.latest_candles.get(self.timeframe, {}).get('timestamp'),
            'orderbook_timestamp': self.orderbook_data.get('timestamp'),
            'trade_count': len(self.trade_data)
        }

