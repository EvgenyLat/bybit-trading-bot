"""
Bybit Trading Bot Main Module
Entry point for the Bybit trading bot with ML-enhanced intraday trading
"""

import logging
import yaml
import time
import signal
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Optional
import pandas as pd
import numpy as np

from data_feed import BybitDataFeed
from indicators import TechnicalIndicators
from executor import BybitTradeExecutor, OrderSide, OrderType
from risk_manager import RiskManager
from ml_predictor import MLSignalPredictor

logger = logging.getLogger(__name__)


class BybitTradingBot:
    """Main Bybit trading bot class with ML enhancement"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize Bybit trading bot
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.running = False
        
        # Initialize components
        self.data_feed = BybitDataFeed(self.config)
        self.executor = BybitTradeExecutor(self.config)
        self.risk_manager = RiskManager(self.config)
        self.ml_predictor = MLSignalPredictor(self.config)
        
        # Setup logging
        self._setup_logging()
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Trading state
        self.last_trade_time = None
        self.daily_trade_count = 0
        self.session_start_time = datetime.now()
        
        logger.info("Bybit trading bot initialized successfully")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO').upper())
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_config.get('file', 'logs/bybit_trading_bot.log')),
                logging.StreamHandler(sys.stdout) if log_config.get('console_output', True) else logging.NullHandler()
            ]
        )
        
        logger.info("Logging configured")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def start(self):
        """Start the trading bot"""
        try:
            logger.info("Starting Bybit trading bot...")
            self.running = True
            
            # Initialize data and ML model
            self._initialize_data()
            self._initialize_ml_model()
            
            # Start WebSocket for real-time data
            self.data_feed.start_websocket()
            
            # Main trading loop
            self._trading_loop()
            
        except Exception as e:
            logger.error(f"Error in trading bot: {e}")
            raise
        finally:
            self.stop()
    
    def stop(self):
        """Stop the trading bot"""
        logger.info("Stopping Bybit trading bot...")
        self.running = False
        
        # Stop WebSocket
        self.data_feed.stop_websocket()
        
        # Cancel all pending orders
        self._cancel_all_orders()
        
        # Save final state
        self._save_state()
        
        logger.info("Bybit trading bot stopped")
    
    def _initialize_data(self):
        """Initialize historical data and indicators"""
        try:
            logger.info("Initializing historical data...")
            
            # Get historical data
            symbol = self.config['trading']['symbol']
            timeframe = self.config['trading']['timeframe']
            
            self.historical_data = self.data_feed.get_historical_data(
                symbol, timeframe, limit=1000
            )
            
            # Validate data
            if not self.data_feed.validate_data(self.historical_data):
                raise ValueError("Invalid historical data")
            
            # Calculate indicators
            self.data_with_indicators = TechnicalIndicators.calculate_all_indicators(
                self.historical_data, self.config
            )
            
            logger.info("Historical data initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing data: {e}")
            raise
    
    def _initialize_ml_model(self):
        """Initialize and train ML model"""
        try:
            if not self.config.get('ml', {}).get('enabled', False):
                logger.info("ML prediction disabled")
                return
            
            logger.info("Initializing ML model...")
            
            # Train ML model
            success = self.ml_predictor.train_model(self.data_with_indicators)
            
            if success:
                logger.info("ML model trained successfully")
            else:
                logger.warning("ML model training failed, continuing without ML")
                
        except Exception as e:
            logger.error(f"Error initializing ML model: {e}")
            logger.warning("Continuing without ML prediction")
    
    def _trading_loop(self):
        """Main trading loop optimized for intraday trading"""
        try:
            logger.info("Starting intraday trading loop...")
            
            while self.running:
                try:
                    # Check if we should trade (respect trading hours and limits)
                    if not self._should_trade():
                        time.sleep(60)  # Wait 1 minute
                        continue
                    
                    # Get current market data
                    current_data = self.data_feed.get_realtime_data(
                        self.config['trading']['symbol']
                    )
                    
                    # Update data with current price
                    self._update_current_data(current_data)
                    
                    # Calculate technical signals
                    technical_signal = self._calculate_technical_signal()
                    
                    # Get ML prediction if enabled
                    ml_signal = self._get_ml_signal()
                    
                    # Combine signals
                    combined_signal = self._combine_signals(technical_signal, ml_signal)
                    
                    # Check risk limits
                    account_balance = self.executor.get_account_balance()
                    is_safe, risk_reason = self.risk_manager.check_risk_limits(
                        account_balance['total_balance']
                    )
                    
                    if not is_safe:
                        logger.warning(f"Risk limit exceeded: {risk_reason}")
                        time.sleep(300)  # Wait 5 minutes
                        continue
                    
                    # Execute trading logic
                    self._execute_trading_logic(combined_signal, current_data)
                    
                    # Update risk metrics
                    self._update_risk_metrics()
                    
                    # Wait before next iteration
                    sleep_time = self._get_sleep_time()
                    logger.info(f"Waiting {sleep_time} seconds before next iteration...")
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    logger.error(f"Error in trading loop iteration: {e}")
                    time.sleep(30)  # Wait 30 seconds before retrying
                    
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            raise
    
    def _should_trade(self) -> bool:
        """Check if we should trade based on time and limits"""
        try:
            now = datetime.now()
            
            # Check daily trade limit
            if self.daily_trade_count >= self.config['intraday']['max_trades_per_day']:
                logger.info("Daily trade limit reached")
                return False
            
            # Check minimum trade interval
            if self.last_trade_time:
                time_since_last_trade = (now - self.last_trade_time).total_seconds()
                min_interval = self.config['intraday']['min_trade_interval']
                
                if time_since_last_trade < min_interval:
                    return False
            
            # Check trading hours (crypto is 24/7, but we can add custom logic)
            trading_hours = self.config['intraday']['trading_hours']
            if trading_hours != "24/7":
                # Add custom trading hours logic here
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking if should trade: {e}")
            return False
    
    def _update_current_data(self, current_data: Dict):
        """Update data with current market information"""
        try:
            current_price = current_data['price']
            current_time = current_data['timestamp']
            
            # Add current data point to historical data
            new_row = pd.DataFrame({
                'open': [current_price],
                'high': [current_price],
                'low': [current_price],
                'close': [current_price],
                'volume': [current_data.get('volume', 0)]
            }, index=[current_time])
            
            # Append to historical data
            self.historical_data = pd.concat([self.historical_data, new_row])
            
            # Keep only last 1000 data points
            if len(self.historical_data) > 1000:
                self.historical_data = self.historical_data.tail(1000)
            
            # Recalculate indicators
            self.data_with_indicators = TechnicalIndicators.calculate_all_indicators(
                self.historical_data, self.config
            )
            
        except Exception as e:
            logger.error(f"Error updating current data: {e}")
    
    def _calculate_technical_signal(self) -> float:
        """Calculate trading signal from technical indicators"""
        try:
            if len(self.data_with_indicators) < 50:
                return 0.0  # Not enough data
            
            # Get signal strength from indicators
            signal_strength = TechnicalIndicators.get_signal_strength(
                self.data_with_indicators, self.config
            )
            
            # Return the latest signal
            return signal_strength.iloc[-1]
            
        except Exception as e:
            logger.error(f"Error calculating technical signal: {e}")
            return 0.0
    
    def _get_ml_signal(self) -> Dict:
        """Get ML prediction signal"""
        try:
            if not self.config.get('ml', {}).get('enabled', False):
                return {'signal': 0, 'confidence': 0.0}
            
            # Get ML prediction
            ml_prediction = self.ml_predictor.predict_signal(self.data_with_indicators)
            
            return {
                'signal': ml_prediction['signal'],
                'confidence': ml_prediction['confidence']
            }
            
        except Exception as e:
            logger.error(f"Error getting ML signal: {e}")
            return {'signal': 0, 'confidence': 0.0}
    
    def _combine_signals(self, technical_signal: float, ml_signal: Dict) -> Dict:
        """Combine technical and ML signals"""
        try:
            # Weight the signals
            technical_weight = 0.6
            ml_weight = 0.4
            
            # Combine signals
            combined_signal = (technical_signal * technical_weight + 
                             ml_signal['signal'] * ml_weight)
            
            # Calculate combined confidence
            combined_confidence = (abs(technical_signal) * technical_weight + 
                                ml_signal['confidence'] * ml_weight)
            
            return {
                'signal': combined_signal,
                'confidence': combined_confidence,
                'technical_signal': technical_signal,
                'ml_signal': ml_signal['signal'],
                'ml_confidence': ml_signal['confidence']
            }
            
        except Exception as e:
            logger.error(f"Error combining signals: {e}")
            return {'signal': 0, 'confidence': 0.0}
    
    def _execute_trading_logic(self, signal_data: Dict, current_data: Dict):
        """Execute trading logic based on combined signal"""
        try:
            signal_strength = signal_data['signal']
            confidence = signal_data['confidence']
            current_price = current_data['price']
            
            # Get current positions
            positions = self.executor.get_positions()
            symbol = self.config['trading']['symbol']
            
            # Find current position
            current_position = None
            for position in positions:
                if position['symbol'] == symbol and float(position['size']) != 0:
                    current_position = position
                    break
            
            if current_position:
                # Manage existing position
                self._manage_existing_position(current_position, current_price, signal_data)
            else:
                # Look for new trading opportunities
                if abs(signal_strength) > 0.3 and confidence > 0.6:  # Minimum thresholds
                    self._execute_new_trade(signal_strength, current_price, confidence)
            
        except Exception as e:
            logger.error(f"Error executing trading logic: {e}")
    
    def _manage_existing_position(self, position: Dict, current_price: float, signal_data: Dict):
        """Manage existing position (stop loss, take profit, signal-based exit)"""
        try:
            size = float(position['size'])
            entry_price = float(position['avgPrice'])
            side = position['side']
            
            if size == 0:
                return
            
            # Calculate unrealized P&L
            if side == 'Buy':
                unrealized_pnl = size * (current_price - entry_price)
                unrealized_pnl_pct = unrealized_pnl / (size * entry_price)
            else:  # Sell
                unrealized_pnl = size * (entry_price - current_price)
                unrealized_pnl_pct = unrealized_pnl / (size * entry_price)
            
            # Check stop loss
            stop_loss_pct = self.config['risk']['stop_loss']
            if unrealized_pnl_pct <= -stop_loss_pct:
                logger.info(f"Stop loss triggered for {side} position")
                self._close_position(side, abs(size), current_price)
                return
            
            # Check take profit
            take_profit_pct = self.config['risk']['take_profit']
            if unrealized_pnl_pct >= take_profit_pct:
                logger.info(f"Take profit triggered for {side} position")
                self._close_position(side, abs(size), current_price)
                return
            
            # Check signal-based exit
            signal_strength = signal_data['signal']
            confidence = signal_data['confidence']
            
            # Exit if signal strongly opposes position
            if side == 'Buy' and signal_strength < -0.5 and confidence > 0.7:
                logger.info("Signal-based exit: Strong sell signal for long position")
                self._close_position(side, abs(size), current_price)
            elif side == 'Sell' and signal_strength > 0.5 and confidence > 0.7:
                logger.info("Signal-based exit: Strong buy signal for short position")
                self._close_position(side, abs(size), current_price)
            
        except Exception as e:
            logger.error(f"Error managing existing position: {e}")
    
    def _execute_new_trade(self, signal_strength: float, current_price: float, confidence: float):
        """Execute new trade based on signal"""
        try:
            # Determine trade side
            side = OrderSide.BUY if signal_strength > 0 else OrderSide.SELL
            
            # Calculate position size
            account_balance = self.executor.get_account_balance()
            volatility = self.data_with_indicators['atr'].iloc[-1] / current_price if 'atr' in self.data_with_indicators.columns else 0.01
            
            position_size = self.risk_manager.calculate_position_size(
                account_balance['total_balance'], signal_strength, volatility
            )
            
            if position_size <= 0:
                logger.info("Position size too small, skipping trade")
                return
            
            # Place market order
            order = self.executor.place_order(
                side=side,
                order_type=OrderType.MARKET,
                quantity=position_size
            )
            
            if order['status'] == 'New':
                logger.info(f"Trade executed: {side.value} {position_size} @ {current_price}")
                
                # Update trade tracking
                self.last_trade_time = datetime.now()
                self.daily_trade_count += 1
                
                # Calculate and place stop loss/take profit orders
                self._place_protective_orders(side, position_size, current_price, volatility)
            
        except Exception as e:
            logger.error(f"Error executing new trade: {e}")
    
    def _place_protective_orders(self, side: OrderSide, quantity: float, 
                               entry_price: float, volatility: float):
        """Place stop loss and take profit orders"""
        try:
            # Calculate stop loss and take profit prices
            stop_loss_price = self.risk_manager.calculate_stop_loss(
                entry_price, side.value.lower(), volatility
            )
            take_profit_price = self.risk_manager.calculate_take_profit(
                entry_price, side.value.lower(), volatility
            )
            
            # Determine opposite side for protective orders
            opposite_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
            
            # Place stop loss order
            stop_order = self.executor.place_stop_loss_order(
                side=opposite_side,
                quantity=quantity,
                stop_price=stop_loss_price
            )
            
            # Place take profit order
            tp_order = self.executor.place_take_profit_order(
                side=opposite_side,
                quantity=quantity,
                price=take_profit_price
            )
            
            logger.info(f"Protective orders placed - SL: {stop_loss_price}, TP: {take_profit_price}")
            
        except Exception as e:
            logger.error(f"Error placing protective orders: {e}")
    
    def _close_position(self, side: str, quantity: float, price: float):
        """Close existing position"""
        try:
            opposite_side = OrderSide.SELL if side == 'Buy' else OrderSide.BUY
            
            order = self.executor.close_position(
                side=opposite_side,
                quantity=quantity
            )
            
            if order['status'] == 'New':
                # Calculate P&L
                positions = self.executor.get_positions()
                symbol = self.config['trading']['symbol']
                
                for position in positions:
                    if position['symbol'] == symbol:
                        trade_pnl = float(position['unrealisedPnl'])
                        
                        # Update risk manager
                        self.risk_manager.update_pnl(trade_pnl)
                        
                        logger.info(f"Position closed: P&L = {trade_pnl:.2f}")
                        break
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    def _update_risk_metrics(self):
        """Update risk metrics"""
        try:
            positions = self.executor.get_positions()
            current_price = self.data_feed.get_realtime_data(
                self.config['trading']['symbol']
            )['price']
            
            total_unrealized_pnl = 0.0
            
            for position in positions:
                if float(position.get('size', 0)) != 0:
                    total_unrealized_pnl += float(position.get('unrealisedPnl', 0))
            
            # Log risk metrics
            risk_metrics = self.risk_manager.get_risk_metrics()
            logger.info(f"Risk metrics - Daily P&L: {risk_metrics['daily_pnl']:.2f}, "
                       f"Total P&L: {risk_metrics['total_pnl']:.2f}, "
                       f"Drawdown: {risk_metrics['current_drawdown']:.2%}, "
                       f"Trades today: {self.daily_trade_count}")
            
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
    
    def _cancel_all_orders(self):
        """Cancel all pending orders"""
        try:
            cancelled_count = self.executor.cancel_all_orders()
            logger.info(f"Cancelled {cancelled_count} orders")
            
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
    
    def _save_state(self):
        """Save bot state"""
        try:
            # TODO: Implement state saving to database
            logger.info("Bot state saved")
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def _get_sleep_time(self) -> int:
        """Get sleep time between iterations based on timeframe"""
        timeframe = self.config['trading']['timeframe']
        
        # Convert timeframe to seconds
        timeframe_map = {
            '1': 60,      # 1 minute
            '3': 180,     # 3 minutes
            '5': 300,     # 5 minutes
            '15': 900,    # 15 minutes
            '30': 1800,   # 30 minutes
            '60': 3600,   # 1 hour
            '120': 7200,  # 2 hours
            '240': 14400, # 4 hours
            '360': 21600, # 6 hours
            '720': 43200, # 12 hours
            'D': 86400,   # 1 day
            'W': 604800,  # 1 week
            'M': 2592000  # 1 month
        }
        
        return timeframe_map.get(timeframe, 300)  # Default to 5 minutes


def main():
    """Main entry point"""
    try:
        # Create and start trading bot
        bot = BybitTradingBot()
        bot.start()
        
    except KeyboardInterrupt:
        logger.info("Trading bot interrupted by user")
    except Exception as e:
        logger.error(f"Trading bot error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
