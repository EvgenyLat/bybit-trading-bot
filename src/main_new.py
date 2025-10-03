"""
Bybit Trading Bot - Main Application
Professional algorithmic trading bot with ML integration
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional
import pandas as pd
import numpy as np
import yaml
import os
from dotenv import load_dotenv

# Import services
from services.data_collector import DataCollector
from services.feature_engineering import FeatureBuilder
from services.model_training import ModelTrainer
from services.signal_service import SignalGenerator
from services.risk_manager import RiskManager
from services.executor import OrderExecutor
from services.backtester import Backtester

# Import monitoring
from monitoring.telegram_bot import TelegramBot
from monitoring.metrics_collector import MetricsCollector

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class BybitTradingBot:
    """Main trading bot application"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.running = False
        
        # Initialize services
        self.data_collector = None
        self.feature_builder = None
        self.model_trainer = None
        self.signal_generator = None
        self.risk_manager = None
        self.order_executor = None
        self.backtester = None
        
        # Initialize monitoring
        self.telegram_bot = None
        self.metrics_collector = None
        
        # Trading state
        self.current_positions = {}
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_trade_time = None
        
        logger.info("Bybit Trading Bot initialized")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    async def initialize(self):
        """Initialize all services"""
        try:
            logger.info("Initializing services...")
            
            # Initialize data collector
            self.data_collector = DataCollector(self.config_path)
            
            # Initialize feature builder
            self.feature_builder = FeatureBuilder(self.config)
            
            # Initialize model trainer
            self.model_trainer = ModelTrainer(self.config)
            
            # Initialize signal generator
            self.signal_generator = SignalGenerator(self.config)
            
            # Initialize risk manager
            self.risk_manager = RiskManager(self.config)
            
            # Initialize order executor
            self.order_executor = OrderExecutor(self.config)
            
            # Initialize backtester
            self.backtester = Backtester(self.config)
            
            # Initialize monitoring
            self.telegram_bot = TelegramBot(self.config)
            self.metrics_collector = MetricsCollector(self.config)
            
            # Download historical data if needed
            await self._download_historical_data()
            
            # Train initial model
            await self._train_initial_model()
            
            logger.info("All services initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing services: {e}")
            raise
    
    async def _download_historical_data(self):
        """Download historical data for training"""
        try:
            symbol = self.config['trading']['symbol']
            timeframe = self.config['trading']['timeframe']
            days = self.config.get('ml', {}).get('training_data_days', 365)
            
            logger.info(f"Downloading {days} days of historical data...")
            await self.data_collector.download_historical_data(symbol, timeframe, days)
            
        except Exception as e:
            logger.error(f"Error downloading historical data: {e}")
            raise
    
    async def _train_initial_model(self):
        """Train initial ML model"""
        try:
            if not self.config.get('ml', {}).get('enabled', False):
                logger.info("ML training disabled")
                return
            
            logger.info("Training initial ML model...")
            
            # Get historical data
            symbol = self.config['trading']['symbol']
            timeframe = self.config['trading']['timeframe']
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config.get('ml', {}).get('training_data_days', 365))
            
            historical_data = self.data_collector.get_historical_data(
                symbol, timeframe, start_date, end_date
            )
            
            # Create features
            features_df = self.feature_builder.create_all_features(historical_data)
            
            # Train model
            await self.model_trainer.train_model(features_df)
            
            logger.info("Initial model training completed")
            
        except Exception as e:
            logger.error(f"Error training initial model: {e}")
            raise
    
    async def start(self):
        """Start the trading bot"""
        try:
            logger.info("Starting Bybit Trading Bot...")
            
            # Initialize services
            await self.initialize()
            
            # Start real-time data collection
            self.data_collector.start_real_time_collection()
            
            # Start monitoring
            await self.telegram_bot.start()
            self.metrics_collector.start()
            
            # Send startup notification
            await self.telegram_bot.send_message("🚀 Bybit Trading Bot started successfully!")
            
            # Set running flag
            self.running = True
            
            # Start main trading loop
            await self._trading_loop()
            
        except Exception as e:
            logger.error(f"Error starting trading bot: {e}")
            await self.telegram_bot.send_message(f"❌ Error starting bot: {e}")
            raise
    
    async def stop(self):
        """Stop the trading bot"""
        try:
            logger.info("Stopping Bybit Trading Bot...")
            
            self.running = False
            
            # Stop data collection
            if self.data_collector:
                self.data_collector.stop_real_time_collection()
            
            # Close all positions
            if self.order_executor:
                await self.order_executor.close_all_positions()
            
            # Stop monitoring
            if self.metrics_collector:
                self.metrics_collector.stop()
            
            # Send shutdown notification
            if self.telegram_bot:
                await self.telegram_bot.send_message("🛑 Bybit Trading Bot stopped")
            
            logger.info("Bybit Trading Bot stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping trading bot: {e}")
    
    async def _trading_loop(self):
        """Main trading loop"""
        try:
            logger.info("Starting main trading loop...")
            
            while self.running:
                try:
                    # Check if we should trade
                    if not self._should_trade():
                        await asyncio.sleep(60)  # Wait 1 minute
                        continue
                    
                    # Get latest data
                    latest_data = await self._get_latest_data()
                    
                    # Create features
                    features_df = self.feature_builder.create_all_features(latest_data)
                    
                    # Generate signals
                    technical_signal = self.signal_generator.generate_technical_signal(features_df)
                    ml_signal = await self.signal_generator.generate_ml_signal(features_df)
                    
                    # Combine signals
                    combined_signal = self.signal_generator.combine_signals(technical_signal, ml_signal)
                    
                    # Risk management check
                    risk_check = await self.risk_manager.check_risk_limits(combined_signal)
                    
                    if risk_check['approved']:
                        # Execute trade
                        await self._execute_trade(combined_signal, risk_check)
                    
                    # Update metrics
                    self.metrics_collector.update_metrics({
                        'equity_usd': await self.order_executor.get_account_balance(),
                        'open_positions': len(self.current_positions),
                        'daily_pnl': self.daily_pnl,
                        'daily_trades': self.daily_trades
                    })
                    
                    # Sleep based on timeframe
                    sleep_time = self._get_sleep_time()
                    await asyncio.sleep(sleep_time)
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    await self.telegram_bot.send_message(f"⚠️ Trading loop error: {e}")
                    await asyncio.sleep(60)  # Wait before retrying
            
        except Exception as e:
            logger.error(f"Error in main trading loop: {e}")
            raise
    
    def _should_trade(self) -> bool:
        """Check if we should trade based on various conditions"""
        try:
            # Check trading hours
            current_hour = datetime.now().hour
            trading_hours = self.config.get('intraday', {}).get('trading_hours', '24/7')
            
            if trading_hours != '24/7':
                start_hour = int(self.config['intraday']['session_start'].split(':')[0])
                end_hour = int(self.config['intraday']['session_end'].split(':')[0])
                
                if not (start_hour <= current_hour <= end_hour):
                    return False
            
            # Check daily trade limit
            max_trades = self.config.get('intraday', {}).get('max_trades_per_day', 50)
            if self.daily_trades >= max_trades:
                logger.info("Daily trade limit reached")
                return False
            
            # Check minimum trade interval
            min_interval = self.config.get('intraday', {}).get('min_trade_interval', 300)
            if self.last_trade_time:
                time_since_last = (datetime.now() - self.last_trade_time).total_seconds()
                if time_since_last < min_interval:
                    return False
            
            # Check if we have open positions
            max_positions = self.config.get('trading', {}).get('max_positions', 2)
            if len(self.current_positions) >= max_positions:
                logger.info("Maximum positions reached")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking trade conditions: {e}")
            return False
    
    async def _get_latest_data(self) -> pd.DataFrame:
        """Get latest market data"""
        try:
            symbol = self.config['trading']['symbol']
            timeframe = self.config['trading']['timeframe']
            
            # Get latest data from storage
            latest_data = self.data_collector.get_latest_data(symbol, timeframe, 1000)
            
            return latest_data
            
        except Exception as e:
            logger.error(f"Error getting latest data: {e}")
            raise
    
    async def _execute_trade(self, signal: Dict, risk_check: Dict):
        """Execute a trade based on signal and risk check"""
        try:
            symbol = self.config['trading']['symbol']
            side = signal['side']
            confidence = signal['confidence']
            
            # Calculate position size
            position_size = risk_check['position_size']
            
            # Place order
            order_result = await self.order_executor.place_order(
                symbol=symbol,
                side=side,
                quantity=position_size,
                order_type='Market'
            )
            
            if order_result['success']:
                # Update position tracking
                self.current_positions[order_result['order_id']] = {
                    'symbol': symbol,
                    'side': side,
                    'quantity': position_size,
                    'entry_price': order_result['price'],
                    'timestamp': datetime.now(),
                    'signal_confidence': confidence
                }
                
                # Update counters
                self.daily_trades += 1
                self.last_trade_time = datetime.now()
                
                # Place protective orders
                await self._place_protective_orders(order_result['order_id'])
                
                # Send notification
                await self.telegram_bot.send_message(
                    f"✅ Trade executed: {side} {position_size} {symbol} "
                    f"at {order_result['price']} (confidence: {confidence:.2f})"
                )
                
                logger.info(f"Trade executed: {side} {position_size} {symbol}")
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            await self.telegram_bot.send_message(f"❌ Trade execution error: {e}")
    
    async def _place_protective_orders(self, order_id: str):
        """Place stop-loss and take-profit orders"""
        try:
            position = self.current_positions.get(order_id)
            if not position:
                return
            
            symbol = position['symbol']
            side = position['side']
            quantity = position['quantity']
            entry_price = position['entry_price']
            
            # Calculate stop-loss and take-profit prices
            stop_loss_pct = self.config.get('risk', {}).get('stop_loss', 0.015)
            take_profit_pct = self.config.get('risk', {}).get('take_profit', 0.03)
            
            if side == 'Buy':
                stop_loss_price = entry_price * (1 - stop_loss_pct)
                take_profit_price = entry_price * (1 + take_profit_pct)
            else:
                stop_loss_price = entry_price * (1 + stop_loss_pct)
                take_profit_price = entry_price * (1 - take_profit_pct)
            
            # Place stop-loss order
            await self.order_executor.place_stop_loss_order(
                symbol=symbol,
                side='Sell' if side == 'Buy' else 'Buy',
                quantity=quantity,
                stop_price=stop_loss_price
            )
            
            # Place take-profit order
            await self.order_executor.place_take_profit_order(
                symbol=symbol,
                side='Sell' if side == 'Buy' else 'Buy',
                quantity=quantity,
                limit_price=take_profit_price
            )
            
            logger.info(f"Protective orders placed for {order_id}")
            
        except Exception as e:
            logger.error(f"Error placing protective orders: {e}")
    
    def _get_sleep_time(self) -> int:
        """Get sleep time based on timeframe"""
        timeframe = self.config['trading']['timeframe']
        
        # Convert timeframe to seconds
        timeframe_map = {
            '1': 60, '3': 180, '5': 300, '15': 900, '30': 1800,
            '60': 3600, '120': 7200, '240': 14400, '360': 21600,
            '720': 43200, 'D': 86400, 'W': 604800, 'M': 2592000
        }
        
        return timeframe_map.get(timeframe, 60)


async def main():
    """Main function"""
    bot = BybitTradingBot()
    
    # Set up signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(bot.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())

