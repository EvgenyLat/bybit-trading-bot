"""
Risk Management Module for Trading Bot
Handles position sizing, stop losses, and risk controls
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class RiskManager:
    """Handles risk management and position sizing"""
    
    def __init__(self, config: Dict):
        """
        Initialize RiskManager with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.stop_loss = config['risk']['stop_loss']
        self.take_profit = config['risk']['take_profit']
        self.max_daily_loss = config['risk']['max_daily_loss']
        self.max_drawdown = config['risk']['max_drawdown']
        
        # Risk tracking
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.max_portfolio_value = 0.0
        self.current_drawdown = 0.0
        self.trade_history = []
        
        # Risk limits
        self.daily_loss_limit = None
        self.drawdown_limit = None
        
    def calculate_position_size(self, account_balance: float, 
                              signal_strength: float, 
                              volatility: float) -> float:
        """
        Calculate position size based on risk parameters
        
        Args:
            account_balance: Current account balance
            signal_strength: Signal strength (-1 to 1)
            volatility: Market volatility (ATR or similar)
            
        Returns:
            Position size in base currency
        """
        try:
            # Base position size from config
            base_size = self.config['trading']['position_size']
            
            # Adjust for signal strength
            signal_multiplier = abs(signal_strength)
            
            # Adjust for volatility (higher volatility = smaller position)
            volatility_multiplier = 1.0 / (1.0 + volatility)
            
            # Calculate position size
            position_size = account_balance * base_size * signal_multiplier * volatility_multiplier
            
            # Apply risk limits
            position_size = self._apply_risk_limits(position_size, account_balance)
            
            logger.info(f"Calculated position size: {position_size:.2f}")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def _apply_risk_limits(self, position_size: float, account_balance: float) -> float:
        """Apply risk limits to position size"""
        try:
            # Maximum position size (e.g., 10% of account)
            max_position_size = account_balance * 0.1
            position_size = min(position_size, max_position_size)
            
            # Check daily loss limit
            if self.daily_pnl < -self.max_daily_loss * account_balance:
                logger.warning("Daily loss limit reached, reducing position size")
                position_size *= 0.5
            
            # Check drawdown limit
            if self.current_drawdown > self.max_drawdown:
                logger.warning("Drawdown limit reached, reducing position size")
                position_size *= 0.3
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error applying risk limits: {e}")
            return position_size
    
    def calculate_stop_loss(self, entry_price: float, side: str, 
                          volatility: float) -> float:
        """
        Calculate stop loss price
        
        Args:
            entry_price: Entry price
            side: Trade side ('buy' or 'sell')
            volatility: Market volatility
            
        Returns:
            Stop loss price
        """
        try:
            # Base stop loss from config
            base_stop_loss = self.stop_loss
            
            # Adjust for volatility
            volatility_adjustment = volatility * 0.5
            adjusted_stop_loss = base_stop_loss + volatility_adjustment
            
            # Calculate stop loss price
            if side == 'buy':
                stop_loss_price = entry_price * (1 - adjusted_stop_loss)
            else:  # sell
                stop_loss_price = entry_price * (1 + adjusted_stop_loss)
            
            logger.info(f"Calculated stop loss: {stop_loss_price:.2f}")
            return stop_loss_price
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            return entry_price
    
    def calculate_take_profit(self, entry_price: float, side: str, 
                            volatility: float) -> float:
        """
        Calculate take profit price
        
        Args:
            entry_price: Entry price
            side: Trade side ('buy' or 'sell')
            volatility: Market volatility
            
        Returns:
            Take profit price
        """
        try:
            # Base take profit from config
            base_take_profit = self.take_profit
            
            # Adjust for volatility
            volatility_adjustment = volatility * 0.3
            adjusted_take_profit = base_take_profit + volatility_adjustment
            
            # Calculate take profit price
            if side == 'buy':
                take_profit_price = entry_price * (1 + adjusted_take_profit)
            else:  # sell
                take_profit_price = entry_price * (1 - adjusted_take_profit)
            
            logger.info(f"Calculated take profit: {take_profit_price:.2f}")
            return take_profit_price
            
        except Exception as e:
            logger.error(f"Error calculating take profit: {e}")
            return entry_price
    
    def check_risk_limits(self, account_balance: float) -> Tuple[bool, str]:
        """
        Check if risk limits are exceeded
        
        Args:
            account_balance: Current account balance
            
        Returns:
            Tuple of (is_safe, reason)
        """
        try:
            # Check daily loss limit
            daily_loss_threshold = self.max_daily_loss * account_balance
            if self.daily_pnl < -daily_loss_threshold:
                return False, f"Daily loss limit exceeded: {self.daily_pnl:.2f}"
            
            # Check drawdown limit
            if self.current_drawdown > self.max_drawdown:
                return False, f"Drawdown limit exceeded: {self.current_drawdown:.2%}"
            
            # Check if account balance is too low
            min_balance = account_balance * 0.1  # 10% of original balance
            if account_balance < min_balance:
                return False, f"Account balance too low: {account_balance:.2f}"
            
            return True, "Risk limits OK"
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False, f"Error checking risk limits: {e}"
    
    def update_pnl(self, trade_pnl: float, trade_type: str = "trade"):
        """
        Update P&L tracking
        
        Args:
            trade_pnl: P&L from trade
            trade_type: Type of trade/update
        """
        try:
            self.daily_pnl += trade_pnl
            self.total_pnl += trade_pnl
            
            # Update max portfolio value for drawdown calculation
            if self.total_pnl > 0:
                self.max_portfolio_value = max(self.max_portfolio_value, self.total_pnl)
            
            # Calculate current drawdown
            if self.max_portfolio_value > 0:
                self.current_drawdown = (self.max_portfolio_value - self.total_pnl) / self.max_portfolio_value
            
            # Record trade
            trade_record = {
                'timestamp': datetime.now(),
                'type': trade_type,
                'pnl': trade_pnl,
                'daily_pnl': self.daily_pnl,
                'total_pnl': self.total_pnl,
                'drawdown': self.current_drawdown
            }
            self.trade_history.append(trade_record)
            
            logger.info(f"P&L updated: {trade_pnl:.2f}, Total: {self.total_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating P&L: {e}")
    
    def reset_daily_tracking(self):
        """Reset daily tracking (call at start of new trading day)"""
        try:
            self.daily_pnl = 0.0
            logger.info("Daily tracking reset")
            
        except Exception as e:
            logger.error(f"Error resetting daily tracking: {e}")
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        return {
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'daily_loss_limit': self.max_daily_loss,
            'max_portfolio_value': self.max_portfolio_value,
            'trade_count': len(self.trade_history)
        }
    
    def calculate_var(self, returns: List[float], confidence_level: float = 0.05) -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: List of historical returns
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
            
        Returns:
            VaR value
        """
        try:
            if not returns:
                return 0.0
            
            returns_array = np.array(returns)
            var = np.percentile(returns_array, confidence_level * 100)
            
            logger.info(f"VaR calculated: {var:.4f}")
            return var
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0
    
    def calculate_sharpe_ratio(self, returns: List[float], 
                             risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: List of historical returns
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Sharpe ratio
        """
        try:
            if not returns or len(returns) < 2:
                return 0.0
            
            returns_array = np.array(returns)
            excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate
            
            if np.std(excess_returns) == 0:
                return 0.0
            
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            
            logger.info(f"Sharpe ratio calculated: {sharpe_ratio:.4f}")
            return sharpe_ratio
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def get_position_risk(self, position: Dict, current_price: float) -> Dict:
        """
        Calculate risk metrics for a position
        
        Args:
            position: Position dictionary
            current_price: Current market price
            
        Returns:
            Risk metrics dictionary
        """
        try:
            quantity = position['quantity']
            average_price = position['average_price']
            
            if quantity == 0:
                return {
                    'unrealized_pnl': 0.0,
                    'unrealized_pnl_pct': 0.0,
                    'risk_amount': 0.0,
                    'risk_pct': 0.0
                }
            
            # Calculate unrealized P&L
            unrealized_pnl = quantity * (current_price - average_price)
            unrealized_pnl_pct = unrealized_pnl / (quantity * average_price)
            
            # Calculate risk amount (distance to stop loss)
            if quantity > 0:  # Long position
                risk_amount = quantity * (average_price - current_price * (1 - self.stop_loss))
            else:  # Short position
                risk_amount = abs(quantity) * (current_price * (1 + self.stop_loss) - average_price)
            
            risk_pct = risk_amount / (abs(quantity) * average_price)
            
            return {
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': unrealized_pnl_pct,
                'risk_amount': risk_amount,
                'risk_pct': risk_pct
            }
            
        except Exception as e:
            logger.error(f"Error calculating position risk: {e}")
            return {}

