"""
Enhanced Risk Management System
Advanced risk controls, position sizing, and emergency procedures
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from collections import deque
import json

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PositionStatus(Enum):
    """Position status"""
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"
    PENDING = "pending"


@dataclass
class Position:
    """Position information"""
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    timestamp: datetime
    status: PositionStatus
    leverage: float
    margin_used: float


@dataclass
class RiskMetrics:
    """Risk metrics container"""
    total_equity: float
    available_margin: float
    used_margin: float
    margin_ratio: float
    daily_pnl: float
    weekly_pnl: float
    monthly_pnl: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    max_consecutive_losses: int
    current_consecutive_losses: int


class AdvancedRiskManager:
    """Advanced risk management system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.risk_config = config.get('risk', {})
        
        # Risk limits
        self.max_daily_loss = self.risk_config.get('max_daily_loss', 0.03)
        self.max_weekly_loss = self.risk_config.get('max_weekly_loss', 0.07)
        self.max_monthly_loss = self.risk_config.get('max_monthly_loss', 0.15)
        self.max_drawdown = self.risk_config.get('max_drawdown', 0.15)
        self.max_position_size = self.risk_config.get('max_position_size', 0.05)
        self.max_concurrent_positions = self.risk_config.get('max_concurrent_positions', 1)
        self.max_leverage = self.risk_config.get('max_leverage', 2)
        
        # Emergency controls
        self.emergency_stop_enabled = self.risk_config.get('emergency_stop_enabled', True)
        self.emergency_stop_threshold = self.risk_config.get('emergency_stop_threshold', 0.05)
        
        # Risk tracking
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.daily_pnl_history = deque(maxlen=30)  # Last 30 days
        self.equity_history = deque(maxlen=1000)  # Last 1000 data points
        self.trade_history = deque(maxlen=1000)
        
        # Risk state
        self.risk_level = RiskLevel.LOW
        self.emergency_stop_active = False
        self.last_risk_check = datetime.now()
        
        # Position sizing models
        self.position_sizing_models = {
            'fixed': self._fixed_position_sizing,
            'kelly': self._kelly_position_sizing,
            'volatility_adjusted': self._volatility_adjusted_sizing,
            'risk_parity': self._risk_parity_sizing
        }
        
        logger.info("Advanced Risk Manager initialized")
    
    async def check_risk_limits(self, signal: Dict, account_balance: float) -> Dict[str, Any]:
        """Comprehensive risk check before trade execution"""
        try:
            # Update risk metrics
            risk_metrics = await self._calculate_risk_metrics(account_balance)
            
            # Check emergency stop
            if self.emergency_stop_active:
                return {
                    'approved': False,
                    'reason': 'Emergency stop is active',
                    'risk_level': RiskLevel.CRITICAL,
                    'metrics': risk_metrics
                }
            
            # Check daily loss limit
            if risk_metrics.daily_pnl < -self.max_daily_loss * account_balance:
                await self._trigger_emergency_stop("Daily loss limit exceeded")
                return {
                    'approved': False,
                    'reason': 'Daily loss limit exceeded',
                    'risk_level': RiskLevel.CRITICAL,
                    'metrics': risk_metrics
                }
            
            # Check weekly loss limit
            if risk_metrics.weekly_pnl < -self.max_weekly_loss * account_balance:
                await self._trigger_emergency_stop("Weekly loss limit exceeded")
                return {
                    'approved': False,
                    'reason': 'Weekly loss limit exceeded',
                    'risk_level': RiskLevel.CRITICAL,
                    'metrics': risk_metrics
                }
            
            # Check monthly loss limit
            if risk_metrics.monthly_pnl < -self.max_monthly_loss * account_balance:
                await self._trigger_emergency_stop("Monthly loss limit exceeded")
                return {
                    'approved': False,
                    'reason': 'Monthly loss limit exceeded',
                    'risk_level': RiskLevel.CRITICAL,
                    'metrics': risk_metrics
                }
            
            # Check drawdown limit
            if risk_metrics.current_drawdown > self.max_drawdown:
                await self._trigger_emergency_stop("Maximum drawdown exceeded")
                return {
                    'approved': False,
                    'reason': 'Maximum drawdown exceeded',
                    'risk_level': RiskLevel.CRITICAL,
                    'metrics': risk_metrics
                }
            
            # Check concurrent positions limit
            if len(self.positions) >= self.max_concurrent_positions:
                return {
                    'approved': False,
                    'reason': 'Maximum concurrent positions reached',
                    'risk_level': RiskLevel.HIGH,
                    'metrics': risk_metrics
                }
            
            # Check margin requirements
            if risk_metrics.margin_ratio > 0.8:  # 80% margin usage
                return {
                    'approved': False,
                    'reason': 'Margin usage too high',
                    'risk_level': RiskLevel.HIGH,
                    'metrics': risk_metrics
                }
            
            # Check consecutive losses
            if risk_metrics.current_consecutive_losses >= 5:
                return {
                    'approved': False,
                    'reason': 'Too many consecutive losses',
                    'risk_level': RiskLevel.HIGH,
                    'metrics': risk_metrics
                }
            
            # Calculate position size
            position_size = await self._calculate_position_size(
                signal, account_balance, risk_metrics
            )
            
            # Check position size limits
            if position_size > self.max_position_size * account_balance:
                position_size = self.max_position_size * account_balance
                logger.warning("Position size limited by maximum position size rule")
            
            # Determine risk level
            risk_level = self._determine_risk_level(risk_metrics)
            
            return {
                'approved': True,
                'position_size': position_size,
                'risk_level': risk_level,
                'metrics': risk_metrics,
                'stop_loss': self._calculate_stop_loss(signal),
                'take_profit': self._calculate_take_profit(signal)
            }
            
        except Exception as e:
            logger.error(f"Error in risk check: {e}")
            return {
                'approved': False,
                'reason': f'Risk check error: {e}',
                'risk_level': RiskLevel.CRITICAL,
                'metrics': None
            }
    
    async def _calculate_risk_metrics(self, account_balance: float) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            # Calculate daily PnL
            daily_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            
            # Calculate weekly PnL
            week_ago = datetime.now() - timedelta(days=7)
            weekly_pnl = sum(
                pos.realized_pnl for pos in self.closed_positions 
                if pos.timestamp >= week_ago
            ) + daily_pnl
            
            # Calculate monthly PnL
            month_ago = datetime.now() - timedelta(days=30)
            monthly_pnl = sum(
                pos.realized_pnl for pos in self.closed_positions 
                if pos.timestamp >= month_ago
            ) + daily_pnl
            
            # Calculate drawdown
            if self.equity_history:
                peak_equity = max(self.equity_history)
                current_drawdown = (peak_equity - account_balance) / peak_equity
                max_drawdown = max(
                    (peak_equity - equity) / peak_equity 
                    for equity in self.equity_history
                )
            else:
                current_drawdown = 0.0
                max_drawdown = 0.0
            
            # Calculate Sharpe ratio
            if len(self.daily_pnl_history) > 30:
                returns = np.array(list(self.daily_pnl_history))
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            
            # Calculate win rate
            if self.trade_history:
                winning_trades = sum(1 for trade in self.trade_history if trade > 0)
                win_rate = winning_trades / len(self.trade_history)
            else:
                win_rate = 0.0
            
            # Calculate profit factor
            if self.trade_history:
                gross_profit = sum(trade for trade in self.trade_history if trade > 0)
                gross_loss = abs(sum(trade for trade in self.trade_history if trade < 0))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
            else:
                profit_factor = 0.0
            
            # Calculate consecutive losses
            consecutive_losses = 0
            for trade in reversed(self.trade_history):
                if trade < 0:
                    consecutive_losses += 1
                else:
                    break
            
            # Calculate margin metrics
            used_margin = sum(pos.margin_used for pos in self.positions.values())
            available_margin = account_balance - used_margin
            margin_ratio = used_margin / account_balance if account_balance > 0 else 0.0
            
            return RiskMetrics(
                total_equity=account_balance,
                available_margin=available_margin,
                used_margin=used_margin,
                margin_ratio=margin_ratio,
                daily_pnl=daily_pnl,
                weekly_pnl=weekly_pnl,
                monthly_pnl=monthly_pnl,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                sharpe_ratio=sharpe_ratio,
                win_rate=win_rate,
                profit_factor=profit_factor,
                max_consecutive_losses=max(consecutive_losses, 0),
                current_consecutive_losses=consecutive_losses
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return RiskMetrics(
                total_equity=account_balance,
                available_margin=0.0,
                used_margin=0.0,
                margin_ratio=0.0,
                daily_pnl=0.0,
                weekly_pnl=0.0,
                monthly_pnl=0.0,
                max_drawdown=0.0,
                current_drawdown=0.0,
                sharpe_ratio=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                max_consecutive_losses=0,
                current_consecutive_losses=0
            )
    
    async def _calculate_position_size(self, signal: Dict, account_balance: float, 
                                      risk_metrics: RiskMetrics) -> float:
        """Calculate optimal position size using multiple models"""
        try:
            # Get position sizing method from config
            sizing_method = self.config.get('position_sizing_method', 'fixed')
            
            if sizing_method not in self.position_sizing_models:
                sizing_method = 'fixed'
            
            # Calculate base position size
            base_size = self.position_sizing_models[sizing_method](
                signal, account_balance, risk_metrics
            )
            
            # Apply risk adjustments
            adjusted_size = self._apply_risk_adjustments(base_size, risk_metrics)
            
            return max(0.0, adjusted_size)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def _fixed_position_sizing(self, signal: Dict, account_balance: float, 
                              risk_metrics: RiskMetrics) -> float:
        """Fixed percentage position sizing"""
        return account_balance * self.risk_config.get('risk_per_trade', 0.01)
    
    def _kelly_position_sizing(self, signal: Dict, account_balance: float, 
                             risk_metrics: RiskMetrics) -> float:
        """Kelly criterion position sizing"""
        try:
            if risk_metrics.win_rate == 0 or risk_metrics.profit_factor == 0:
                return self._fixed_position_sizing(signal, account_balance, risk_metrics)
            
            # Kelly formula: f = (bp - q) / b
            # where b = profit factor, p = win rate, q = 1 - win rate
            b = risk_metrics.profit_factor
            p = risk_metrics.win_rate
            q = 1 - p
            
            kelly_fraction = (b * p - q) / b
            
            # Cap Kelly at 25% of account
            kelly_fraction = min(kelly_fraction, 0.25)
            
            return account_balance * kelly_fraction
            
        except Exception as e:
            logger.error(f"Error in Kelly position sizing: {e}")
            return self._fixed_position_sizing(signal, account_balance, risk_metrics)
    
    def _volatility_adjusted_sizing(self, signal: Dict, account_balance: float, 
                                  risk_metrics: RiskMetrics) -> float:
        """Volatility-adjusted position sizing"""
        try:
            # Get volatility from signal
            volatility = signal.get('volatility', 0.02)  # Default 2%
            
            # Base position size
            base_size = account_balance * self.risk_config.get('risk_per_trade', 0.01)
            
            # Adjust for volatility (higher volatility = smaller position)
            volatility_multiplier = 1.0 / (1.0 + volatility * 10)
            
            return base_size * volatility_multiplier
            
        except Exception as e:
            logger.error(f"Error in volatility-adjusted sizing: {e}")
            return self._fixed_position_sizing(signal, account_balance, risk_metrics)
    
    def _risk_parity_sizing(self, signal: Dict, account_balance: float, 
                          risk_metrics: RiskMetrics) -> float:
        """Risk parity position sizing"""
        try:
            # Equal risk contribution across positions
            num_positions = len(self.positions) + 1  # Including new position
            equal_risk_contribution = account_balance / num_positions
            
            return equal_risk_contribution
            
        except Exception as e:
            logger.error(f"Error in risk parity sizing: {e}")
            return self._fixed_position_sizing(signal, account_balance, risk_metrics)
    
    def _apply_risk_adjustments(self, base_size: float, risk_metrics: RiskMetrics) -> float:
        """Apply risk-based adjustments to position size"""
        try:
            adjusted_size = base_size
            
            # Reduce size if drawdown is high
            if risk_metrics.current_drawdown > 0.05:  # 5% drawdown
                drawdown_multiplier = 1.0 - (risk_metrics.current_drawdown * 2)
                adjusted_size *= max(0.1, drawdown_multiplier)
            
            # Reduce size if consecutive losses
            if risk_metrics.current_consecutive_losses > 2:
                loss_multiplier = 1.0 - (risk_metrics.current_consecutive_losses * 0.1)
                adjusted_size *= max(0.1, loss_multiplier)
            
            # Reduce size if Sharpe ratio is low
            if risk_metrics.sharpe_ratio < 0.5:
                sharpe_multiplier = max(0.5, risk_metrics.sharpe_ratio)
                adjusted_size *= sharpe_multiplier
            
            return adjusted_size
            
        except Exception as e:
            logger.error(f"Error applying risk adjustments: {e}")
            return base_size
    
    def _calculate_stop_loss(self, signal: Dict) -> Optional[float]:
        """Calculate stop loss level"""
        try:
            current_price = signal.get('price', 0)
            volatility = signal.get('volatility', 0.02)
            
            # Use ATR-based stop loss if available
            if 'atr' in signal:
                atr = signal['atr']
                stop_distance = atr * 2.0  # 2x ATR
            else:
                stop_distance = current_price * volatility * 2.0
            
            # Determine stop loss based on signal direction
            if signal.get('direction') == 'long':
                stop_loss = current_price - stop_distance
            else:
                stop_loss = current_price + stop_distance
            
            return stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            return None
    
    def _calculate_take_profit(self, signal: Dict) -> Optional[float]:
        """Calculate take profit level"""
        try:
            current_price = signal.get('price', 0)
            volatility = signal.get('volatility', 0.02)
            
            # Use risk-reward ratio
            risk_reward_ratio = self.config.get('risk_reward_ratio', 2.0)
            
            # Calculate stop loss distance
            if 'atr' in signal:
                atr = signal['atr']
                stop_distance = atr * 2.0
            else:
                stop_distance = current_price * volatility * 2.0
            
            # Calculate take profit distance
            profit_distance = stop_distance * risk_reward_ratio
            
            # Determine take profit based on signal direction
            if signal.get('direction') == 'long':
                take_profit = current_price + profit_distance
            else:
                take_profit = current_price - profit_distance
            
            return take_profit
            
        except Exception as e:
            logger.error(f"Error calculating take profit: {e}")
            return None
    
    def _determine_risk_level(self, risk_metrics: RiskMetrics) -> RiskLevel:
        """Determine current risk level"""
        try:
            # Check critical conditions
            if (risk_metrics.current_drawdown > 0.1 or 
                risk_metrics.daily_pnl < -0.05 or
                risk_metrics.current_consecutive_losses > 5):
                return RiskLevel.CRITICAL
            
            # Check high risk conditions
            if (risk_metrics.current_drawdown > 0.05 or
                risk_metrics.daily_pnl < -0.02 or
                risk_metrics.current_consecutive_losses > 3):
                return RiskLevel.HIGH
            
            # Check medium risk conditions
            if (risk_metrics.current_drawdown > 0.02 or
                risk_metrics.daily_pnl < -0.01 or
                risk_metrics.current_consecutive_losses > 1):
                return RiskLevel.MEDIUM
            
            return RiskLevel.LOW
            
        except Exception as e:
            logger.error(f"Error determining risk level: {e}")
            return RiskLevel.HIGH
    
    async def _trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop"""
        try:
            self.emergency_stop_active = True
            logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
            
            # Close all positions
            await self._close_all_positions()
            
            # Send emergency alert
            await self._send_emergency_alert(reason)
            
        except Exception as e:
            logger.error(f"Error in emergency stop: {e}")
    
    async def _close_all_positions(self):
        """Close all open positions"""
        try:
            for position_id, position in self.positions.items():
                logger.info(f"Closing position {position_id} due to emergency stop")
                # This would trigger position closure
                # await self.order_manager.close_position(position_id)
            
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
    
    async def _send_emergency_alert(self, reason: str):
        """Send emergency alert"""
        try:
            alert_message = {
                'type': 'emergency_stop',
                'reason': reason,
                'timestamp': datetime.now().isoformat(),
                'positions': len(self.positions),
                'equity': sum(pos.unrealized_pnl for pos in self.positions.values())
            }
            
            # Send to monitoring system
            logger.critical(f"EMERGENCY ALERT: {json.dumps(alert_message)}")
            
        except Exception as e:
            logger.error(f"Error sending emergency alert: {e}")
    
    def update_position(self, position_id: str, position_data: Dict):
        """Update position information"""
        try:
            if position_id in self.positions:
                position = self.positions[position_id]
                position.current_price = position_data.get('current_price', position.current_price)
                position.unrealized_pnl = position_data.get('unrealized_pnl', position.unrealized_pnl)
                position.status = PositionStatus(position_data.get('status', position.status.value))
            else:
                # Create new position
                position = Position(
                    symbol=position_data['symbol'],
                    side=position_data['side'],
                    size=position_data['size'],
                    entry_price=position_data['entry_price'],
                    current_price=position_data['current_price'],
                    unrealized_pnl=position_data.get('unrealized_pnl', 0.0),
                    realized_pnl=position_data.get('realized_pnl', 0.0),
                    stop_loss=position_data.get('stop_loss'),
                    take_profit=position_data.get('take_profit'),
                    timestamp=datetime.now(),
                    status=PositionStatus(position_data.get('status', 'open')),
                    leverage=position_data.get('leverage', 1.0),
                    margin_used=position_data.get('margin_used', 0.0)
                )
                self.positions[position_id] = position
            
        except Exception as e:
            logger.error(f"Error updating position: {e}")
    
    def close_position(self, position_id: str, realized_pnl: float):
        """Close position and update history"""
        try:
            if position_id in self.positions:
                position = self.positions[position_id]
                position.realized_pnl = realized_pnl
                position.status = PositionStatus.CLOSED
                
                # Move to closed positions
                self.closed_positions.append(position)
                del self.positions[position_id]
                
                # Update trade history
                self.trade_history.append(realized_pnl)
                
                logger.info(f"Position {position_id} closed with PnL: {realized_pnl}")
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        try:
            return {
                'risk_level': self.risk_level.value,
                'emergency_stop_active': self.emergency_stop_active,
                'open_positions': len(self.positions),
                'closed_positions': len(self.closed_positions),
                'total_trades': len(self.trade_history),
                'win_rate': self._calculate_win_rate(),
                'profit_factor': self._calculate_profit_factor(),
                'max_drawdown': self._calculate_max_drawdown(),
                'current_drawdown': self._calculate_current_drawdown(),
                'daily_pnl': self._calculate_daily_pnl(),
                'weekly_pnl': self._calculate_weekly_pnl(),
                'monthly_pnl': self._calculate_monthly_pnl(),
                'sharpe_ratio': self._calculate_sharpe_ratio(),
                'consecutive_losses': self._calculate_consecutive_losses(),
                'margin_usage': self._calculate_margin_usage(),
                'last_risk_check': self.last_risk_check.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            return {}
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate"""
        if not self.trade_history:
            return 0.0
        winning_trades = sum(1 for trade in self.trade_history if trade > 0)
        return winning_trades / len(self.trade_history)
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        if not self.trade_history:
            return 0.0
        gross_profit = sum(trade for trade in self.trade_history if trade > 0)
        gross_loss = abs(sum(trade for trade in self.trade_history if trade < 0))
        return gross_profit / gross_loss if gross_loss > 0 else 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.equity_history:
            return 0.0
        peak = self.equity_history[0]
        max_dd = 0.0
        for equity in self.equity_history:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
        return max_dd
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown"""
        if not self.equity_history:
            return 0.0
        peak = max(self.equity_history)
        current_equity = self.equity_history[-1]
        return (peak - current_equity) / peak
    
    def _calculate_daily_pnl(self) -> float:
        """Calculate daily PnL"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    def _calculate_weekly_pnl(self) -> float:
        """Calculate weekly PnL"""
        week_ago = datetime.now() - timedelta(days=7)
        weekly_realized = sum(
            pos.realized_pnl for pos in self.closed_positions 
            if pos.timestamp >= week_ago
        )
        return weekly_realized + self._calculate_daily_pnl()
    
    def _calculate_monthly_pnl(self) -> float:
        """Calculate monthly PnL"""
        month_ago = datetime.now() - timedelta(days=30)
        monthly_realized = sum(
            pos.realized_pnl for pos in self.closed_positions 
            if pos.timestamp >= month_ago
        )
        return monthly_realized + self._calculate_daily_pnl()
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        if len(self.daily_pnl_history) < 30:
            return 0.0
        returns = np.array(list(self.daily_pnl_history))
        return np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    def _calculate_consecutive_losses(self) -> int:
        """Calculate consecutive losses"""
        consecutive = 0
        for trade in reversed(self.trade_history):
            if trade < 0:
                consecutive += 1
            else:
                break
        return consecutive
    
    def _calculate_margin_usage(self) -> float:
        """Calculate margin usage"""
        total_margin = sum(pos.margin_used for pos in self.positions.values())
        total_equity = sum(pos.unrealized_pnl for pos in self.positions.values())
        return total_margin / total_equity if total_equity > 0 else 0.0

