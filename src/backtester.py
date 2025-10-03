"""
Backtesting Module for Trading Bot
Implements comprehensive backtesting for strategy validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

from indicators import TechnicalIndicators
from risk_manager import RiskManager
from ml_predictor import MLSignalPredictor

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Trade data structure"""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    side: str  # 'buy' or 'sell'
    quantity: float
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float


@dataclass
class BacktestResults:
    """Backtest results data structure"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    avg_trade_duration: float
    avg_winning_trade: float
    avg_losing_trade: float
    profit_factor: float
    trades: List[Trade]
    equity_curve: pd.DataFrame


class Backtester:
    """Comprehensive backtesting system"""
    
    def __init__(self, config: Dict):
        """
        Initialize Backtester
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.backtest_config = config.get('backtesting', {})
        
        # Backtest parameters
        self.initial_capital = self.backtest_config.get('initial_capital', 10000)
        self.commission = self.backtest_config.get('commission', 0.001)
        self.slippage = self.backtest_config.get('slippage', 0.0005)
        
        # Risk management
        self.risk_manager = RiskManager(config)
        
        # ML predictor (if enabled)
        self.ml_predictor = None
        if config.get('ml', {}).get('enabled', False):
            self.ml_predictor = MLSignalPredictor(config)
        
        logger.info("Backtester initialized")
    
    def run_backtest(self, data: pd.DataFrame, start_date: str = None, 
                    end_date: str = None) -> BacktestResults:
        """
        Run comprehensive backtest
        
        Args:
            data: Historical OHLCV data
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            
        Returns:
            BacktestResults object
        """
        try:
            logger.info("Starting backtest...")
            
            # Filter data by date range
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            
            if len(data) < 100:
                raise ValueError("Insufficient data for backtesting")
            
            # Calculate technical indicators
            data_with_indicators = TechnicalIndicators.calculate_all_indicators(
                data, self.config
            )
            
            # Train ML model if enabled
            if self.ml_predictor:
                logger.info("Training ML model for backtest...")
                self.ml_predictor.train_model(data_with_indicators)
            
            # Run simulation
            trades, equity_curve = self._run_simulation(data_with_indicators)
            
            # Calculate performance metrics
            results = self._calculate_performance_metrics(trades, equity_curve)
            
            logger.info(f"Backtest completed: {results.total_trades} trades, "
                       f"{results.win_rate:.2%} win rate, "
                       f"{results.total_return:.2%} total return")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            raise
    
    def _run_simulation(self, data: pd.DataFrame) -> Tuple[List[Trade], pd.DataFrame]:
        """Run trading simulation"""
        try:
            trades = []
            equity_curve = []
            current_capital = self.initial_capital
            position = None
            max_capital = self.initial_capital
            
            # Initialize risk manager
            self.risk_manager.reset_daily_tracking()
            
            for i in range(50, len(data)):  # Start after enough data for indicators
                current_time = data.index[i]
                current_price = data['close'].iloc[i]
                
                # Get current data slice
                current_data = data.iloc[:i+1]
                
                # Calculate signals
                technical_signal = self._calculate_technical_signal(current_data)
                ml_signal = self._get_ml_signal(current_data)
                combined_signal = self._combine_signals(technical_signal, ml_signal)
                
                # Check if we should trade
                if position is None and abs(combined_signal['signal']) > 0.3:
                    # Open new position
                    position = self._open_position(
                        current_time, current_price, combined_signal, current_capital
                    )
                
                elif position is not None:
                    # Check exit conditions
                    exit_reason = self._check_exit_conditions(
                        position, current_time, current_price, current_data
                    )
                    
                    if exit_reason:
                        # Close position
                        trade = self._close_position(
                            position, current_time, current_price, exit_reason
                        )
                        trades.append(trade)
                        
                        # Update capital
                        current_capital += trade.pnl
                        max_capital = max(max_capital, current_capital)
                        
                        # Update risk manager
                        self.risk_manager.update_pnl(trade.pnl)
                        
                        position = None
                
                # Record equity curve
                equity_curve.append({
                    'timestamp': current_time,
                    'equity': current_capital,
                    'price': current_price,
                    'drawdown': (max_capital - current_capital) / max_capital
                })
            
            # Close any remaining position
            if position is not None:
                trade = self._close_position(
                    position, data.index[-1], data['close'].iloc[-1], 'end_of_data'
                )
                trades.append(trade)
            
            equity_df = pd.DataFrame(equity_curve)
            equity_df.set_index('timestamp', inplace=True)
            
            return trades, equity_df
            
        except Exception as e:
            logger.error(f"Error in simulation: {e}")
            raise
    
    def _calculate_technical_signal(self, data: pd.DataFrame) -> float:
        """Calculate technical signal"""
        try:
            if len(data) < 50:
                return 0.0
            
            signal_strength = TechnicalIndicators.get_signal_strength(data, self.config)
            return signal_strength.iloc[-1] if len(signal_strength) > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating technical signal: {e}")
            return 0.0
    
    def _get_ml_signal(self, data: pd.DataFrame) -> Dict:
        """Get ML signal"""
        try:
            if not self.ml_predictor or not self.ml_predictor.is_trained:
                return {'signal': 0, 'confidence': 0.0}
            
            ml_prediction = self.ml_predictor.predict_signal(data)
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
            technical_weight = 0.6
            ml_weight = 0.4
            
            combined_signal = (technical_signal * technical_weight + 
                             ml_signal['signal'] * ml_weight)
            combined_confidence = (abs(technical_signal) * technical_weight + 
                                ml_signal['confidence'] * ml_weight)
            
            return {
                'signal': combined_signal,
                'confidence': combined_confidence
            }
            
        except Exception as e:
            logger.error(f"Error combining signals: {e}")
            return {'signal': 0, 'confidence': 0.0}
    
    def _open_position(self, entry_time: datetime, entry_price: float, 
                      signal: Dict, capital: float) -> Dict:
        """Open new position"""
        try:
            # Calculate position size
            volatility = 0.01  # Default volatility
            position_size = self.risk_manager.calculate_position_size(
                capital, signal['signal'], volatility
            )
            
            # Determine side
            side = 'buy' if signal['signal'] > 0 else 'sell'
            
            # Apply slippage
            if side == 'buy':
                actual_entry_price = entry_price * (1 + self.slippage)
            else:
                actual_entry_price = entry_price * (1 - self.slippage)
            
            return {
                'entry_time': entry_time,
                'entry_price': actual_entry_price,
                'side': side,
                'quantity': position_size,
                'signal_strength': signal['signal'],
                'confidence': signal['confidence']
            }
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return None
    
    def _check_exit_conditions(self, position: Dict, current_time: datetime, 
                             current_price: float, data: pd.DataFrame) -> Optional[str]:
        """Check if position should be closed"""
        try:
            entry_price = position['entry_price']
            side = position['side']
            
            # Calculate current P&L
            if side == 'buy':
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price
            
            # Check stop loss
            stop_loss = self.config['risk']['stop_loss']
            if pnl_pct <= -stop_loss:
                return 'stop_loss'
            
            # Check take profit
            take_profit = self.config['risk']['take_profit']
            if pnl_pct >= take_profit:
                return 'take_profit'
            
            # Check signal reversal
            technical_signal = self._calculate_technical_signal(data)
            ml_signal = self._get_ml_signal(data)
            combined_signal = self._combine_signals(technical_signal, ml_signal)
            
            # Exit if signal strongly opposes position
            if side == 'buy' and combined_signal['signal'] < -0.5:
                return 'signal_reversal'
            elif side == 'sell' and combined_signal['signal'] > 0.5:
                return 'signal_reversal'
            
            # Check time-based exit (optional)
            time_held = (current_time - position['entry_time']).total_seconds() / 3600  # hours
            if time_held > 24:  # Close after 24 hours
                return 'time_exit'
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return None
    
    def _close_position(self, position: Dict, exit_time: datetime, 
                       exit_price: float, exit_reason: str) -> Trade:
        """Close position and create trade record"""
        try:
            entry_price = position['entry_price']
            side = position['side']
            quantity = position['quantity']
            
            # Apply slippage
            if side == 'buy':
                actual_exit_price = exit_price * (1 - self.slippage)
            else:
                actual_exit_price = exit_price * (1 + self.slippage)
            
            # Calculate P&L
            if side == 'buy':
                pnl = quantity * (actual_exit_price - entry_price)
            else:
                pnl = quantity * (entry_price - actual_exit_price)
            
            pnl_pct = pnl / (quantity * entry_price)
            
            # Calculate commission
            commission = quantity * entry_price * self.commission + quantity * actual_exit_price * self.commission
            
            # Calculate slippage cost
            slippage_cost = quantity * abs(exit_price - actual_exit_price)
            
            # Net P&L
            net_pnl = pnl - commission - slippage_cost
            
            return Trade(
                entry_time=position['entry_time'],
                exit_time=exit_time,
                entry_price=entry_price,
                exit_price=actual_exit_price,
                side=side,
                quantity=quantity,
                pnl=net_pnl,
                pnl_pct=pnl_pct,
                commission=commission,
                slippage=slippage_cost
            )
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            raise
    
    def _calculate_performance_metrics(self, trades: List[Trade], 
                                     equity_curve: pd.DataFrame) -> BacktestResults:
        """Calculate comprehensive performance metrics"""
        try:
            if not trades:
                return BacktestResults(
                    total_trades=0, winning_trades=0, losing_trades=0,
                    win_rate=0, total_pnl=0, total_return=0, max_drawdown=0,
                    sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
                    avg_trade_duration=0, avg_winning_trade=0, avg_losing_trade=0,
                    profit_factor=0, trades=[], equity_curve=equity_curve
                )
            
            # Basic metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.pnl > 0])
            losing_trades = len([t for t in trades if t.pnl < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # P&L metrics
            total_pnl = sum(t.pnl for t in trades)
            total_return = total_pnl / self.initial_capital
            
            # Trade duration
            durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades]
            avg_trade_duration = np.mean(durations) if durations else 0
            
            # Win/Loss metrics
            winning_pnls = [t.pnl for t in trades if t.pnl > 0]
            losing_pnls = [t.pnl for t in trades if t.pnl < 0]
            
            avg_winning_trade = np.mean(winning_pnls) if winning_pnls else 0
            avg_losing_trade = np.mean(losing_pnls) if losing_pnls else 0
            
            # Profit factor
            gross_profit = sum(winning_pnls) if winning_pnls else 0
            gross_loss = abs(sum(losing_pnls)) if losing_pnls else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Drawdown
            equity_values = equity_curve['equity'].values
            peak = np.maximum.accumulate(equity_values)
            drawdown = (peak - equity_values) / peak
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
            
            # Risk-adjusted returns
            returns = equity_curve['equity'].pct_change().dropna()
            
            if len(returns) > 1:
                sharpe_ratio = self._calculate_sharpe_ratio(returns)
                sortino_ratio = self._calculate_sortino_ratio(returns)
            else:
                sharpe_ratio = 0
                sortino_ratio = 0
            
            # Calmar ratio
            calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
            
            return BacktestResults(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                total_return=total_return,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                avg_trade_duration=avg_trade_duration,
                avg_winning_trade=avg_winning_trade,
                avg_losing_trade=avg_losing_trade,
                profit_factor=profit_factor,
                trades=trades,
                equity_curve=equity_curve
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            raise
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
            if excess_returns.std() == 0:
                return 0
            return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        except:
            return 0
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        try:
            excess_returns = returns - (risk_free_rate / 252)
            downside_returns = excess_returns[excess_returns < 0]
            if len(downside_returns) == 0 or downside_returns.std() == 0:
                return 0
            return excess_returns.mean() / downside_returns.std() * np.sqrt(252)
        except:
            return 0
    
    def plot_results(self, results: BacktestResults, save_path: str = None):
        """Plot backtest results"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Equity curve
            axes[0, 0].plot(results.equity_curve.index, results.equity_curve['equity'])
            axes[0, 0].set_title('Equity Curve')
            axes[0, 0].set_ylabel('Equity')
            axes[0, 0].grid(True)
            
            # Drawdown
            axes[0, 1].fill_between(results.equity_curve.index, 
                                  results.equity_curve['drawdown'], 0, alpha=0.3, color='red')
            axes[0, 1].set_title('Drawdown')
            axes[0, 1].set_ylabel('Drawdown %')
            axes[0, 1].grid(True)
            
            # P&L distribution
            pnls = [t.pnl for t in results.trades]
            axes[1, 0].hist(pnls, bins=20, alpha=0.7)
            axes[1, 0].set_title('P&L Distribution')
            axes[1, 0].set_xlabel('P&L')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True)
            
            # Monthly returns
            monthly_returns = results.equity_curve['equity'].resample('M').last().pct_change()
            axes[1, 1].bar(range(len(monthly_returns)), monthly_returns)
            axes[1, 1].set_title('Monthly Returns')
            axes[1, 1].set_ylabel('Return %')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting results: {e}")
    
    def generate_report(self, results: BacktestResults) -> str:
        """Generate comprehensive backtest report"""
        try:
            report = f"""
BACKTEST REPORT
===============

PERFORMANCE METRICS
-------------------
Total Trades: {results.total_trades}
Winning Trades: {results.winning_trades}
Losing Trades: {results.losing_trades}
Win Rate: {results.win_rate:.2%}

RETURNS
-------
Total P&L: ${results.total_pnl:.2f}
Total Return: {results.total_return:.2%}
Max Drawdown: {results.max_drawdown:.2%}

RISK METRICS
------------
Sharpe Ratio: {results.sharpe_ratio:.3f}
Sortino Ratio: {results.sortino_ratio:.3f}
Calmar Ratio: {results.calmar_ratio:.3f}

TRADE STATISTICS
----------------
Average Trade Duration: {results.avg_trade_duration:.1f} hours
Average Winning Trade: ${results.avg_winning_trade:.2f}
Average Losing Trade: ${results.avg_losing_trade:.2f}
Profit Factor: {results.profit_factor:.2f}

CONFIGURATION
-------------
Initial Capital: ${self.initial_capital:,.2f}
Commission: {self.commission:.3%}
Slippage: {self.slippage:.3%}
"""
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return "Error generating report"

