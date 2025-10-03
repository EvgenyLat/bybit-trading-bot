"""
Strategy Management System
Handles multiple trading strategies with switching and portfolio allocation
"""

import asyncio
import logging
import yaml
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class StrategyMode(Enum):
    """Strategy execution modes"""
    SINGLE = "single"           # Only one strategy active
    PARALLEL = "parallel"       # Multiple strategies running simultaneously
    PORTFOLIO = "portfolio"     # Portfolio allocation across strategies
    ADAPTIVE = "adaptive"       # Dynamic strategy switching based on performance


class StrategyStatus(Enum):
    """Strategy status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PAUSED = "paused"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    strategy_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_trade_duration: float
    last_update: datetime


@dataclass
class StrategyAllocation:
    """Strategy allocation in portfolio mode"""
    strategy_name: str
    allocation_percent: float
    max_allocation: float
    min_allocation: float
    current_exposure: float
    performance_score: float


class StrategyManager:
    """Manages multiple trading strategies"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.strategies_config = self._load_strategies_config()
        
        # Strategy management
        self.active_strategies: Dict[str, Any] = {}
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.strategy_allocations: Dict[str, StrategyAllocation] = {}
        self.strategy_mode = StrategyMode(config.get('strategy_mode', 'single'))
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        self.switch_history = []
        
        # Risk management
        self.max_total_exposure = config.get('max_total_exposure', 1.0)
        self.strategy_correlation_threshold = config.get('strategy_correlation_threshold', 0.7)
        
        logger.info(f"Strategy Manager initialized in {self.strategy_mode.value} mode")
    
    def _load_strategies_config(self) -> Dict:
        """Load strategies configuration"""
        try:
            with open('config/advanced_strategies.yaml', 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading strategies config: {e}")
            return {}
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategies"""
        strategies = []
        for strategy_name, strategy_config in self.strategies_config.get('strategies', {}).items():
            if strategy_config.get('enabled', False):
                strategies.append(strategy_name)
        return strategies
    
    def get_strategy_status(self, strategy_name: str) -> StrategyStatus:
        """Get current status of a strategy"""
        if strategy_name not in self.active_strategies:
            return StrategyStatus.INACTIVE
        
        strategy = self.active_strategies[strategy_name]
        if hasattr(strategy, 'status'):
            return StrategyStatus(strategy.status)
        
        return StrategyStatus.ACTIVE
    
    def activate_strategy(self, strategy_name: str) -> bool:
        """Activate a strategy"""
        try:
            if strategy_name not in self.strategies_config.get('strategies', {}):
                logger.error(f"Strategy {strategy_name} not found in configuration")
                return False
            
            strategy_config = self.strategies_config['strategies'][strategy_name]
            if not strategy_config.get('enabled', False):
                logger.error(f"Strategy {strategy_name} is disabled in configuration")
                return False
            
            # Create strategy instance based on type
            strategy_instance = self._create_strategy_instance(strategy_name, strategy_config)
            
            if strategy_instance:
                self.active_strategies[strategy_name] = strategy_instance
                logger.info(f"Strategy {strategy_name} activated")
                return True
            else:
                logger.error(f"Failed to create strategy instance for {strategy_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error activating strategy {strategy_name}: {e}")
            return False
    
    def deactivate_strategy(self, strategy_name: str) -> bool:
        """Deactivate a strategy"""
        try:
            if strategy_name in self.active_strategies:
                strategy = self.active_strategies[strategy_name]
                
                # Close any open positions
                if hasattr(strategy, 'close_all_positions'):
                    strategy.close_all_positions()
                
                # Clean up resources
                if hasattr(strategy, 'cleanup'):
                    strategy.cleanup()
                
                del self.active_strategies[strategy_name]
                logger.info(f"Strategy {strategy_name} deactivated")
                return True
            else:
                logger.warning(f"Strategy {strategy_name} is not active")
                return False
                
        except Exception as e:
            logger.error(f"Error deactivating strategy {strategy_name}: {e}")
            return False
    
    def switch_to_single_strategy(self, strategy_name: str) -> bool:
        """Switch to single strategy mode"""
        try:
            # Deactivate all strategies
            for active_strategy in list(self.active_strategies.keys()):
                self.deactivate_strategy(active_strategy)
            
            # Activate the selected strategy
            if self.activate_strategy(strategy_name):
                self.strategy_mode = StrategyMode.SINGLE
                self.switch_history.append({
                    'timestamp': datetime.now(),
                    'mode': 'single',
                    'strategy': strategy_name,
                    'reason': 'manual_switch'
                })
                logger.info(f"Switched to single strategy mode: {strategy_name}")
                return True
            else:
                logger.error(f"Failed to switch to strategy {strategy_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error switching to single strategy: {e}")
            return False
    
    def enable_parallel_strategies(self, strategy_names: List[str]) -> bool:
        """Enable parallel execution of multiple strategies"""
        try:
            # Check if strategies are compatible for parallel execution
            if not self._check_strategy_compatibility(strategy_names):
                logger.error("Strategies are not compatible for parallel execution")
                return False
            
            # Deactivate current strategies
            for active_strategy in list(self.active_strategies.keys()):
                self.deactivate_strategy(active_strategy)
            
            # Activate selected strategies
            activated_count = 0
            for strategy_name in strategy_names:
                if self.activate_strategy(strategy_name):
                    activated_count += 1
            
            if activated_count > 0:
                self.strategy_mode = StrategyMode.PARALLEL
                self.switch_history.append({
                    'timestamp': datetime.now(),
                    'mode': 'parallel',
                    'strategies': strategy_names,
                    'reason': 'manual_enable'
                })
                logger.info(f"Enabled parallel strategies: {strategy_names}")
                return True
            else:
                logger.error("No strategies were successfully activated")
                return False
                
        except Exception as e:
            logger.error(f"Error enabling parallel strategies: {e}")
            return False
    
    def setup_portfolio_mode(self, allocations: Dict[str, float]) -> bool:
        """Setup portfolio allocation mode"""
        try:
            # Validate allocations
            total_allocation = sum(allocations.values())
            if abs(total_allocation - 1.0) > 0.01:  # Allow small rounding errors
                logger.error(f"Total allocation must be 1.0, got {total_allocation}")
                return False
            
            # Setup strategy allocations
            for strategy_name, allocation in allocations.items():
                self.strategy_allocations[strategy_name] = StrategyAllocation(
                    strategy_name=strategy_name,
                    allocation_percent=allocation,
                    max_allocation=min(allocation * 1.5, 0.5),  # Max 50% per strategy
                    min_allocation=max(allocation * 0.5, 0.05),  # Min 5% per strategy
                    current_exposure=0.0,
                    performance_score=0.0
                )
            
            # Activate strategies
            for strategy_name in allocations.keys():
                if not self.activate_strategy(strategy_name):
                    logger.error(f"Failed to activate strategy {strategy_name}")
                    return False
            
            self.strategy_mode = StrategyMode.PORTFOLIO
            self.switch_history.append({
                'timestamp': datetime.now(),
                'mode': 'portfolio',
                'allocations': allocations,
                'reason': 'manual_setup'
            })
            logger.info(f"Portfolio mode setup with allocations: {allocations}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up portfolio mode: {e}")
            return False
    
    def enable_adaptive_mode(self) -> bool:
        """Enable adaptive strategy switching based on performance"""
        try:
            # Get all available strategies
            available_strategies = self.get_available_strategies()
            
            if len(available_strategies) < 2:
                logger.error("Need at least 2 strategies for adaptive mode")
                return False
            
            # Initialize performance tracking for all strategies
            for strategy_name in available_strategies:
                if strategy_name not in self.strategy_performance:
                    self.strategy_performance[strategy_name] = StrategyPerformance(
                        strategy_name=strategy_name,
                        total_trades=0,
                        winning_trades=0,
                        losing_trades=0,
                        win_rate=0.0,
                        total_pnl=0.0,
                        max_drawdown=0.0,
                        sharpe_ratio=0.0,
                        profit_factor=0.0,
                        avg_trade_duration=0.0,
                        last_update=datetime.now()
                    )
            
            self.strategy_mode = StrategyMode.ADAPTIVE
            self.switch_history.append({
                'timestamp': datetime.now(),
                'mode': 'adaptive',
                'strategies': available_strategies,
                'reason': 'manual_enable'
            })
            logger.info("Adaptive mode enabled")
            return True
            
        except Exception as e:
            logger.error(f"Error enabling adaptive mode: {e}")
            return False
    
    def _create_strategy_instance(self, strategy_name: str, strategy_config: Dict) -> Optional[Any]:
        """Create strategy instance based on configuration"""
        try:
            # This would create actual strategy instances
            # For now, return a mock strategy
            class MockStrategy:
                def __init__(self, name: str, config: Dict):
                    self.name = name
                    self.config = config
                    self.status = 'active'
                    self.positions = []
                
                def generate_signal(self, market_data: Dict) -> Dict:
                    return {'direction': 'hold', 'confidence': 0.5}
                
                def close_all_positions(self):
                    self.positions = []
                
                def cleanup(self):
                    pass
            
            return MockStrategy(strategy_name, strategy_config)
            
        except Exception as e:
            logger.error(f"Error creating strategy instance: {e}")
            return None
    
    def _check_strategy_compatibility(self, strategy_names: List[str]) -> bool:
        """Check if strategies are compatible for parallel execution"""
        try:
            # Check for conflicting strategies
            conflicting_pairs = [
                ('market_making', 'momentum'),  # Market making vs momentum
                ('mean_reversion', 'momentum'),  # Mean reversion vs momentum
            ]
            
            for strategy1, strategy2 in conflicting_pairs:
                if strategy1 in strategy_names and strategy2 in strategy_names:
                    logger.warning(f"Potential conflict between {strategy1} and {strategy2}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking strategy compatibility: {e}")
            return False
    
    def update_strategy_performance(self, strategy_name: str, trade_data: Dict):
        """Update strategy performance metrics"""
        try:
            if strategy_name not in self.strategy_performance:
                self.strategy_performance[strategy_name] = StrategyPerformance(
                    strategy_name=strategy_name,
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    win_rate=0.0,
                    total_pnl=0.0,
                    max_drawdown=0.0,
                    sharpe_ratio=0.0,
                    profit_factor=0.0,
                    avg_trade_duration=0.0,
                    last_update=datetime.now()
                )
            
            performance = self.strategy_performance[strategy_name]
            
            # Update metrics
            performance.total_trades += 1
            if trade_data.get('pnl', 0) > 0:
                performance.winning_trades += 1
            else:
                performance.losing_trades += 1
            
            performance.win_rate = performance.winning_trades / performance.total_trades
            performance.total_pnl += trade_data.get('pnl', 0)
            performance.last_update = datetime.now()
            
            # Update performance history
            self.performance_history[strategy_name].append({
                'timestamp': datetime.now(),
                'pnl': trade_data.get('pnl', 0),
                'trade_duration': trade_data.get('duration', 0)
            })
            
            # Keep only last 1000 trades
            if len(self.performance_history[strategy_name]) > 1000:
                self.performance_history[strategy_name] = self.performance_history[strategy_name][-1000:]
            
            logger.debug(f"Updated performance for {strategy_name}: {performance.total_pnl}")
            
        except Exception as e:
            logger.error(f"Error updating strategy performance: {e}")
    
    def get_strategy_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report for all strategies"""
        try:
            report = {
                'mode': self.strategy_mode.value,
                'active_strategies': list(self.active_strategies.keys()),
                'strategy_performance': {},
                'portfolio_metrics': {},
                'recommendations': []
            }
            
            # Individual strategy performance
            for strategy_name, performance in self.strategy_performance.items():
                report['strategy_performance'][strategy_name] = {
                    'total_trades': performance.total_trades,
                    'win_rate': performance.win_rate,
                    'total_pnl': performance.total_pnl,
                    'max_drawdown': performance.max_drawdown,
                    'sharpe_ratio': performance.sharpe_ratio,
                    'profit_factor': performance.profit_factor,
                    'last_update': performance.last_update.isoformat()
                }
            
            # Portfolio metrics
            if self.strategy_mode == StrategyMode.PORTFOLIO:
                total_pnl = sum(p.total_pnl for p in self.strategy_performance.values())
                report['portfolio_metrics'] = {
                    'total_pnl': total_pnl,
                    'active_strategies': len(self.active_strategies),
                    'allocations': {name: alloc.allocation_percent for name, alloc in self.strategy_allocations.items()}
                }
            
            # Generate recommendations
            report['recommendations'] = self._generate_recommendations()
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {}
    
    def _generate_recommendations(self) -> List[str]:
        """Generate strategy recommendations based on performance"""
        recommendations = []
        
        try:
            # Find best performing strategy
            if self.strategy_performance:
                best_strategy = max(
                    self.strategy_performance.items(),
                    key=lambda x: x[1].total_pnl
                )
                recommendations.append(f"Best performing strategy: {best_strategy[0]} (PnL: {best_strategy[1].total_pnl:.2f})")
            
            # Check for underperforming strategies
            for strategy_name, performance in self.strategy_performance.items():
                if performance.total_trades > 10 and performance.win_rate < 0.4:
                    recommendations.append(f"Consider reviewing {strategy_name} (win rate: {performance.win_rate:.1%})")
            
            # Mode-specific recommendations
            if self.strategy_mode == StrategyMode.SINGLE:
                recommendations.append("Consider enabling parallel strategies for diversification")
            elif self.strategy_mode == StrategyMode.PARALLEL:
                recommendations.append("Consider portfolio allocation mode for better risk management")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def get_strategy_commands(self) -> Dict[str, str]:
        """Get available strategy management commands"""
        return {
            'list_strategies': 'Show all available strategies',
            'activate <strategy>': 'Activate a specific strategy',
            'deactivate <strategy>': 'Deactivate a specific strategy',
            'switch_single <strategy>': 'Switch to single strategy mode',
            'enable_parallel <strategy1,strategy2,...>': 'Enable parallel strategies',
            'setup_portfolio <allocations>': 'Setup portfolio allocation mode',
            'enable_adaptive': 'Enable adaptive strategy switching',
            'performance_report': 'Show performance report',
            'status': 'Show current strategy status'
        }
    
    def execute_command(self, command: str, args: List[str] = None) -> str:
        """Execute strategy management command"""
        try:
            if command == 'list_strategies':
                strategies = self.get_available_strategies()
                return f"Available strategies: {', '.join(strategies)}"
            
            elif command == 'activate':
                if not args:
                    return "Error: Strategy name required"
                strategy_name = args[0]
                if self.activate_strategy(strategy_name):
                    return f"Strategy {strategy_name} activated"
                else:
                    return f"Failed to activate strategy {strategy_name}"
            
            elif command == 'deactivate':
                if not args:
                    return "Error: Strategy name required"
                strategy_name = args[0]
                if self.deactivate_strategy(strategy_name):
                    return f"Strategy {strategy_name} deactivated"
                else:
                    return f"Failed to deactivate strategy {strategy_name}"
            
            elif command == 'switch_single':
                if not args:
                    return "Error: Strategy name required"
                strategy_name = args[0]
                if self.switch_to_single_strategy(strategy_name):
                    return f"Switched to single strategy mode: {strategy_name}"
                else:
                    return f"Failed to switch to strategy {strategy_name}"
            
            elif command == 'enable_parallel':
                if not args:
                    return "Error: Strategy names required"
                strategy_names = args[0].split(',')
                if self.enable_parallel_strategies(strategy_names):
                    return f"Enabled parallel strategies: {', '.join(strategy_names)}"
                else:
                    return f"Failed to enable parallel strategies"
            
            elif command == 'setup_portfolio':
                if not args:
                    return "Error: Allocations required (format: strategy1:0.4,strategy2:0.6)"
                try:
                    allocations = {}
                    for allocation_str in args[0].split(','):
                        strategy, percent = allocation_str.split(':')
                        allocations[strategy] = float(percent)
                    
                    if self.setup_portfolio_mode(allocations):
                        return f"Portfolio mode setup: {allocations}"
                    else:
                        return "Failed to setup portfolio mode"
                except Exception as e:
                    return f"Error parsing allocations: {e}"
            
            elif command == 'enable_adaptive':
                if self.enable_adaptive_mode():
                    return "Adaptive mode enabled"
                else:
                    return "Failed to enable adaptive mode"
            
            elif command == 'performance_report':
                report = self.get_strategy_performance_report()
                return json.dumps(report, indent=2, default=str)
            
            elif command == 'status':
                status = {
                    'mode': self.strategy_mode.value,
                    'active_strategies': list(self.active_strategies.keys()),
                    'available_strategies': self.get_available_strategies()
                }
                return json.dumps(status, indent=2)
            
            else:
                return f"Unknown command: {command}"
                
        except Exception as e:
            return f"Error executing command: {e}"
