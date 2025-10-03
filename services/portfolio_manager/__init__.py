"""
Portfolio Manager Service
Advanced portfolio optimization and multi-asset trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import yaml

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Portfolio optimization using modern portfolio theory"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.assets = config.get('assets', ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'])
        self.lookback_period = config.get('lookback_period', 252)  # 1 year
        self.rebalance_frequency = config.get('rebalance_frequency', 7)  # weekly
        
    def calculate_expected_returns(self, prices: pd.DataFrame) -> np.ndarray:
        """Calculate expected returns using historical data"""
        try:
            returns = prices.pct_change().dropna()
            
            # Use multiple methods for expected returns
            methods = {
                'historical': returns.mean() * 252,  # Annualized
                'exponential': returns.ewm(span=30).mean().iloc[-1] * 252,
                'momentum': returns.tail(20).mean() * 252
            }
            
            # Weighted average of methods
            weights = [0.4, 0.3, 0.3]
            expected_returns = sum(w * method for w, method in zip(weights, methods.values()))
            
            return expected_returns.values
            
        except Exception as e:
            logger.error(f"Error calculating expected returns: {e}")
            return np.zeros(len(self.assets))
    
    def calculate_covariance_matrix(self, prices: pd.DataFrame) -> np.ndarray:
        """Calculate covariance matrix with shrinkage"""
        try:
            returns = prices.pct_change().dropna()
            
            # Use Ledoit-Wolf shrinkage estimator
            lw = LedoitWolf()
            cov_matrix = lw.fit(returns).covariance_
            
            return cov_matrix
            
        except Exception as e:
            logger.error(f"Error calculating covariance matrix: {e}")
            return np.eye(len(self.assets))
    
    def optimize_portfolio(self, expected_returns: np.ndarray, 
                          cov_matrix: np.ndarray, 
                          risk_tolerance: float = 0.5) -> Dict:
        """Optimize portfolio using Markowitz mean-variance optimization"""
        try:
            n_assets = len(expected_returns)
            
            # Objective function: maximize Sharpe ratio
            def objective(weights):
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
                return -sharpe_ratio  # Minimize negative Sharpe ratio
            
            # Constraints
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
            
            # Bounds: each weight between 0 and 1
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Initial guess: equal weights
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(objective, initial_weights, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                
                # Calculate portfolio metrics
                portfolio_return = np.sum(optimal_weights * expected_returns)
                portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
                sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
                
                return {
                    'weights': optimal_weights,
                    'expected_return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'success': True
                }
            else:
                logger.warning("Portfolio optimization failed, using equal weights")
                return {
                    'weights': initial_weights,
                    'expected_return': np.mean(expected_returns),
                    'volatility': np.sqrt(np.mean(np.diag(cov_matrix))),
                    'sharpe_ratio': 0,
                    'success': False
                }
                
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {e}")
            return {
                'weights': np.array([1/len(self.assets)] * len(self.assets)),
                'expected_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'success': False
            }


class CorrelationAnalyzer:
    """Analyze correlations between assets"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def calculate_correlation_matrix(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix"""
        try:
            returns = prices.pct_change().dropna()
            correlation_matrix = returns.corr()
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()
    
    def find_correlation_opportunities(self, correlation_matrix: pd.DataFrame, 
                                     threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Find assets with high correlation for pairs trading"""
        try:
            opportunities = []
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    asset1 = correlation_matrix.columns[i]
                    asset2 = correlation_matrix.columns[j]
                    correlation = correlation_matrix.iloc[i, j]
                    
                    if abs(correlation) >= threshold:
                        opportunities.append((asset1, asset2, correlation))
            
            return sorted(opportunities, key=lambda x: abs(x[2]), reverse=True)
            
        except Exception as e:
            logger.error(f"Error finding correlation opportunities: {e}")
            return []


class RiskParityOptimizer:
    """Risk Parity portfolio optimization"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def optimize_risk_parity(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Optimize portfolio for equal risk contribution"""
        try:
            n_assets = cov_matrix.shape[0]
            
            def risk_parity_objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                risk_contributions = (weights * np.dot(cov_matrix, weights)) / portfolio_vol
                
                # Target equal risk contribution
                target_risk = 1.0 / n_assets
                risk_deviation = np.sum((risk_contributions - target_risk) ** 2)
                
                return risk_deviation
            
            # Constraints
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(n_assets))
            initial_weights = np.array([1/n_assets] * n_assets)
            
            result = minimize(risk_parity_objective, initial_weights, 
                            method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                return result.x
            else:
                return initial_weights
                
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {e}")
            return np.array([1/cov_matrix.shape[0]] * cov_matrix.shape[0])


class PortfolioManager:
    """Main portfolio management service"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.optimizer = PortfolioOptimizer(config)
        self.correlation_analyzer = CorrelationAnalyzer(config)
        self.risk_parity_optimizer = RiskParityOptimizer(config)
        
        self.current_weights = {}
        self.last_rebalance = None
        
    def rebalance_portfolio(self, prices: pd.DataFrame, 
                          method: str = 'markowitz') -> Dict:
        """Rebalance portfolio based on optimization method"""
        try:
            logger.info(f"Rebalancing portfolio using {method} method")
            
            # Calculate expected returns and covariance
            expected_returns = self.optimizer.calculate_expected_returns(prices)
            cov_matrix = self.optimizer.calculate_covariance_matrix(prices)
            
            if method == 'markowitz':
                result = self.optimizer.optimize_portfolio(expected_returns, cov_matrix)
                weights = result['weights']
                
            elif method == 'risk_parity':
                weights = self.risk_parity_optimizer.optimize_risk_parity(cov_matrix)
                
            elif method == 'equal_weight':
                weights = np.array([1/len(self.assets)] * len(self.assets))
                
            else:
                logger.warning(f"Unknown optimization method: {method}")
                weights = np.array([1/len(self.assets)] * len(self.assets))
            
            # Update current weights
            self.current_weights = dict(zip(self.assets, weights))
            self.last_rebalance = pd.Timestamp.now()
            
            logger.info(f"Portfolio rebalanced: {self.current_weights}")
            
            return {
                'weights': self.current_weights,
                'method': method,
                'timestamp': self.last_rebalance,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error rebalancing portfolio: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_correlation_opportunities(self, prices: pd.DataFrame) -> List[Dict]:
        """Get correlation-based trading opportunities"""
        try:
            correlation_matrix = self.correlation_analyzer.calculate_correlation_matrix(prices)
            opportunities = self.correlation_analyzer.find_correlation_opportunities(correlation_matrix)
            
            return [
                {
                    'asset1': opp[0],
                    'asset2': opp[1],
                    'correlation': opp[2],
                    'strategy': 'pairs_trading'
                }
                for opp in opportunities
            ]
            
        except Exception as e:
            logger.error(f"Error getting correlation opportunities: {e}")
            return []
    
    def calculate_portfolio_metrics(self, prices: pd.DataFrame) -> Dict:
        """Calculate current portfolio performance metrics"""
        try:
            if not self.current_weights:
                return {}
            
            # Calculate portfolio returns
            returns = prices.pct_change().dropna()
            portfolio_returns = pd.Series(index=returns.index)
            
            for asset, weight in self.current_weights.items():
                if asset in returns.columns:
                    portfolio_returns += weight * returns[asset]
            
            # Calculate metrics
            metrics = {
                'total_return': (1 + portfolio_returns).prod() - 1,
                'annualized_return': portfolio_returns.mean() * 252,
                'volatility': portfolio_returns.std() * np.sqrt(252),
                'sharpe_ratio': portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252),
                'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
                'win_rate': (portfolio_returns > 0).mean(),
                'current_weights': self.current_weights
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {}
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0

