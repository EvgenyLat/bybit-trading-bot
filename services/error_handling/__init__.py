"""
Enhanced Error Handling and Validation
Comprehensive error handling, input validation, and recovery mechanisms
"""

import logging
import traceback
import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import functools
import json

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories"""
    API_ERROR = "api_error"
    NETWORK_ERROR = "network_error"
    DATA_ERROR = "data_error"
    CONFIG_ERROR = "config_error"
    RISK_ERROR = "risk_error"
    TRADING_ERROR = "trading_error"
    SYSTEM_ERROR = "system_error"


@dataclass
class ErrorInfo:
    """Error information container"""
    error: Exception
    severity: ErrorSeverity
    category: ErrorCategory
    context: Dict[str, Any]
    timestamp: datetime
    retry_count: int = 0
    max_retries: int = 3
    recovery_action: Optional[str] = None


class ErrorHandler:
    """Centralized error handling system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.error_history = []
        self.circuit_breakers = {}
        self.recovery_strategies = self._init_recovery_strategies()
        
    def _init_recovery_strategies(self) -> Dict[ErrorCategory, Callable]:
        """Initialize recovery strategies for different error types"""
        return {
            ErrorCategory.API_ERROR: self._recover_from_api_error,
            ErrorCategory.NETWORK_ERROR: self._recover_from_network_error,
            ErrorCategory.DATA_ERROR: self._recover_from_data_error,
            ErrorCategory.CONFIG_ERROR: self._recover_from_config_error,
            ErrorCategory.RISK_ERROR: self._recover_from_risk_error,
            ErrorCategory.TRADING_ERROR: self._recover_from_trading_error,
            ErrorCategory.SYSTEM_ERROR: self._recover_from_system_error,
        }
    
    def handle_error(self, error: Exception, context: Dict[str, Any], 
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    category: ErrorCategory = ErrorCategory.SYSTEM_ERROR) -> bool:
        """Handle error with appropriate recovery strategy"""
        try:
            error_info = ErrorInfo(
                error=error,
                severity=severity,
                category=category,
                context=context,
                timestamp=datetime.now()
            )
            
            # Log error
            self._log_error(error_info)
            
            # Add to history
            self.error_history.append(error_info)
            
            # Check circuit breaker
            if self._is_circuit_breaker_open(category):
                logger.warning(f"Circuit breaker open for {category.value}")
                return False
            
            # Attempt recovery
            recovery_success = self._attempt_recovery(error_info)
            
            # Update circuit breaker
            self._update_circuit_breaker(category, recovery_success)
            
            return recovery_success
            
        except Exception as e:
            logger.critical(f"Error in error handler: {e}")
            return False
    
    def _log_error(self, error_info: ErrorInfo):
        """Log error with appropriate level"""
        log_message = f"{error_info.category.value}: {str(error_info.error)}"
        log_context = f"Context: {error_info.context}"
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"{log_message} | {log_context}")
            logger.critical(f"Traceback: {traceback.format_exc()}")
        elif error_info.severity == ErrorSeverity.HIGH:
            logger.error(f"{log_message} | {log_context}")
        elif error_info.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"{log_message} | {log_context}")
        else:
            logger.info(f"{log_message} | {log_context}")
    
    def _attempt_recovery(self, error_info: ErrorInfo) -> bool:
        """Attempt to recover from error"""
        try:
            recovery_strategy = self.recovery_strategies.get(error_info.category)
            if recovery_strategy:
                return recovery_strategy(error_info)
            else:
                logger.warning(f"No recovery strategy for {error_info.category.value}")
                return False
        except Exception as e:
            logger.error(f"Error in recovery attempt: {e}")
            return False
    
    def _recover_from_api_error(self, error_info: ErrorInfo) -> bool:
        """Recover from API errors"""
        try:
            # Check if it's a rate limit error
            if "rate limit" in str(error_info.error).lower():
                logger.info("Rate limit hit, waiting 60 seconds")
                time.sleep(60)
                return True
            
            # Check if it's an authentication error
            if "unauthorized" in str(error_info.error).lower():
                logger.error("Authentication failed, stopping trading")
                return False
            
            # Check if it's a temporary server error
            if "server error" in str(error_info.error).lower():
                logger.info("Server error, retrying in 30 seconds")
                time.sleep(30)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in API recovery: {e}")
            return False
    
    def _recover_from_network_error(self, error_info: ErrorInfo) -> bool:
        """Recover from network errors"""
        try:
            # Wait and retry
            wait_time = min(60, 5 * (2 ** error_info.retry_count))
            logger.info(f"Network error, waiting {wait_time} seconds")
            time.sleep(wait_time)
            return True
            
        except Exception as e:
            logger.error(f"Error in network recovery: {e}")
            return False
    
    def _recover_from_data_error(self, error_info: ErrorInfo) -> bool:
        """Recover from data errors"""
        try:
            # Try to fetch fresh data
            if 'data_fetcher' in error_info.context:
                data_fetcher = error_info.context['data_fetcher']
                symbol = error_info.context.get('symbol', 'BTCUSDT')
                logger.info(f"Attempting to fetch fresh data for {symbol}")
                # This would call the data fetcher to get new data
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in data recovery: {e}")
            return False
    
    def _recover_from_config_error(self, error_info: ErrorInfo) -> bool:
        """Recover from configuration errors"""
        try:
            # Try to reload configuration
            if 'config_path' in error_info.context:
                config_path = error_info.context['config_path']
                logger.info(f"Attempting to reload configuration from {config_path}")
                # This would reload the configuration
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in config recovery: {e}")
            return False
    
    def _recover_from_risk_error(self, error_info: ErrorInfo) -> bool:
        """Recover from risk management errors"""
        try:
            # Risk errors are usually critical and require immediate action
            logger.critical("Risk management error detected, stopping trading")
            return False
            
        except Exception as e:
            logger.error(f"Error in risk recovery: {e}")
            return False
    
    def _recover_from_trading_error(self, error_info: ErrorInfo) -> bool:
        """Recover from trading errors"""
        try:
            # Cancel any pending orders
            if 'order_manager' in error_info.context:
                order_manager = error_info.context['order_manager']
                logger.info("Cancelling pending orders due to trading error")
                # This would cancel pending orders
            
            # Wait before retrying
            time.sleep(10)
            return True
            
        except Exception as e:
            logger.error(f"Error in trading recovery: {e}")
            return False
    
    def _recover_from_system_error(self, error_info: ErrorInfo) -> bool:
        """Recover from system errors"""
        try:
            # System errors are usually critical
            logger.critical("System error detected, stopping trading")
            return False
            
        except Exception as e:
            logger.error(f"Error in system recovery: {e}")
            return False
    
    def _is_circuit_breaker_open(self, category: ErrorCategory) -> bool:
        """Check if circuit breaker is open for category"""
        if category not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[category]
        if breaker['state'] == 'open':
            # Check if enough time has passed to try again
            if time.time() - breaker['last_failure'] > breaker['timeout']:
                breaker['state'] = 'half-open'
                return False
            return True
        
        return False
    
    def _update_circuit_breaker(self, category: ErrorCategory, success: bool):
        """Update circuit breaker state"""
        if category not in self.circuit_breakers:
            self.circuit_breakers[category] = {
                'state': 'closed',
                'failure_count': 0,
                'last_failure': 0,
                'timeout': 300  # 5 minutes
            }
        
        breaker = self.circuit_breakers[category]
        
        if success:
            breaker['state'] = 'closed'
            breaker['failure_count'] = 0
        else:
            breaker['failure_count'] += 1
            breaker['last_failure'] = time.time()
            
            if breaker['failure_count'] >= 5:  # Threshold for opening circuit
                breaker['state'] = 'open'
                logger.warning(f"Circuit breaker opened for {category.value}")


class InputValidator:
    """Input validation and sanitization"""
    
    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """Validate trading symbol"""
        try:
            if not symbol or not isinstance(symbol, str):
                return False
            
            # Check format (e.g., BTCUSDT, ETHUSDT)
            if len(symbol) < 6 or len(symbol) > 12:
                return False
            
            # Check if it contains only alphanumeric characters
            if not symbol.isalnum():
                return False
            
            return True
            
        except Exception:
            return False
    
    @staticmethod
    def validate_price(price: Union[float, int, str]) -> bool:
        """Validate price value"""
        try:
            price_float = float(price)
            
            # Check if price is positive
            if price_float <= 0:
                return False
            
            # Check if price is reasonable (not too large)
            if price_float > 1e12:  # 1 trillion
                return False
            
            # Check if price has reasonable precision
            if len(str(price_float).split('.')[-1]) > 8:  # Max 8 decimal places
                return False
            
            return True
            
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_quantity(quantity: Union[float, int, str]) -> bool:
        """Validate quantity value"""
        try:
            qty_float = float(quantity)
            
            # Check if quantity is positive
            if qty_float <= 0:
                return False
            
            # Check if quantity is reasonable
            if qty_float > 1e9:  # 1 billion
                return False
            
            return True
            
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_leverage(leverage: Union[float, int, str]) -> bool:
        """Validate leverage value"""
        try:
            lev_float = float(leverage)
            
            # Check if leverage is positive
            if lev_float <= 0:
                return False
            
            # Check if leverage is within reasonable bounds
            if lev_float > 100:  # Max 100x leverage
                return False
            
            return True
            
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_config(config: Dict) -> List[str]:
        """Validate configuration dictionary"""
        errors = []
        
        try:
            # Check required sections
            required_sections = ['api', 'trading', 'risk']
            for section in required_sections:
                if section not in config:
                    errors.append(f"Missing required section: {section}")
            
            # Validate API section
            if 'api' in config:
                api_config = config['api']
                if 'api_key' not in api_config:
                    errors.append("Missing API key")
                if 'api_secret' not in api_config:
                    errors.append("Missing API secret")
                if 'testnet' not in api_config:
                    errors.append("Missing testnet flag")
            
            # Validate trading section
            if 'trading' in config:
                trading_config = config['trading']
                if 'symbol' not in trading_config:
                    errors.append("Missing trading symbol")
                elif not InputValidator.validate_symbol(trading_config['symbol']):
                    errors.append("Invalid trading symbol")
                
                if 'leverage' in trading_config:
                    if not InputValidator.validate_leverage(trading_config['leverage']):
                        errors.append("Invalid leverage value")
            
            # Validate risk section
            if 'risk' in config:
                risk_config = config['risk']
                risk_params = ['stop_loss', 'take_profit', 'max_daily_loss']
                for param in risk_params:
                    if param in risk_config:
                        value = risk_config[param]
                        if not isinstance(value, (int, float)) or value <= 0 or value >= 1:
                            errors.append(f"Invalid {param} value: {value}")
            
            return errors
            
        except Exception as e:
            errors.append(f"Configuration validation error: {e}")
            return errors


def retry_on_error(max_retries: int = 3, delay: float = 1.0, 
                  backoff: float = 2.0, exceptions: tuple = (Exception,)):
    """Decorator for retrying functions on error"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise e
                    
                    wait_time = delay * (backoff ** attempt)
                    logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}), "
                                 f"retrying in {wait_time:.2f} seconds: {e}")
                    time.sleep(wait_time)
            
            raise last_exception
        
        return wrapper
    return decorator


def async_retry_on_error(max_retries: int = 3, delay: float = 1.0, 
                        backoff: float = 2.0, exceptions: tuple = (Exception,)):
    """Async decorator for retrying functions on error"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Async function {func.__name__} failed after {max_retries} retries: {e}")
                        raise e
                    
                    wait_time = delay * (backoff ** attempt)
                    logger.warning(f"Async function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}), "
                                 f"retrying in {wait_time:.2f} seconds: {e}")
                    await asyncio.sleep(wait_time)
            
            raise last_exception
        
        return wrapper
    return decorator


def validate_inputs(**validators):
    """Decorator for input validation"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate inputs
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValueError(f"Invalid input for parameter '{param_name}': {value}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

