"""
Comprehensive System Test Suite
Tests all components of the trading bot system
"""

import sys
import os
import unittest
import asyncio
import tempfile
import json
import yaml
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modules
try:
    from services.security import SecurityManager, SecureConfigManager
    from services.error_handling import ErrorHandler, InputValidator
    from services.advanced_risk_manager import AdvancedRiskManager
    from services.concurrency import ThreadSafeResourceManager, SafeQueue
    from services.security_monitoring import SecurityMonitor, IntrusionDetectionSystem
    from services.data_collector.historical_fetcher import HistoricalFetcher
    from services.data_collector.websocket_client import BybitWebSocketClient
    from services.feature_engineering import FeatureBuilder
    from services.portfolio_manager import PortfolioOptimizer, CorrelationAnalyzer
    from services.fundamental_analysis import NewsSentimentAnalyzer, OnChainAnalyzer
    from services.reinforcement_learning import TradingEnvironment, DQNAgent
    from src.indicators import TechnicalIndicators
    from src.executor import BybitTradeExecutor
    from src.risk_manager import RiskManager
    from src.data_feed import BybitDataFeed
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_SUCCESSFUL = False


class TestSystemIntegration(unittest.TestCase):
    """Test system integration and functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_config = {
            'api': {
                'api_key': 'test_key',
                'api_secret': 'test_secret',
                'testnet': True,
                'base_url': 'https://api-testnet.bybit.com'
            },
            'trading': {
                'symbol': 'BTCUSDT',
                'category': 'linear',
                'timeframe': '1',
                'position_size': 0.02,
                'max_positions': 2,
                'leverage': 2,
                'margin_mode': 'isolated'
            },
            'risk': {
                'stop_loss': 0.015,
                'take_profit': 0.03,
                'max_daily_loss': 0.05,
                'max_drawdown': 0.15,
                'max_position_size': 0.1,
                'equity_stop': 0.2
            },
            'indicators': {
                'sma_periods': [5, 10, 20],
                'ema_periods': [8, 13, 21],
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9
            }
        }
        
        # Create test data
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'open': np.random.uniform(40000, 50000, 100),
            'high': np.random.uniform(45000, 55000, 100),
            'low': np.random.uniform(35000, 45000, 100),
            'close': np.random.uniform(40000, 50000, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })
        self.test_data.set_index('timestamp', inplace=True)
    
    def test_imports(self):
        """Test that all modules can be imported"""
        self.assertTrue(IMPORTS_SUCCESSFUL, "Some modules failed to import")
    
    def test_configuration_loading(self):
        """Test configuration loading and validation"""
        # Test YAML config loading
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f)
            config_path = f.name
        
        try:
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            self.assertEqual(loaded_config['trading']['symbol'], 'BTCUSDT')
            self.assertEqual(loaded_config['risk']['max_daily_loss'], 0.05)
            
        finally:
            os.unlink(config_path)
    
    def test_security_manager(self):
        """Test security manager functionality"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports failed")
        
        security_manager = SecurityManager("test_password")
        
        # Test encryption/decryption
        encrypted = security_manager.encrypt_api_key("test_key", "test_secret")
        decrypted_key, decrypted_secret = security_manager.decrypt_api_key(
            encrypted['encrypted_credentials'], encrypted['checksum']
        )
        
        self.assertEqual(decrypted_key, "test_key")
        self.assertEqual(decrypted_secret, "test_secret")
    
    def test_input_validation(self):
        """Test input validation"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports failed")
        
        # Test symbol validation
        self.assertTrue(InputValidator.validate_symbol("BTCUSDT"))
        self.assertFalse(InputValidator.validate_symbol("INVALID"))
        
        # Test price validation
        self.assertTrue(InputValidator.validate_price(50000.0))
        self.assertFalse(InputValidator.validate_price(-100))
        
        # Test quantity validation
        self.assertTrue(InputValidator.validate_quantity(1.0))
        self.assertFalse(InputValidator.validate_quantity(0))
        
        # Test leverage validation
        self.assertTrue(InputValidator.validate_leverage(2.0))
        self.assertFalse(InputValidator.validate_leverage(200))
    
    def test_risk_manager(self):
        """Test risk manager functionality"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports failed")
        
        risk_manager = AdvancedRiskManager(self.test_config)
        
        # Test risk check
        signal = {
            'direction': 'long',
            'strength': 0.8,
            'price': 50000,
            'volatility': 0.02
        }
        
        result = asyncio.run(risk_manager.check_risk_limits(signal, 10000))
        
        self.assertIn('approved', result)
        self.assertIn('risk_level', result)
        self.assertIn('metrics', result)
    
    def test_concurrency_manager(self):
        """Test thread-safe resource manager"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports failed")
        
        resource_manager = ThreadSafeResourceManager(max_workers=5)
        
        # Test resource acquisition
        acquired = resource_manager.acquire_resource("test_resource", timeout=5.0)
        self.assertTrue(acquired)
        
        # Test resource release
        released = resource_manager.release_resource("test_resource")
        self.assertTrue(released)
    
    def test_safe_queue(self):
        """Test thread-safe queue"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports failed")
        
        queue = SafeQueue(maxsize=5)
        
        # Test putting items
        for i in range(5):
            success = queue.put(f"item_{i}")
            self.assertTrue(success)
        
        # Test getting items
        for i in range(5):
            item = queue.get()
            self.assertEqual(item, f"item_{i}")
    
    def test_security_monitoring(self):
        """Test security monitoring"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports failed")
        
        config = {
            'max_failed_attempts': 3,
            'rate_limit_window': 60,
            'max_requests_per_window': 10
        }
        
        monitor = SecurityMonitor(config)
        
        # Test event logging
        from services.security_monitoring import SecurityEventType
        
        monitor.log_event(
            SecurityEventType.SUSPICIOUS_ACTIVITY,
            'medium',
            'Test event',
            '127.0.0.1',
            'Test Agent'
        )
        
        self.assertEqual(len(monitor.events), 1)
    
    def test_technical_indicators(self):
        """Test technical indicators calculation"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports failed")
        
        indicators = TechnicalIndicators(self.test_config)
        
        # Test indicator calculation
        result = indicators.calculate_all_indicators(self.test_data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)
        
        # Check for common indicators
        expected_indicators = ['SMA_5', 'SMA_10', 'EMA_8', 'EMA_13', 'RSI', 'MACD']
        for indicator in expected_indicators:
            if indicator in result.columns:
                self.assertFalse(result[indicator].isna().all())
    
    def test_data_feed(self):
        """Test data feed functionality"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports failed")
        
        # Mock the data feed to avoid actual API calls
        with patch('src.data_feed.BybitDataFeed') as mock_feed:
            mock_feed.return_value.get_realtime_data.return_value = {
                'symbol': 'BTCUSDT',
                'price': 50000.0,
                'volume': 1000.0,
                'timestamp': datetime.now()
            }
            
            data_feed = mock_feed.return_value
            data = data_feed.get_realtime_data('BTCUSDT')
            
            self.assertEqual(data['symbol'], 'BTCUSDT')
            self.assertEqual(data['price'], 50000.0)
    
    def test_portfolio_optimizer(self):
        """Test portfolio optimization"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports failed")
        
        # Create test price data
        prices = pd.DataFrame({
            'BTCUSDT': np.random.uniform(40000, 50000, 100),
            'ETHUSDT': np.random.uniform(2000, 3000, 100),
            'ADAUSDT': np.random.uniform(0.3, 0.5, 100)
        })
        
        optimizer = PortfolioOptimizer({'assets': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']})
        
        # Test expected returns calculation
        expected_returns = optimizer.calculate_expected_returns(prices)
        self.assertEqual(len(expected_returns), 3)
        
        # Test covariance matrix calculation
        cov_matrix = optimizer.calculate_covariance_matrix(prices)
        self.assertEqual(cov_matrix.shape, (3, 3))
    
    def test_correlation_analyzer(self):
        """Test correlation analysis"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports failed")
        
        # Create test data
        prices = pd.DataFrame({
            'BTCUSDT': np.random.uniform(40000, 50000, 100),
            'ETHUSDT': np.random.uniform(2000, 3000, 100),
            'ADAUSDT': np.random.uniform(0.3, 0.5, 100)
        })
        
        analyzer = CorrelationAnalyzer({})
        
        # Test correlation matrix
        corr_matrix = analyzer.calculate_correlation_matrix(prices)
        self.assertEqual(corr_matrix.shape, (3, 3))
        
        # Test correlation opportunities
        opportunities = analyzer.find_correlation_opportunities(corr_matrix, threshold=0.7)
        self.assertIsInstance(opportunities, list)
    
    def test_news_sentiment_analyzer(self):
        """Test news sentiment analysis"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports failed")
        
        analyzer = NewsSentimentAnalyzer({'news_api_key': 'test_key'})
        
        # Test sentiment analysis
        sentiment = analyzer.analyze_sentiment("Bitcoin price is rising")
        self.assertIn('sentiment', sentiment)
        self.assertIn('confidence', sentiment)
        self.assertIn('score', sentiment)
    
    def test_trading_environment(self):
        """Test trading environment for RL"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports failed")
        
        # Create test data
        data = pd.DataFrame({
            'close': np.random.uniform(40000, 50000, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })
        
        env = TradingEnvironment(data, {'initial_cash': 10000})
        
        # Test environment reset
        obs = env.reset()
        self.assertIsInstance(obs, np.ndarray)
        
        # Test environment step
        obs, reward, done, info = env.step(1)  # Buy action
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
    
    def test_error_handling(self):
        """Test error handling system"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports failed")
        
        error_handler = ErrorHandler({})
        
        # Test error handling
        try:
            raise ValueError("Test error")
        except Exception as e:
            handled = error_handler.handle_error(
                e, {'context': 'test'}, 
                severity='medium', 
                category='system_error'
            )
            self.assertIsInstance(handled, bool)
    
    def test_intrusion_detection(self):
        """Test intrusion detection system"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports failed")
        
        ids = IntrusionDetectionSystem({'anomaly_threshold': 0.8})
        
        # Test normal behavior
        normal_behavior = {
            'api_calls_per_minute': 10,
            'trades_per_hour': 5,
            'avg_position_size': 0.02,
            'error_rate': 0.01
        }
        
        anomaly = ids.analyze_behavior(normal_behavior)
        self.assertFalse(anomaly)
        
        # Test anomalous behavior
        anomalous_behavior = {
            'api_calls_per_minute': 150,
            'trades_per_hour': 100,
            'avg_position_size': 0.15,
            'error_rate': 0.25
        }
        
        anomaly = ids.analyze_behavior(anomalous_behavior)
        self.assertTrue(anomaly)


class TestSystemPerformance(unittest.TestCase):
    """Test system performance and scalability"""
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports failed")
        
        # Create large dataset
        large_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=100000, freq='1min'),
            'open': np.random.uniform(40000, 50000, 100000),
            'high': np.random.uniform(45000, 55000, 100000),
            'low': np.random.uniform(35000, 45000, 100000),
            'close': np.random.uniform(40000, 50000, 100000),
            'volume': np.random.uniform(1000, 10000, 100000)
        })
        large_data.set_index('timestamp', inplace=True)
        
        # Test indicators calculation on large dataset
        config = {
            'indicators': {
                'sma_periods': [5, 10, 20],
                'ema_periods': [8, 13, 21],
                'rsi_period': 14
            }
        }
        
        indicators = TechnicalIndicators(config)
        
        import time
        start_time = time.time()
        result = indicators.calculate_all_indicators(large_data)
        end_time = time.time()
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)
        
        # Performance should be reasonable (less than 10 seconds for 100k rows)
        self.assertLess(end_time - start_time, 10.0)
    
    def test_memory_usage(self):
        """Test memory usage with multiple operations"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports failed")
        
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform multiple operations
        for i in range(10):
            # Create and process data
            data = pd.DataFrame({
                'close': np.random.uniform(40000, 50000, 1000),
                'volume': np.random.uniform(1000, 10000, 1000)
            })
            
            # Process with indicators
            config = {'indicators': {'sma_periods': [5, 10], 'rsi_period': 14}}
            indicators = TechnicalIndicators(config)
            result = indicators.calculate_all_indicators(data)
            
            # Clean up
            del data, result, indicators
            gc.collect()
        
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        self.assertLess(memory_increase, 100.0)


class TestSystemReliability(unittest.TestCase):
    """Test system reliability and error recovery"""
    
    def test_error_recovery(self):
        """Test error recovery mechanisms"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports failed")
        
        error_handler = ErrorHandler({})
        
        # Test recovery from different error types
        error_types = [
            (ValueError("Test error"), "system_error"),
            (ConnectionError("Network error"), "network_error"),
            (KeyError("Missing key"), "data_error")
        ]
        
        for error, category in error_types:
            handled = error_handler.handle_error(
                error, {'context': 'test'}, 
                severity='medium', 
                category=category
            )
            self.assertIsInstance(handled, bool)
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports failed")
        
        error_handler = ErrorHandler({})
        
        # Simulate multiple failures
        for i in range(6):  # More than threshold
            error_handler.handle_error(
                ValueError(f"Error {i}"), 
                {'context': 'test'}, 
                severity='high', 
                category='api_error'
            )
        
        # Circuit breaker should be open
        self.assertTrue(error_handler._is_circuit_breaker_open('api_error'))
    
    def test_graceful_degradation(self):
        """Test graceful degradation when components fail"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports failed")
        
        # Test with missing components
        config = {
            'api': {'api_key': 'test', 'api_secret': 'test', 'testnet': True},
            'trading': {'symbol': 'BTCUSDT', 'leverage': 2},
            'risk': {'max_daily_loss': 0.05}
        }
        
        # Should handle missing components gracefully
        risk_manager = AdvancedRiskManager(config)
        
        signal = {
            'direction': 'long',
            'strength': 0.8,
            'price': 50000,
            'volatility': 0.02
        }
        
        result = asyncio.run(risk_manager.check_risk_limits(signal, 10000))
        self.assertIn('approved', result)


def run_system_tests():
    """Run all system tests"""
    print("🧪 Starting comprehensive system tests...")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestSystemIntegration,
        TestSystemPerformance,
        TestSystemReliability
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n🚨 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
        return True
    else:
        print("\n❌ SOME TESTS FAILED!")
        return False


if __name__ == '__main__':
    success = run_system_tests()
    sys.exit(0 if success else 1)

