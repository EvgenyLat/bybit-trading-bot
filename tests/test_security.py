"""
Security Tests
Comprehensive security testing suite
"""

import pytest
import asyncio
import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json
import time
from datetime import datetime, timedelta

# Import our security modules
from services.security import SecurityManager, SecureConfigManager
from services.error_handling import ErrorHandler, InputValidator
from services.advanced_risk_manager import AdvancedRiskManager
from services.concurrency import ThreadSafeResourceManager, SafeQueue
from services.security_monitoring import SecurityMonitor, IntrusionDetectionSystem


class TestSecurityManager(unittest.TestCase):
    """Test security manager functionality"""
    
    def setUp(self):
        self.security_manager = SecurityManager("test_password")
        self.test_api_key = "test_api_key_123"
        self.test_api_secret = "test_api_secret_456"
    
    def test_encrypt_decrypt_api_key(self):
        """Test API key encryption and decryption"""
        # Encrypt credentials
        encrypted_data = self.security_manager.encrypt_api_key(
            self.test_api_key, self.test_api_secret
        )
        
        # Verify encryption worked
        self.assertIn('encrypted_credentials', encrypted_data)
        self.assertIn('checksum', encrypted_data)
        self.assertNotEqual(encrypted_data['encrypted_credentials'], self.test_api_key)
        
        # Decrypt credentials
        decrypted_key, decrypted_secret = self.security_manager.decrypt_api_key(
            encrypted_data['encrypted_credentials'], encrypted_data['checksum']
        )
        
        # Verify decryption worked
        self.assertEqual(decrypted_key, self.test_api_key)
        self.assertEqual(decrypted_secret, self.test_api_secret)
    
    def test_checksum_verification(self):
        """Test checksum verification"""
        encrypted_data = self.security_manager.encrypt_api_key(
            self.test_api_key, self.test_api_secret
        )
        
        # Test with correct checksum
        decrypted_key, decrypted_secret = self.security_manager.decrypt_api_key(
            encrypted_data['encrypted_credentials'], encrypted_data['checksum']
        )
        self.assertEqual(decrypted_key, self.test_api_key)
        
        # Test with incorrect checksum
        with self.assertRaises(ValueError):
            self.security_manager.decrypt_api_key(
                encrypted_data['encrypted_credentials'], "wrong_checksum"
            )
    
    def test_api_permissions_validation(self):
        """Test API permissions validation"""
        permissions = self.security_manager.validate_api_permissions(
            self.test_api_key, self.test_api_secret
        )
        
        self.assertIn('has_read_permission', permissions)
        self.assertIn('has_trade_permission', permissions)
        self.assertIn('has_withdraw_permission', permissions)
        self.assertIn('is_testnet', permissions)
        
        # Withdraw permission should be False for security
        self.assertFalse(permissions['has_withdraw_permission'])


class TestInputValidator(unittest.TestCase):
    """Test input validation functionality"""
    
    def test_validate_symbol(self):
        """Test symbol validation"""
        # Valid symbols
        self.assertTrue(InputValidator.validate_symbol("BTCUSDT"))
        self.assertTrue(InputValidator.validate_symbol("ETHUSDT"))
        self.assertTrue(InputValidator.validate_symbol("ADAUSDT"))
        
        # Invalid symbols
        self.assertFalse(InputValidator.validate_symbol(""))
        self.assertFalse(InputValidator.validate_symbol("BTC"))
        self.assertFalse(InputValidator.validate_symbol("BTC-USDT"))
        self.assertFalse(InputValidator.validate_symbol("1234567890123"))
        self.assertFalse(InputValidator.validate_symbol(None))
    
    def test_validate_price(self):
        """Test price validation"""
        # Valid prices
        self.assertTrue(InputValidator.validate_price(100.0))
        self.assertTrue(InputValidator.validate_price("100.50"))
        self.assertTrue(InputValidator.validate_price(0.0001))
        
        # Invalid prices
        self.assertFalse(InputValidator.validate_price(0))
        self.assertFalse(InputValidator.validate_price(-100))
        self.assertFalse(InputValidator.validate_price(1e15))
        self.assertFalse(InputValidator.validate_price("invalid"))
        self.assertFalse(InputValidator.validate_price(None))
    
    def test_validate_quantity(self):
        """Test quantity validation"""
        # Valid quantities
        self.assertTrue(InputValidator.validate_quantity(1.0))
        self.assertTrue(InputValidator.validate_quantity("0.001"))
        self.assertTrue(InputValidator.validate_quantity(1000000))
        
        # Invalid quantities
        self.assertFalse(InputValidator.validate_quantity(0))
        self.assertFalse(InputValidator.validate_quantity(-1))
        self.assertFalse(InputValidator.validate_quantity(1e12))
        self.assertFalse(InputValidator.validate_quantity("invalid"))
    
    def test_validate_leverage(self):
        """Test leverage validation"""
        # Valid leverage
        self.assertTrue(InputValidator.validate_leverage(1.0))
        self.assertTrue(InputValidator.validate_leverage("2.5"))
        self.assertTrue(InputValidator.validate_leverage(100))
        
        # Invalid leverage
        self.assertFalse(InputValidator.validate_leverage(0))
        self.assertFalse(InputValidator.validate_leverage(-1))
        self.assertFalse(InputValidator.validate_leverage(101))
        self.assertFalse(InputValidator.validate_leverage("invalid"))
    
    def test_validate_config(self):
        """Test configuration validation"""
        # Valid config
        valid_config = {
            'api': {
                'api_key': 'test_key',
                'api_secret': 'test_secret',
                'testnet': True
            },
            'trading': {
                'symbol': 'BTCUSDT',
                'leverage': 2
            },
            'risk': {
                'stop_loss': 0.01,
                'take_profit': 0.02,
                'max_daily_loss': 0.05
            }
        }
        
        errors = InputValidator.validate_config(valid_config)
        self.assertEqual(len(errors), 0)
        
        # Invalid config
        invalid_config = {
            'api': {
                'api_key': 'test_key'
                # Missing api_secret and testnet
            },
            'trading': {
                'symbol': 'INVALID_SYMBOL',
                'leverage': 200  # Too high
            }
            # Missing risk section
        }
        
        errors = InputValidator.validate_config(invalid_config)
        self.assertGreater(len(errors), 0)


class TestAdvancedRiskManager(unittest.TestCase):
    """Test advanced risk manager functionality"""
    
    def setUp(self):
        self.config = {
            'risk': {
                'max_daily_loss': 0.03,
                'max_weekly_loss': 0.07,
                'max_monthly_loss': 0.15,
                'max_drawdown': 0.15,
                'max_position_size': 0.05,
                'max_concurrent_positions': 1,
                'max_leverage': 2,
                'emergency_stop_enabled': True,
                'emergency_stop_threshold': 0.05,
                'risk_per_trade': 0.01
            },
            'position_sizing_method': 'fixed'
        }
        self.risk_manager = AdvancedRiskManager(self.config)
    
    def test_risk_limits_check(self):
        """Test risk limits checking"""
        # Test with normal conditions
        signal = {
            'direction': 'long',
            'strength': 0.8,
            'price': 50000,
            'volatility': 0.02
        }
        
        result = asyncio.run(self.risk_manager.check_risk_limits(signal, 10000))
        
        self.assertIn('approved', result)
        self.assertIn('risk_level', result)
        self.assertIn('metrics', result)
    
    def test_emergency_stop(self):
        """Test emergency stop functionality"""
        # Simulate critical loss
        signal = {
            'direction': 'long',
            'strength': 0.8,
            'price': 50000,
            'volatility': 0.02
        }
        
        # Mock daily PnL to trigger emergency stop
        self.risk_manager.daily_pnl = -500  # 5% loss
        
        result = asyncio.run(self.risk_manager.check_risk_limits(signal, 10000))
        
        self.assertFalse(result['approved'])
        self.assertEqual(result['reason'], 'Daily loss limit exceeded')
        self.assertTrue(self.risk_manager.emergency_stop_active)
    
    def test_position_sizing(self):
        """Test position sizing calculations"""
        signal = {
            'direction': 'long',
            'strength': 0.8,
            'price': 50000,
            'volatility': 0.02
        }
        
        account_balance = 10000
        risk_metrics = asyncio.run(self.risk_manager._calculate_risk_metrics(account_balance))
        
        position_size = asyncio.run(self.risk_manager._calculate_position_size(
            signal, account_balance, risk_metrics
        ))
        
        self.assertGreater(position_size, 0)
        self.assertLessEqual(position_size, account_balance * 0.05)  # Max 5%


class TestThreadSafeResourceManager(unittest.TestCase):
    """Test thread-safe resource manager"""
    
    def setUp(self):
        self.resource_manager = ThreadSafeResourceManager(max_workers=5)
    
    def test_resource_acquisition(self):
        """Test resource acquisition and release"""
        resource_id = "test_resource"
        
        # Acquire resource
        acquired = self.resource_manager.acquire_resource(resource_id, timeout=5.0)
        self.assertTrue(acquired)
        
        # Try to acquire same resource from another thread (should fail)
        def try_acquire():
            return self.resource_manager.acquire_resource(resource_id, timeout=1.0)
        
        import threading
        thread = threading.Thread(target=try_acquire)
        thread.start()
        thread.join()
        
        # Release resource
        released = self.resource_manager.release_resource(resource_id)
        self.assertTrue(released)
    
    def test_resource_lock_context_manager(self):
        """Test resource lock context manager"""
        resource_id = "test_resource"
        
        with self.resource_manager.resource_lock(resource_id):
            # Resource should be locked
            self.assertTrue(self.resource_manager._is_resource_locked(resource_id))
        
        # Resource should be released after context
        self.assertFalse(self.resource_manager._is_resource_locked(resource_id))


class TestSafeQueue(unittest.TestCase):
    """Test thread-safe queue"""
    
    def setUp(self):
        self.queue = SafeQueue(maxsize=5)
    
    def test_queue_operations(self):
        """Test queue operations"""
        # Test putting items
        for i in range(5):
            success = self.queue.put(f"item_{i}")
            self.assertTrue(success)
        
        # Test queue is full
        success = self.queue.put("overflow_item")
        self.assertFalse(success)
        
        # Test getting items
        for i in range(5):
            item = self.queue.get()
            self.assertEqual(item, f"item_{i}")
        
        # Test queue is empty
        item = self.queue.get(timeout=0.1)
        self.assertIsNone(item)
    
    def test_queue_stats(self):
        """Test queue statistics"""
        # Add some items
        self.queue.put("item1")
        self.queue.put("item2")
        
        # Get one item
        self.queue.get()
        
        stats = self.queue.get_stats()
        self.assertEqual(stats['put_count'], 2)
        self.assertEqual(stats['get_count'], 1)


class TestSecurityMonitor(unittest.TestCase):
    """Test security monitoring"""
    
    def setUp(self):
        self.config = {
            'max_failed_attempts': 3,
            'rate_limit_window': 60,
            'max_requests_per_window': 10
        }
        self.security_monitor = SecurityMonitor(self.config)
    
    def test_event_logging(self):
        """Test security event logging"""
        from services.security_monitoring import SecurityEventType
        
        # Log a security event
        self.security_monitor.log_event(
            SecurityEventType.SUSPICIOUS_ACTIVITY,
            'medium',
            'Test suspicious activity',
            '192.168.1.100',
            'Test Agent'
        )
        
        # Check event was logged
        self.assertEqual(len(self.security_monitor.events), 1)
        
        event = self.security_monitor.events[0]
        self.assertEqual(event.event_type, SecurityEventType.SUSPICIOUS_ACTIVITY)
        self.assertEqual(event.severity, 'medium')
        self.assertEqual(event.source_ip, '192.168.1.100')
    
    def test_brute_force_detection(self):
        """Test brute force attack detection"""
        from services.security_monitoring import SecurityEventType
        
        # Simulate multiple failed attempts
        for i in range(4):
            self.security_monitor.log_event(
                SecurityEventType.UNAUTHORIZED_ACCESS,
                'medium',
                f'Failed login attempt {i}',
                '192.168.1.100',
                'Test Agent'
            )
        
        # Check if IP is blocked
        self.assertTrue(self.security_monitor.is_ip_blocked('192.168.1.100'))
    
    def test_rate_limiting(self):
        """Test rate limiting"""
        from services.security_monitoring import SecurityEventType
        
        # Simulate many requests
        for i in range(12):
            self.security_monitor.log_event(
                SecurityEventType.API_ERROR,
                'low',
                f'API request {i}',
                '192.168.1.101',
                'Test Agent'
            )
        
        # Check rate limit was triggered
        events = list(self.security_monitor.events)
        rate_limit_events = [
            event for event in events
            if event.event_type == SecurityEventType.RATE_LIMIT_EXCEEDED
        ]
        self.assertGreater(len(rate_limit_events), 0)


class TestIntrusionDetectionSystem(unittest.TestCase):
    """Test intrusion detection system"""
    
    def setUp(self):
        self.config = {
            'anomaly_threshold': 0.8,
            'config_hash': 'test_hash_123'
        }
        self.ids = IntrusionDetectionSystem(self.config)
    
    def test_behavior_analysis(self):
        """Test behavioral analysis"""
        # Normal behavior
        normal_behavior = {
            'api_calls_per_minute': 10,
            'trades_per_hour': 5,
            'avg_position_size': 0.02,
            'error_rate': 0.01
        }
        
        anomaly_detected = self.ids.analyze_behavior(normal_behavior)
        self.assertFalse(anomaly_detected)
        
        # Anomalous behavior
        anomalous_behavior = {
            'api_calls_per_minute': 150,  # Very high
            'trades_per_hour': 100,      # Very high
            'avg_position_size': 0.15,   # Very large
            'error_rate': 0.25           # Very high
        }
        
        anomaly_detected = self.ids.analyze_behavior(anomalous_behavior)
        self.assertTrue(anomaly_detected)
    
    def test_config_tampering_detection(self):
        """Test configuration tampering detection"""
        # Test with correct hash
        tampering_detected = self.ids.detect_config_tampering('test_hash_123')
        self.assertFalse(tampering_detected)
        
        # Test with incorrect hash
        tampering_detected = self.ids.detect_config_tampering('modified_hash_456')
        self.assertTrue(tampering_detected)


# Integration tests
class TestSecurityIntegration(unittest.TestCase):
    """Integration tests for security components"""
    
    def setUp(self):
        self.config = {
            'api': {
                'api_key': 'test_key',
                'api_secret': 'test_secret',
                'testnet': True
            },
            'risk': {
                'max_daily_loss': 0.03,
                'max_position_size': 0.05,
                'emergency_stop_enabled': True
            },
            'security': {
                'max_failed_attempts': 3,
                'rate_limit_window': 60,
                'max_requests_per_window': 10
            }
        }
    
    def test_end_to_end_security_flow(self):
        """Test end-to-end security flow"""
        # Initialize security components
        security_manager = SecurityManager("test_password")
        risk_manager = AdvancedRiskManager(self.config)
        security_monitor = SecurityMonitor(self.config['security'])
        
        # Encrypt API credentials
        encrypted_creds = security_manager.encrypt_api_key(
            self.config['api']['api_key'],
            self.config['api']['api_secret']
        )
        
        # Decrypt and validate
        decrypted_key, decrypted_secret = security_manager.decrypt_api_key(
            encrypted_creds['encrypted_credentials'],
            encrypted_creds['checksum']
        )
        
        self.assertEqual(decrypted_key, self.config['api']['api_key'])
        self.assertEqual(decrypted_secret, self.config['api']['api_secret'])
        
        # Test risk management
        signal = {
            'direction': 'long',
            'strength': 0.8,
            'price': 50000,
            'volatility': 0.02
        }
        
        result = asyncio.run(risk_manager.check_risk_limits(signal, 10000))
        self.assertIn('approved', result)
        
        # Test security monitoring
        from services.security_monitoring import SecurityEventType
        
        security_monitor.log_event(
            SecurityEventType.SUSPICIOUS_ACTIVITY,
            'medium',
            'Integration test event',
            '127.0.0.1',
            'Test Agent'
        )
        
        self.assertEqual(len(security_monitor.events), 1)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)

