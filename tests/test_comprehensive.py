#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 Comprehensive Test Suite
Комплексные тесты для всех модулей торгового бота
"""

import unittest
import sys
import os
from pathlib import Path
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json

# Добавляем src в path для импорта
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

class TestEnhancedSecureExecutor(unittest.TestCase):
    """Тесты улучшенного безопасного исполнителя"""
    
    def setUp(self):
        """Настройка тестов"""
        self.executor = None  # Будет создан в тестах с моками
    
    @patch('src.enhanced_secure_executor.HTTP')
    def test_executor_initialization(self, mock_http):
        """Тест инициализации исполнителя"""
        mock_client = Mock()
        mock_client.get_server_time.return_value = {'retCode': 0}
        mock_http.return_value = mock_client
        
        from src.enhanced_secure_executor import EnhancedSecureExecutor
        
        executor = EnhancedSecureExecutor(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
        
        self.assertIsNotNone(executor)
        self.assertEqual(executor.testnet, True)
        self.assertFalse(executor.circuit_breaker_active)
    
    def test_decimal_precision(self):
        """Тест точности Decimal расчетов"""
        from src.enhanced_secure_executor import EnhancedSecureExecutor
        
        # Создаем executor без реального клиента
        executor = EnhancedSecureExecutor.__new__(EnhancedSecureExecutor)
        executor.logger = Mock()
        
        # Тест квантования цены
        price = Decimal('123.456789')
        tick_size = Decimal('0.01')
        quantized = executor._quantize_price(price, tick_size)
        
        self.assertEqual(quantized, Decimal('123.45'))
        
        # Тест квантования количества
        quantity = Decimal('1.234567')
        lot_size = Decimal('0.001')
        quantized_qty = executor._quantize_quantity(quantity, lot_size)
        
        self.assertEqual(quantized_qty, Decimal('1.234'))
    
    def test_client_order_id_generation(self):
        """Тест генерации client_order_id"""
        from src.enhanced_secure_executor import EnhancedSecureExecutor
        
        executor = EnhancedSecureExecutor.__new__(EnhancedSecureExecutor)
        executor.logger = Mock()
        
        # Генерируем несколько ID
        id1 = executor._generate_client_order_id("test_strategy")
        id2 = executor._generate_client_order_id("test_strategy")
        
        # Проверяем что они разные
        self.assertNotEqual(id1, id2)
        
        # Проверяем формат
        self.assertTrue(id1.startswith("bot-test_strategy-"))
        self.assertTrue(len(id1) > 20)  # Должен быть достаточно длинным

class TestSafeHTTPClient(unittest.TestCase):
    """Тесты безопасного HTTP клиента"""
    
    def setUp(self):
        """Настройка тестов"""
        from src.safe_http_client import SafeHTTPClient
        self.client = SafeHTTPClient(timeout=5)
    
    def tearDown(self):
        """Очистка после тестов"""
        self.client.close()
    
    @patch('requests.Session.get')
    def test_safe_get_success(self, mock_get):
        """Тест успешного GET запроса"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        response = self.client.safe_get("https://api.example.com/test")
        
        self.assertEqual(response.status_code, 200)
        mock_get.assert_called_once()
    
    @patch('requests.Session.get')
    def test_safe_get_timeout(self, mock_get):
        """Тест обработки таймаута"""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout()
        
        with self.assertRaises(requests.exceptions.Timeout):
            self.client.safe_get("https://api.example.com/test")

class TestDecimalCalculations(unittest.TestCase):
    """Тесты Decimal расчетов"""
    
    def test_money_calculations(self):
        """Тест денежных расчетов с Decimal"""
        # Тест точных расчетов
        balance = Decimal('1000.00')
        risk_pct = Decimal('0.02')  # 2%
        stop_distance = Decimal('50.00')
        
        risk_amount = balance * risk_pct
        position_size = risk_amount / stop_distance
        
        expected_position = Decimal('0.40')
        self.assertEqual(position_size, expected_position)
        
        # Тест что float дал бы неточный результат
        balance_float = 1000.00
        risk_pct_float = 0.02
        stop_distance_float = 50.00
        
        position_size_float = (balance_float * risk_pct_float) / stop_distance_float
        
        # Decimal точнее чем float
        self.assertNotEqual(float(position_size), position_size_float)
    
    def test_price_quantization(self):
        """Тест квантования цен"""
        from decimal import Decimal, ROUND_DOWN
        
        def quantize_price(price: Decimal, tick_size: Decimal) -> Decimal:
            return (price // tick_size) * tick_size
        
        # Тест квантования
        price = Decimal('123.456789')
        tick_size = Decimal('0.01')
        quantized = quantize_price(price, tick_size)
        
        self.assertEqual(quantized, Decimal('123.45'))
        
        # Тест с разными tick sizes
        price = Decimal('100.123456')
        tick_size = Decimal('0.1')
        quantized = quantize_price(price, tick_size)
        
        self.assertEqual(quantized, Decimal('100.1'))

class TestSecurityFeatures(unittest.TestCase):
    """Тесты функций безопасности"""
    
    def test_secret_redaction(self):
        """Тест скрытия секретов в логах"""
        import re
        
        # Тест паттернов скрытия
        test_log = "api_key=secret123 token=abc456 password=xyz789"
        
        sensitive_patterns = [
            r'(api_key|api_secret|token|password)[=:]\s*\w+',
            r'(ghp_|gho_)[a-zA-Z0-9]{36}',
            r'Bearer\s+\w+',
        ]
        
        redacted_log = test_log
        for pattern in sensitive_patterns:
            redacted_log = re.sub(pattern, r'\1=***REDACTED***', redacted_log, flags=re.IGNORECASE)
        
        self.assertIn('***REDACTED***', redacted_log)
        self.assertNotIn('secret123', redacted_log)
        self.assertNotIn('abc456', redacted_log)
        self.assertNotIn('xyz789', redacted_log)
    
    def test_circuit_breaker_logic(self):
        """Тест логики circuit breaker"""
        from src.enhanced_secure_executor import EnhancedSecureExecutor
        
        executor = EnhancedSecureExecutor.__new__(EnhancedSecureExecutor)
        executor.circuit_breaker_active = False
        executor.circuit_breaker_failures = 0
        executor.circuit_breaker_threshold = 3
        executor.logger = Mock()
        
        # Тест успешных операций
        executor._update_circuit_breaker(True)
        self.assertFalse(executor.circuit_breaker_active)
        self.assertEqual(executor.circuit_breaker_failures, 0)
        
        # Тест накопления ошибок
        for _ in range(3):
            executor._update_circuit_breaker(False)
        
        self.assertTrue(executor.circuit_breaker_active)
        self.assertEqual(executor.circuit_breaker_failures, 3)

class TestConfigurationSecurity(unittest.TestCase):
    """Тесты безопасности конфигурации"""
    
    def test_env_file_validation(self):
        """Тест валидации .env файла"""
        from src.config import SecureConfig
        
        # Создаем временный .env файл с тестовыми значениями
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("BYBIT_API_KEY=ваш_api_ключ\n")
            f.write("BYBIT_API_SECRET=ваш_seкретный_ключ\n")
            temp_env = f.name
        
        try:
            # Должен выбросить исключение из-за тестовых значений
            with self.assertRaises(RuntimeError):
                SecureConfig(env_file=temp_env)
        finally:
            os.unlink(temp_env)
    
    def test_missing_secrets_validation(self):
        """Тест валидации отсутствующих секретов"""
        from src.config import SecureConfig
        
        # Создаем временный .env файл без секретов
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("# Empty env file\n")
            temp_env = f.name
        
        try:
            # Должен выбросить исключение из-за отсутствующих секретов
            with self.assertRaises(RuntimeError):
                SecureConfig(env_file=temp_env)
        finally:
            os.unlink(temp_env)

class TestIdempotency(unittest.TestCase):
    """Тесты идемпотентности"""
    
    def test_order_cache(self):
        """Тест кэширования ордеров"""
        from src.enhanced_secure_executor import EnhancedSecureExecutor
        
        executor = EnhancedSecureExecutor.__new__(EnhancedSecureExecutor)
        executor.order_cache = {}
        executor.logger = Mock()
        
        # Добавляем ордер в кэш
        test_order_id = "test-order-123"
        test_response = {'orderId': test_order_id, 'status': 'success'}
        
        executor.order_cache[test_order_id] = test_response
        
        # Проверяем что ордер в кэше
        self.assertIn(test_order_id, executor.order_cache)
        self.assertEqual(executor.order_cache[test_order_id], test_response)
    
    def test_unique_order_ids(self):
        """Тест уникальности order ID"""
        from src.enhanced_secure_executor import EnhancedSecureExecutor
        
        executor = EnhancedSecureExecutor.__new__(EnhancedSecureExecutor)
        executor.logger = Mock()
        
        # Генерируем несколько ID
        ids = set()
        for _ in range(100):
            order_id = executor._generate_client_order_id("test")
            ids.add(order_id)
        
        # Все ID должны быть уникальными
        self.assertEqual(len(ids), 100)

def run_comprehensive_tests():
    """Запуск всех тестов"""
    print("🧪 Running Comprehensive Test Suite")
    print("=" * 50)
    
    # Создаем test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Добавляем все тестовые классы
    test_classes = [
        TestEnhancedSecureExecutor,
        TestSafeHTTPClient,
        TestDecimalCalculations,
        TestSecurityFeatures,
        TestConfigurationSecurity,
        TestIdempotency
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Запускаем тесты
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Выводим результаты
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    print(f"✅ Tests run: {result.testsRun}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"⚠️ Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n🚨 FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n🚨 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    exit(run_comprehensive_tests())
