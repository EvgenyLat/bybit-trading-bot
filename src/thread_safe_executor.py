#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔒 Thread-Safe Enhanced Executor
Потокобезопасный улучшенный исполнитель ордеров
"""

import time
import uuid
import threading
import logging
from decimal import Decimal, ROUND_DOWN, getcontext
from typing import Dict, Any, Optional, List
from enum import Enum
from collections import defaultdict

from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential, 
    retry_if_exception_type,
    before_sleep_log
)
import requests
from pybit.unified_trading import HTTP

# Импортируем наши модули безопасности
from src.rate_limiter import bybit_rate_limiter
from src.circuit_breaker import advanced_circuit_breaker

# Настройка точности для финансовых расчетов
getcontext().prec = 28

class OrderSide(Enum):
    BUY = "Buy"
    SELL = "Sell"

class OrderType(Enum):
    MARKET = "Market"
    LIMIT = "Limit"
    STOP = "Stop"
    STOP_LIMIT = "StopLimit"

class OrderStatus(Enum):
    NEW = "New"
    PARTIALLY_FILLED = "PartiallyFilled"
    FILLED = "Filled"
    CANCELED = "Canceled"
    REJECTED = "Rejected"

class ThreadSafeExecutor:
    """🔒 Потокобезопасный исполнитель ордеров Bybit"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """
        Инициализация потокобезопасного исполнителя
        
        Args:
            api_key: Bybit API ключ
            api_secret: Bybit API секрет  
            testnet: Использовать тест-сеть
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Потокобезопасные блокировки
        self._lock = threading.RLock()  # Reentrant lock для вложенных вызовов
        self._order_lock = threading.Lock()  # Отдельный lock для ордеров
        self._position_lock = threading.Lock()  # Lock для позиций
        
        # Настройка логирования
        self.logger = self._setup_logging()
        
        # Инициализация клиента
        self.client = self._create_client()
        
        # Потокобезопасные кэши и счетчики
        self.order_cache = {}
        self.order_counter = 0
        self.position_cache = defaultdict(dict)
        
        # Circuit breaker состояние
        self.circuit_breaker_active = False
        self.circuit_breaker_failures = 0
        self.circuit_breaker_threshold = 5
        
        # Статистика (потокобезопасная)
        self.stats = {
            'orders_placed': 0,
            'orders_filled': 0,
            'orders_canceled': 0,
            'total_volume': Decimal('0'),
            'errors': 0
        }
        
    def _setup_logging(self):
        """Настройка безопасного логирования"""
        logger = logging.getLogger('ThreadSafeExecutor')
        logger.setLevel(logging.INFO)
        
        # Безопасный форматтер (скрывает секреты)
        class SafeFormatter(logging.Formatter):
            def format(self, record):
                msg = super().format(record)
                # Удаляем секреты из логов
                sensitive_patterns = [
                    r'(api_key|api_secret|token|password)[=:]\s*\w+',
                    r'(ghp_|gho_)[a-zA-Z0-9]{36}',
                    r'Bearer\s+\w+',
                ]
                import re
                for pattern in sensitive_patterns:
                    msg = re.sub(pattern, r'\1=***REDACTED***', msg, flags=re.IGNORECASE)
                return msg
        
        handler = logging.StreamHandler()
        handler.setFormatter(SafeFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        
        return logger
        
    def _create_client(self) -> HTTP:
        """Создание Bybit клиента с retry логикой"""
        try:
            client = HTTP(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet
            )
            
            # Тестируем подключение
            self._test_connection(client)
            
            self.logger.info(f"✅ Bybit клиент создан (testnet: {self.testnet})")
            return client
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка создания клиента: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, Exception)),
        before_sleep=before_sleep_log(logging.getLogger('ThreadSafeExecutor'), logging.WARNING)
    )
    def _test_connection(self, client: HTTP):
        """Тестирование подключения к Bybit"""
        try:
            # Простой запрос для проверки подключения
            response = client.get_server_time()
            if response.get('retCode') == 0:
                self.logger.info("✅ Подключение к Bybit успешно")
            else:
                raise Exception(f"Bybit API error: {response}")
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка подключения: {e}")
            raise
    
    def _generate_client_order_id(self, strategy_name: str = "default") -> str:
        """Генерация уникального client_order_id для идемпотентности"""
        with self._order_lock:
            timestamp = int(time.time())
            unique_id = str(uuid.uuid4().hex[:8])
            return f"bot-{strategy_name}-{timestamp}-{unique_id}"
    
    def _quantize_price(self, price: Decimal, tick_size: Decimal = Decimal('0.01')) -> Decimal:
        """Квантование цены под tick size биржи"""
        if price <= 0:
            raise ValueError("Price must be positive")
        
        # Округляем вниз до ближайшего tick
        quantized = (price // tick_size) * tick_size
        return quantized.quantize(tick_size, rounding=ROUND_DOWN)
    
    def _quantize_quantity(self, quantity: Decimal, lot_size: Decimal = Decimal('0.001')) -> Decimal:
        """Квантование количества под lot size биржи"""
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        # Округляем вниз до ближайшего lot
        quantized = (quantity // lot_size) * lot_size
        return quantized.quantize(lot_size, rounding=ROUND_DOWN)
    
    def _check_circuit_breaker(self):
        """Проверка circuit breaker"""
        with self._lock:
            if self.circuit_breaker_active:
                raise Exception("🚨 Circuit breaker active - trading suspended")
    
    def _update_circuit_breaker(self, success: bool):
        """Обновление состояния circuit breaker"""
        with self._lock:
            if success:
                self.circuit_breaker_failures = 0
                self.circuit_breaker_active = False
            else:
                self.circuit_breaker_failures += 1
                if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
                    self.circuit_breaker_active = True
                    self.logger.error("🚨 Circuit breaker activated - too many failures")
    
    def _update_stats(self, stat_name: str, value: Any = 1):
        """Потокобезопасное обновление статистики"""
        with self._lock:
            if stat_name in self.stats:
                if isinstance(self.stats[stat_name], Decimal):
                    self.stats[stat_name] += Decimal(str(value))
                else:
                    self.stats[stat_name] += value
    
    @bybit_rate_limiter.order_call
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((requests.exceptions.RequestException, Exception)),
        before_sleep=before_sleep_log(logging.getLogger('ThreadSafeExecutor'), logging.WARNING)
    )
    def place_order_with_retry(self, 
                             symbol: str,
                             side: OrderSide,
                             order_type: OrderType,
                             quantity: Decimal,
                             price: Optional[Decimal] = None,
                             strategy_name: str = "default",
                             **kwargs) -> Dict[str, Any]:
        """
        Размещение ордера с retry логикой и идемпотентностью
        
        Args:
            symbol: Торговая пара
            side: Направление (Buy/Sell)
            order_type: Тип ордера
            quantity: Количество (Decimal)
            price: Цена (для Limit ордеров)
            strategy_name: Имя стратегии
            **kwargs: Дополнительные параметры
            
        Returns:
            Результат размещения ордера
        """
        # Проверяем circuit breaker
        self._check_circuit_breaker()
        
        # Проверяем рыночные условия
        if not advanced_circuit_breaker.should_trade():
            raise Exception("🚨 Market conditions unsafe - trading halted")
        
        # Генерируем уникальный client_order_id
        client_order_id = self._generate_client_order_id(strategy_name)
        
        # Проверяем кэш на дубли (потокобезопасно)
        with self._order_lock:
            if client_order_id in self.order_cache:
                self.logger.warning(f"⚠️ Дублированный client_order_id: {client_order_id}")
                return self.order_cache[client_order_id]
        
        try:
            # Квантуем параметры
            quantity = self._quantize_quantity(quantity)
            
            # Подготавливаем параметры ордера
            order_params = {
                'symbol': symbol,
                'side': side.value,
                'orderType': order_type.value,
                'qty': str(quantity),
                'clientOrderId': client_order_id,
                'timeInForce': kwargs.get('time_in_force', 'GTC')
            }
            
            # Добавляем цену для Limit ордеров
            if order_type == OrderType.LIMIT and price:
                price = self._quantize_price(price)
                order_params['price'] = str(price)
            
            # Добавляем дополнительные параметры
            order_params.update(kwargs)
            
            self.logger.info(f"📤 Размещение ордера: {symbol} {side.value} {quantity} @ {price or 'Market'}")
            
            # Размещаем ордер
            response = self.client.place_order(**order_params)
            
            # Проверяем результат
            if response.get('retCode') == 0:
                order_id = response.get('result', {}).get('orderId')
                self.logger.info(f"✅ Ордер размещен: {order_id}")
                
                # Сохраняем в кэш (потокобезопасно)
                with self._order_lock:
                    self.order_cache[client_order_id] = response
                
                # Обновляем статистику
                self._update_stats('orders_placed')
                self._update_stats('total_volume', quantity)
                
                # Обновляем circuit breaker
                self._update_circuit_breaker(True)
                
                return response
            else:
                error_msg = f"Bybit API error: {response.get('retMsg', 'Unknown error')}"
                self.logger.error(f"❌ {error_msg}")
                raise Exception(error_msg)
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка размещения ордера: {e}")
            
            # Обновляем статистику ошибок
            self._update_stats('errors')
            
            # Обновляем circuit breaker
            self._update_circuit_breaker(False)
            
            raise
    
    @bybit_rate_limiter.rest_api_call
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, Exception))
    )
    def get_account_balance(self, account_type: str = "UNIFIED") -> Decimal:
        """Получение баланса аккаунта"""
        try:
            response = self.client.get_wallet_balance(accountType=account_type)
            
            if response.get('retCode') == 0:
                balance_data = response.get('result', {}).get('list', [{}])[0]
                coin_data = balance_data.get('coin', [{}])[0]
                balance = Decimal(coin_data.get('walletBalance', '0'))
                
                self.logger.debug(f"💰 Баланс: {balance}")
                return balance
            else:
                raise Exception(f"Error getting balance: {response.get('retMsg')}")
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения баланса: {e}")
            raise
    
    @bybit_rate_limiter.order_call
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, Exception))
    )
    def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Отмена ордера"""
        try:
            response = self.client.cancel_order(symbol=symbol, orderId=order_id)
            
            if response.get('retCode') == 0:
                self.logger.info(f"✅ Ордер отменен: {order_id}")
                
                # Обновляем статистику
                self._update_stats('orders_canceled')
                
                return response
            else:
                raise Exception(f"Error canceling order: {response.get('retMsg')}")
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка отмены ордера: {e}")
            raise
    
    @bybit_rate_limiter.rest_api_call
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, Exception))
    )
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Получение открытых позиций"""
        try:
            params = {}
            if symbol:
                params['symbol'] = symbol
                
            response = self.client.get_positions(**params)
            
            if response.get('retCode') == 0:
                positions = response.get('result', {}).get('list', [])
                self.logger.debug(f"📊 Найдено позиций: {len(positions)}")
                
                # Обновляем кэш позиций (потокобезопасно)
                with self._position_lock:
                    for position in positions:
                        pos_symbol = position.get('symbol')
                        if pos_symbol:
                            self.position_cache[pos_symbol] = position
                
                return positions
            else:
                raise Exception(f"Error getting positions: {response.get('retMsg')}")
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения позиций: {e}")
            raise
    
    def emergency_stop(self):
        """Экстренная остановка - закрытие всех позиций"""
        self.logger.critical("🚨 EMERGENCY STOP ACTIVATED")
        
        try:
            # Получаем все позиции
            positions = self.get_positions()
            
            for position in positions:
                symbol = position.get('symbol')
                size = Decimal(position.get('size', '0'))
                
                if size > 0:
                    # Определяем направление закрытия
                    side = OrderSide.SELL if position.get('side') == 'Buy' else OrderSide.BUY
                    
                    # Закрываем позицию
                    self.place_order_with_retry(
                        symbol=symbol,
                        side=side,
                        order_type=OrderType.MARKET,
                        quantity=size,
                        strategy_name="emergency_stop"
                    )
                    
                    self.logger.info(f"🚨 Закрыта позиция: {symbol} {size}")
            
            self.logger.critical("🚨 EMERGENCY STOP COMPLETED")
            
        except Exception as e:
            self.logger.critical(f"🚨 EMERGENCY STOP FAILED: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики (потокобезопасно)"""
        with self._lock:
            return self.stats.copy()
    
    def get_rate_limiter_status(self) -> Dict[str, Any]:
        """Получение статуса rate limiter"""
        return bybit_rate_limiter.get_status()
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Получение статуса circuit breaker"""
        return advanced_circuit_breaker.get_status()

# Пример использования
if __name__ == "__main__":
    # Тестирование (только с тестовыми ключами)
    executor = ThreadSafeExecutor(
        api_key="test_key",
        api_secret="test_secret",
        testnet=True
    )
    
    print("✅ Thread-Safe Executor готов к работе")
    print(f"📊 Статистика: {executor.get_stats()}")
    print(f"🛡️ Rate Limiter: {executor.get_rate_limiter_status()}")
    print(f"🚨 Circuit Breaker: {executor.get_circuit_breaker_status()}")
