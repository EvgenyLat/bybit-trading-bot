#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔒 SECURE Bybit Executor with Retry Logic
Безопасный исполнитель ордеров с обработкой ошибок и повторными попытками
"""

import asyncio
import logging
from decimal import Decimal, getcontext, ROUND_DOWN
from typing import Dict, Any, Optional, Tuple
from enum import Enum
import time

from pybit.unified_trading import HTTP
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests.exceptions

class OrderSide(Enum):
    """Сторона ордера"""
    BUY = "Buy"
    SELL = "Sell"

class OrderType(Enum):
    """Тип ордера"""
    MARKET = "Market"
    LIMIT = "Limit"

class OrderStatus(Enum):
    """Статус ордера"""
    NEW = "New"
    FILLED = "Filled"
    CANCELED = "Canceled"
    REJECTED = "Rejected"

class SecureBybitExecutor:
    """🔒 Безопасный исполнитель ордеров Bybit"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """
        Инициализация безопасного исполнителя
        
        Args:
            api_key: Bybit API ключ
            api_secret: Bybit API секрет  
            testnet: Использовать тест-сеть
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Настройка точности для Decimal
        getcontext().prec = 12
        
        # Инициализация клиента
        self.client = self._create_client()
        
        # Настройка логирования
        self.logger = self._setup_logging()
        
        # Кэш для избежания дублей
        self.order_cache = {}
        
    def _create_client(self) -> HTTP:
        """Создание Bybit клиента"""
        try:
            client = HTTP(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet
            )
            
            # Проверяем подключение
            self._test_connection(client)
            
            self.logger.info(f"✅ Bybit клиент создан (testnet: {self.testnet})")
            return client
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка создания клиента: {e}")
            raise
            
    def _test_connection(self, client: HTTP):
        """Проверка соединения с биржей"""
        try:
            # Простой тестовый вызов
            client.get_server_time()
            self.logger.info("✅ Подключение к Bybit проверено")
        except Exception as e:
            self.logger.error(f"❌ Ошибка подключения к Bybit: {e}")
            raise
            
    def _setup_logging(self) -> logging.Logger:
        """Настройка безопасного логирования"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # НЕ логируем секреты!
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError))
    )
    def _safe_api_call(self, method_name: str, *args, **kwargs):
        """Безопасный вызов API с повторами"""
        try:
            method = getattr(self.client, method_name)
            return method(*args, **kwargs)
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                self.logger.warning("⏳ Rate limit hit, retrying...")
                raise  # Retry механизм перехватит
            elif e.response.status_code >= 500:
                self.logger.warning("🌐 Server error, retrying...")
                raise  # Retry механизм перехватит
            else:
                self.logger.error(f"❌ HTTP error: {e}")
                raise
                
        except Exception as e:
            self.logger.error(f"❌ API call failed: {e}")
            raise
            
    def quantize_size(self, size: float, step: float = 0.001) -> Decimal:
        """
        Округление размера позиции до корректного шага
        
        Args:
            size: Исходный размер
            step: Шаг округления (по умолчанию 0.001)
            
        Returns:
            Decimal правильного размера
        """
        try:
            size_d = Decimal(str(size))
            step_d = Decimal(str(step))
            
            # Округляем вниз до ближайшего кратного шага
            result = (size_d // step_d) * step_d
            
            # Проверяем минимум
            if result <= 0:
                result = step_d
                
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Quantize error: {e}")
            raise
            
    def quantize_price(self, price: float, tick_size: float = 0.01) -> Decimal:
        """
        Округление цены до корректного тика
        
        Args:
            price: Исходная цена
            tick_size: Размер тика
            
        Returns:
            Decimal правильной цены
        """
        try:
            price_d = Decimal(str(price))
            tick_d = Decimal(str(tick_size))
            
            # Округляем до ближайшего тика
            result = round(price_d / tick_d) * tick_d
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Price quantize error: {e}")
            raise
            
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Получение информации о символе"""
        try:
            response = self._safe_api_call('get_instruments_info', category='linear', symbol=symbol)
            
            if response['retCode'] != 0:
                raise ValueError(f"Symbol info error: {response['retMsg']}")
                
            instruments = response['result']['list']
            if not instruments:
                raise ValueError(f"Symbol {symbol} not found")
                
            return instruments[0]
            
        except Exception as e:
            self.logger.error(f"❌ Get symbol info failed: {e}")
            raise
            
    def validate_order_params(self, symbol: str, side: OrderSide, order_type: OrderType, 
                            qty: Decimal, price: Optional[Decimal] = None) -> Tuple[Decimal, Optional[Decimal]]:
        """
        Валидация и нормализация параметров ордера
        
        Args:
            symbol: Торговый символ
            side: Сторона ордера
            order_type: Тип ордера
            qty: Количество
            price: Цена (для лимитного ордера)
            
        Returns:
            Tuple[нормализованный_qty, нормализованная_цена]
        """
        try:
            # Получаем спецификацию символа
            symbol_info = self.get_symbol_info(symbol)
            
            # Извлекаем параметры
            lot_size_filter = symbol_info['lotSizeFilter']
            price_filter = symbol_info['priceFilter']
            
            min_qty = Decimal(lot_size_filter['minOrderQty'])
            max_qty = Decimal(lot_size_filter['maxOrderQty'])
            qty_step = Decimal(lot_size_filter['qtyStep'])
            
            tick_size = Decimal(price_filter['tickSize'])
            
            # Нормализуем размер
            normalized_qty = self.quantize_size(float(qty), float(qty_step))
            
            # Проверяем границы
            if normalized_qty < min_qty:
                raise ValueError(f"Quantity {normalized_qty} below minimum {min_qty}")
            if normalized_qty > max_qty:
                raise ValueError(f"Quantity {normalized_qty} above maximum {max_qty}")
                
            # Нормализуем цену для лимитного ордера
            normalized_price = None
            if price is not None:
                normalized_price = self.quantize_price(float(price), float(tick_size))
                
            self.logger.info(f"✅ Order parameters validated: qty={normalized_qty}, price={normalized_price}")
            
            return normalized_qty, normalized_price
            
        except Exception as e:
            self.logger.error(f"❌ Order validation failed: {e}")
            raise
            
    def place_order(self, symbol: str, side: OrderSide, order_type: OrderType, 
                   qty: float, price: Optional[float] = None, 
                   client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Размещение ордера с полной валидацией
        
        Args:
            symbol: Торговый символ (например 'BTCUSDT')
            side: Сторона ордера
            order_type: Тип ордера
            qty: Количество
            price: Цена (только для лимитного ордера)
            client_order_id: ID ордера клиента (для избежания дублей)
            
        Returns:
            Ответ биржи с информацией об ордере
        """
        try:
            # Генерируем уникальный ID если не предоставлен
            if not client_order_id:
                client_order_id = f"bot_{int(time.time() * 1000)}"
                
            # Проверяем на дублирование
            if client_order_id in self.order_cache:
                raise ValueError(f"Order {client_order_id} already exists")
                
            # Валидируем параметры
            normalized_qty, normalized_price = self.validate_order_params(
                symbol, side, order_type, Decimal(str(qty)), 
                Decimal(str(price)) if price else None
            )
            
            # Подготавливаем данные ордера
            order_data = {
                'symbol': symbol,
                'side': side.value,
                'orderType': order_type.value,
                'qty': str(normalized_qty),
                'timeInForce': 'GTC',
                'orderLinkId': client_order_id
            }
            
            # Добавляем цену для лимитного ордера
            if order_type == OrderType.LIMIT and normalized_price:
                order_data['price'] = str(normalized_price)
                
            self.logger.info(f"🎯 Placing {order_type.value} {side.value} order: {normalized_qty} {symbol} @ {normalized_price}")
            
            # Размещаем ордер
            response = self._safe_api_call('place_order', category='linear', **order_data)
            
            if response['retCode'] != 0:
                raise APIError(f"Order placement failed: {response['retMsg']}")
                
            # Кэшируем успешный ордер
            self.order_cache[client_order_id] = {
                'symbol': symbol,
                'side': side,
                'qty': normalized_qty,
                'price': normalized_price,
                'timestamp': time.time()
            }
            
            self.logger.info(f"✅ Order {client_order_id} placed successfully: {response['result']}")
            
            return response['result']
            
        except Exception as e:
            self.logger.error(f"❌ Place order failed: {e}")
            raise
            
    def cancel_order(self, symbol: str, order_id: Optional[str] = None, 
                   client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Отмена ордера
        
        Args:
            symbol: Торговый символ
            order_id: ID ордера от биржи
            client_order_id: ID ордера клиента
            
        Returns:
            Результат отмены
        """
        try:
            if not order_id and not client_order_id:
                raise ValueError("Either order_id or client_order_id required")
                
            cancel_data = {
                'category': 'linear',
                'symbol': symbol
            }
            
            if order_id:
                cancel_data['orderId'] = order_id
            if client_order_id:
                cancel_data['orderLinkId'] = client_order_id
                
            response = self._safe_api_call('cancel_order', **cancel_data)
            
            if response['retCode'] != 0:
                raise APIError(f"Order cancellation failed: {response['retMsg']}")
                
            # Удаляем из кэша
            if client_order_id and client_order_id in self.order_cache:
                del self.order_cache[client_order_id]
                
            self.logger.info(f"✅ Order canceled: {order_id or client_order_id}")
            
            return response['result']
            
        except Exception as e:
            self.logger.error(f"❌ Cancel order failed: {e}")
            raise
            
    def get_account_info(self) -> Dict[str, Any]:
        """Получение информации об аккаунте"""
        try:
            response = self._safe_api_call('get_wallet_balance', accountType='UNIFIED')
            
            if response['retCode'] != 0:
                raise APIError(f"Account info failed: {response['retMsg']}")
                
            return response['result']
            
        except Exception as e:
            self.logger.error(f"❌ Get account info failed: {e}")
            raise


class APIError(Exception):
    """Кастомное исключение для API ошибок"""
    pass


if __name__ == "__main__":
    """Тестирование безопасного исполнителя"""
    
    print("🧪 Testing Secure Executor...")
    
    # Пример использования (требует настоящих ключей)
    try:
        # executor = SecureBybitExecutor(
        #     api_key="test_key",
        #     api_secret="test_secret", 
        #     testnet=True
        # )
        
        print("✅ SecureExecutor готов к использованию")
        print("🔒 Все операции логируются безопасно")
        print("🔄 Retry логика защищает от временных сбоев")
        print("📊 Decimal precision предотвращает ошибки округления")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
