#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛡️ SAFE Trade Executor 
Максимально безопасный исполнитель ордеров с проверками
"""

import logging
import os
from decimal import Decimal, ROUND_DOWN
from typing import Optional, Dict, Any
from enum import Enum
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class OrderType(Enum):
    LIMIT = "Limit"
    MARKET = "Market"

class OrderSide(Enum):
    BUY = "Buy"
    SELL = "Sell"

class OrderStatus(Enum):
    NEW = "New"
    PARTIALLY_FILLED = "PartiallyFilled"
    FILLED = "Filled"
    CANCELLED = "Cancelled"
    REJECTED = "Rejected"

class SafeBybitExecutor:
    """🛡️ Максимально безопасный исполнитель ордеров"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """
        Инициализация безопасного исполнителя
        
        Args:
            api_key: Bybit API ключ
            api_secret: Bybit API секрет
            testnet: Использовать тестовую сеть
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.logger = self._setup_logger()
        
        # Создаем HTTP клиент БЕЗ передачи секретов в логи
        self._setup_client()
        
    def _setup_logger(self):
        """Настройка безопасного логирования"""
        logger = logging.getLogger('SafeExecutor')
        logger.setLevel(logging.INFO)
        
        # НЕ ЛОГИРУЕМ чувствительные данные
        class SafeFormatter(logging.Formatter):
            def format(self, record):
                msg = super().format(record)
                # Удаляем потенциальные секреты из логов
                sensitive_patterns = [
                    r'(api_key|api_secret|token|password)[=:]\s*\w+',
                    r'(ghp_|gho_)[a-zA-Z0-9]{36}',
                    r'Bearer\s+\w+',
                ]
                for pattern in sensitive_patterns:
                    msg = re.sub(pattern, r'\1=***REDACTED***', msg, flags=re.IGNORECASE)
                return msg
        
        handler = logging.StreamHandler()
        handler.setFormatter(SafeFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        
        return logger
        
    def _setup_client(self):
        """Настройка клиента с минимальным логированием"""
        try:
            from pybit.unified_trading import HTTP
            self.client = HTTP(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet
            )
            self.logger.info("✅ Bybit client initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Bybit client: {e}")
            raise
            
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(Exception)
    )
    def place_order_with_retry(self, 
                             symbol: str,
                             side: OrderSide,
                             order_type: OrderType,
                             quantity: str,
                             price: Optional[str] = None,
                             **params) -> Dict[str, Any]:
        """
        Размещение ордера с автоматическими повторами
        
        Args:
            symbol: Торговая пара (например, BTCUSDT)
            side: Направление ордера (BUY/SELL)
            order_type: Тип ордера (LIMIT/MARKET)
            quantity: Размер позиции
            price: Цена (для лимитных ордеров)
            **params: Дополнительные параметры
        """
        
        # Безопасная валидация параметров
        self._validate_order_params(symbol, side, order_type, quantity, price)
        
        # Подготовка параметров ордера
        order_params = {
            'category': 'linear',
            'symbol': symbol,
            'side': side.value,
            'orderType': order_type.value,
            'qty': quantity,
            'clientOrderId': self._generate_client_order_id()
        }
        
        # Для лимитных ордеров добавляем цену
        if order_type == OrderType.LIMIT and price:
            order_params['price'] = price
            
        # Безопасное логирование
        self.logger.info(f"📝 Placing order: {order_type.value} {side.value} {quantity} {symbol}")
        if price:
            self.logger.info(f"💰 Price: {price}")
            
        try:
            # Размещение ордера
            result = self.client.place_order(**order_params)
            
            if result.get('retCode') == 0:
                order_id = result['result'].get('orderId')
                self.logger.info(f"✅ Order placed successfully: {order_id}")
                return result
            else:
                error_msg = result.get('retMsg', 'Unknown error')
                self.logger.error(f"❌ Order placement failed: {error_msg}")
                raise Exception(f"Order placement failed: {error_msg}")
                
        except Exception as e:
            self.logger.error(f"❌ Error placing order: {e}")
            raise
            
    def _validate_order_params(self, symbol: str, side: OrderSide, 
                             order_type: OrderType, quantity: str, price: Optional[str]):
        """Валидация параметров ордера"""
        
        # Проверка символа
        if not symbol or len(symbol) < 4:
            raise ValueError("Invalid symbol")
            
        # Проверка направления
        if side not in [OrderSide.BUY, OrderSide.SELL]:
            raise ValueError("Invalid order side")
            
        # Проверка типа ордера
        if order_type not in [OrderType.LIMIT, OrderType.MARKET]:
            raise ValueError("Invalid order type")
            
        # Проверка количества
        try:
            qty_decimal = Decimal(quantity)
            if qty_decimal <= 0:
                raise ValueError("Quantity must be positive")
        except Exception:
            raise ValueError("Invalid quantity format")
            
        # Проверка цены для лимитных ордеров
        if order_type == OrderType.LIMIT:
            if not price:
                raise ValueError("Price required for limit orders")
            try:
                price_decimal = Decimal(price)
                if price_decimal <= 0:
                    raise ValueError("Price must be positive")
            except Exception:
                raise ValueError("Invalid price format")
                
    def _generate_client_order_id(self) -> str:
        """Генерация уникального ID ордера"""
        import uuid
        import time
        
        timestamp = int(time.time() * 1000)
        uuid_part = str(uuid.uuid4())[:8]
        return f"BTCBot_{timestamp}_{uuid_part}"
        
    def calculate_position_size(self, account_balance: float, risk_pct: float, 
                               stop_loss_distance: float) -> str:
        """
        Безопасный расчет размера позиции с Decimal
        
        Args:
            account_balance: Баланс аккаунта
            risk_pct: Процент риска (например, 0.02 для 2%)
            stop_loss_distance: Расстояние до стоп-лосса
            
        Returns:
            Размер позиции в виде строки
        """
        
        try:
            # Конвертируем в Decimal для точности
            balance_decimal = Decimal(str(account_balance))
            risk_decimal = Decimal(str(risk_pct))
            stop_distance_decimal = Decimal(str(stop_loss_distance))
            
            # Проверяем входные данные
            if balance_decimal <= 0:
                raise ValueError("Account balance must be positive")
            if risk_decimal <= 0 or risk_decimal > Decimal('0.1'):
                raise ValueError("Risk percentage must be between 0 and 10%")
            if stop_distance_decimal <= 0:
                raise ValueError("Stop loss distance must be positive")
                
            # Расчет размера позиции
            risk_amount = balance_decimal * risk_decimal
            position_size = risk_amount / stop_distance_decimal
            
            # Округляем до допустимого размера
            position_size = position_size.quantize(Decimal('0.001'), rounding=ROUND_DOWN)
            
            result = str(position_size)
            self.logger.info(f"💰 Calculated position size: {result}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating position size: {e}")
            raise
            
    def emergency_stop(self):
        """Экстренная остановка всех операций"""
        self.logger.warning("🚨 EMERGENCY STOP ACTIVATED!")
        
        try:
            # Отменяем все открытые ордера
            cancel_result = self.client.cancel_all_orders(
                category='linear',
                settleCoin='USDT'
            )
            
            if cancel_result.get('retCode') == 0:
                cancelled_count = len(cancel_result['result'].get('list', []))
                self.logger.warning(f"🛑 Cancelled {cancelled_count} pending orders")
            else:
                self.logger.error("❌ Failed to cancel some orders")
                
        except Exception as e:
            self.logger.error(f"❌ Emergency stop error: {e}")
            
        self.logger.warning("🚨 Emergency stop completed")
