#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛡️ Rate Limiter for API Calls
Защита от превышения лимитов API биржи
"""

import time
import threading
from collections import deque
from typing import Callable, Any
import logging

class RateLimiter:
    """🛡️ Rate limiter для защиты от превышения лимитов API"""
    
    def __init__(self, max_calls: int = 100, period: int = 60):
        """
        Инициализация rate limiter
        
        Args:
            max_calls: Максимальное количество вызовов
            period: Период в секундах
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
        self.lock = threading.Lock()
        self.logger = logging.getLogger('RateLimiter')
        
    def __call__(self, func: Callable) -> Callable:
        """Декоратор для применения rate limiting"""
        def wrapper(*args, **kwargs):
            self._wait_if_needed()
            return func(*args, **kwargs)
        return wrapper
    
    def _wait_if_needed(self):
        """Ожидание если превышен лимит"""
        with self.lock:
            now = time.time()
            
            # Удаляем старые вызовы
            while self.calls and self.calls[0] < now - self.period:
                self.calls.popleft()
            
            # Проверяем лимит
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                if sleep_time > 0:
                    self.logger.warning(f"⏰ Rate limit reached, sleeping {sleep_time:.2f}s")
                    time.sleep(sleep_time)
            
            # Добавляем текущий вызов
            self.calls.append(now)
    
    def get_remaining_calls(self) -> int:
        """Получение оставшихся вызовов"""
        with self.lock:
            now = time.time()
            # Удаляем старые вызовы
            while self.calls and self.calls[0] < now - self.period:
                self.calls.popleft()
            
            return self.max_calls - len(self.calls)
    
    def reset(self):
        """Сброс счетчика"""
        with self.lock:
            self.calls.clear()

class BybitRateLimiter:
    """🛡️ Специализированный rate limiter для Bybit API"""
    
    def __init__(self):
        """Инициализация с лимитами Bybit"""
        # Bybit лимиты:
        # - 120 requests per minute для REST API
        # - 10 requests per second для orders
        self.rest_limiter = RateLimiter(max_calls=120, period=60)
        self.order_limiter = RateLimiter(max_calls=10, period=1)
        self.logger = logging.getLogger('BybitRateLimiter')
    
    def rest_api_call(self, func: Callable) -> Callable:
        """Декоратор для REST API вызовов"""
        return self.rest_limiter(func)
    
    def order_call(self, func: Callable) -> Callable:
        """Декоратор для order API вызовов"""
        return self.order_limiter(func)
    
    def get_status(self) -> dict:
        """Получение статуса всех лимитеров"""
        return {
            'rest_remaining': self.rest_limiter.get_remaining_calls(),
            'order_remaining': self.order_limiter.get_remaining_calls(),
            'rest_limit': self.rest_limiter.max_calls,
            'order_limit': self.order_limiter.max_calls
        }

# Глобальный экземпляр для использования в проекте
bybit_rate_limiter = BybitRateLimiter()

# Пример использования
if __name__ == "__main__":
    # Тестирование rate limiter
    limiter = RateLimiter(max_calls=5, period=10)
    
    @limiter
    def test_function():
        print(f"Function called at {time.time()}")
        return "success"
    
    print("Testing rate limiter...")
    for i in range(8):
        result = test_function()
        print(f"Call {i+1}: {result}")
    
    print("Rate limiter test completed")
