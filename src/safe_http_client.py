#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛡️ Safe HTTP Client with Timeouts
Безопасный HTTP клиент с таймаутами и retry логикой
"""

import time
import requests
from typing import Dict, Any, Optional
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential, 
    retry_if_exception_type,
    before_sleep_log
)
import logging

class SafeHTTPClient:
    """🛡️ Безопасный HTTP клиент с таймаутами и retry"""
    
    def __init__(self, timeout: int = 10, max_retries: int = 3):
        """
        Инициализация безопасного HTTP клиента
        
        Args:
            timeout: Таймаут для запросов в секундах
            max_retries: Максимальное количество повторов
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logging.getLogger('SafeHTTPClient')
        
        # Создаем сессию с настройками
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'BybitTradingBot/1.0',
            'Content-Type': 'application/json'
        })
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError
        )),
        before_sleep=before_sleep_log(logging.getLogger('SafeHTTPClient'), logging.WARNING)
    )
    def safe_get(self, url: str, params: Optional[Dict] = None, **kwargs) -> requests.Response:
        """Безопасный GET запрос с таймаутом"""
        kwargs.setdefault('timeout', self.timeout)
        
        try:
            response = self.session.get(url, params=params, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout:
            self.logger.warning(f"⏰ Timeout for GET {url}")
            raise
        except requests.exceptions.ConnectionError:
            self.logger.warning(f"🔌 Connection error for GET {url}")
            raise
        except requests.exceptions.HTTPError as e:
            self.logger.warning(f"🌐 HTTP error for GET {url}: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError
        )),
        before_sleep=before_sleep_log(logging.getLogger('SafeHTTPClient'), logging.WARNING)
    )
    def safe_post(self, url: str, data: Optional[Dict] = None, json: Optional[Dict] = None, **kwargs) -> requests.Response:
        """Безопасный POST запрос с таймаутом"""
        kwargs.setdefault('timeout', self.timeout)
        
        try:
            response = self.session.post(url, data=data, json=json, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout:
            self.logger.warning(f"⏰ Timeout for POST {url}")
            raise
        except requests.exceptions.ConnectionError:
            self.logger.warning(f"🔌 Connection error for POST {url}")
            raise
        except requests.exceptions.HTTPError as e:
            self.logger.warning(f"🌐 HTTP error for POST {url}: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError
        )),
        before_sleep=before_sleep_log(logging.getLogger('SafeHTTPClient'), logging.WARNING)
    )
    def safe_put(self, url: str, data: Optional[Dict] = None, json: Optional[Dict] = None, **kwargs) -> requests.Response:
        """Безопасный PUT запрос с таймаутом"""
        kwargs.setdefault('timeout', self.timeout)
        
        try:
            response = self.session.put(url, data=data, json=json, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout:
            self.logger.warning(f"⏰ Timeout for PUT {url}")
            raise
        except requests.exceptions.ConnectionError:
            self.logger.warning(f"🔌 Connection error for PUT {url}")
            raise
        except requests.exceptions.HTTPError as e:
            self.logger.warning(f"🌐 HTTP error for PUT {url}: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError
        )),
        before_sleep=before_sleep_log(logging.getLogger('SafeHTTPClient'), logging.WARNING)
    )
    def safe_delete(self, url: str, **kwargs) -> requests.Response:
        """Безопасный DELETE запрос с таймаутом"""
        kwargs.setdefault('timeout', self.timeout)
        
        try:
            response = self.session.delete(url, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout:
            self.logger.warning(f"⏰ Timeout for DELETE {url}")
            raise
        except requests.exceptions.ConnectionError:
            self.logger.warning(f"🔌 Connection error for DELETE {url}")
            raise
        except requests.exceptions.HTTPError as e:
            self.logger.warning(f"🌐 HTTP error for DELETE {url}: {e}")
            raise
    
    def close(self):
        """Закрытие сессии"""
        self.session.close()

# Декоратор для автоматического добавления таймаутов
def with_timeout(timeout: int = 10):
    """Декоратор для добавления таймаута к функциям"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            kwargs.setdefault('timeout', timeout)
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Пример использования
if __name__ == "__main__":
    client = SafeHTTPClient(timeout=15)
    
    try:
        # Пример безопасного запроса
        response = client.safe_get("https://api.bybit.com/v5/market/time")
        print(f"✅ Response: {response.status_code}")
        print(f"📊 Data: {response.json()}")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        client.close()
