#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚨 Circuit Breaker for Flash Crash Protection
Защита от аномальных движений рынка и флеш-крэшей
"""

import time
import logging
from collections import deque
from typing import Optional, Dict, Any
from decimal import Decimal

class CircuitBreaker:
    """🚨 Circuit breaker для защиты от аномальных движений рынка"""
    
    def __init__(self, 
                 max_price_move: float = 0.05,  # 5% максимальное движение
                 time_window: int = 60,         # окно времени в секундах
                 max_positions: int = 100):      # максимум позиций в истории
        """
        Инициализация circuit breaker
        
        Args:
            max_price_move: Максимальное движение цены (в процентах)
            time_window: Временное окно для анализа
            max_positions: Максимальное количество позиций в истории
        """
        self.max_price_move = max_price_move
        self.time_window = time_window
        self.price_history = deque(maxlen=max_positions)
        self.logger = logging.getLogger('CircuitBreaker')
        
        # Статистика
        self.anomalies_detected = 0
        self.last_anomaly_time = None
        
    def check_anomaly(self, current_price: float, symbol: str = "UNKNOWN") -> bool:
        """
        Проверка на аномальное движение цены
        
        Args:
            current_price: Текущая цена
            symbol: Торговая пара
            
        Returns:
            True если обнаружена аномалия
        """
        current_time = time.time()
        
        # Добавляем текущую цену в историю
        self.price_history.append((current_time, current_price))
        
        # Нужно минимум 2 точки для анализа
        if len(self.price_history) < 2:
            return False
        
        # Получаем цены в временном окне
        recent_prices = [
            price for timestamp, price in self.price_history 
            if current_time - timestamp <= self.time_window
        ]
        
        if len(recent_prices) < 2:
            return False
        
        # Вычисляем максимальное движение в окне
        min_price = min(recent_prices)
        max_price = max(recent_prices)
        
        # Проверяем аномальное движение
        price_range = max_price - min_price
        avg_price = sum(recent_prices) / len(recent_prices)
        price_move_pct = price_range / avg_price
        
        if price_move_pct > self.max_price_move:
            self.anomalies_detected += 1
            self.last_anomaly_time = current_time
            
            self.logger.critical(
                f"🚨 FLASH CRASH DETECTED! "
                f"Symbol: {symbol}, "
                f"Price move: {price_move_pct:.2%}, "
                f"Range: {min_price:.2f} - {max_price:.2f}, "
                f"Anomalies: {self.anomalies_detected}"
            )
            
            return True
        
        return False
    
    def get_market_volatility(self) -> float:
        """Получение текущей волатильности рынка"""
        if len(self.price_history) < 2:
            return 0.0
        
        current_time = time.time()
        recent_prices = [
            price for timestamp, price in self.price_history 
            if current_time - timestamp <= self.time_window
        ]
        
        if len(recent_prices) < 2:
            return 0.0
        
        # Вычисляем стандартное отклонение
        avg_price = sum(recent_prices) / len(recent_prices)
        variance = sum((price - avg_price) ** 2 for price in recent_prices) / len(recent_prices)
        volatility = (variance ** 0.5) / avg_price
        
        return volatility
    
    def should_trade(self, symbol: str = "UNKNOWN") -> bool:
        """
        Определение можно ли торговать
        
        Args:
            symbol: Торговая пара
            
        Returns:
            True если можно торговать
        """
        # Если недавно была аномалия, блокируем торговлю
        if self.last_anomaly_time:
            time_since_anomaly = time.time() - self.last_anomaly_time
            cooldown_period = 300  # 5 минут cooldown
            
            if time_since_anomaly < cooldown_period:
                self.logger.warning(
                    f"🚨 Trading blocked due to recent anomaly "
                    f"(cooldown: {cooldown_period - time_since_anomaly:.0f}s remaining)"
                )
                return False
        
        # Проверяем текущую волатильность
        volatility = self.get_market_volatility()
        max_volatility = 0.1  # 10% максимальная волатильность
        
        if volatility > max_volatility:
            self.logger.warning(
                f"⚠️ High volatility detected: {volatility:.2%}, "
                f"trading may be risky"
            )
            return False
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Получение статуса circuit breaker"""
        return {
            'anomalies_detected': self.anomalies_detected,
            'last_anomaly_time': self.last_anomaly_time,
            'current_volatility': self.get_market_volatility(),
            'can_trade': self.should_trade(),
            'price_history_size': len(self.price_history)
        }

class AdvancedCircuitBreaker:
    """🚨 Продвинутый circuit breaker с множественными проверками"""
    
    def __init__(self):
        """Инициализация продвинутого circuit breaker"""
        self.price_breaker = CircuitBreaker(max_price_move=0.05, time_window=60)
        self.volume_breaker = VolumeAnomalyDetector()
        self.correlation_breaker = CorrelationMonitor()
        self.logger = logging.getLogger('AdvancedCircuitBreaker')
        
        # Общее состояние
        self.trading_enabled = True
        self.emergency_stop_active = False
        
    def check_market_conditions(self, 
                              current_price: float,
                              volume: float,
                              symbol: str,
                              correlation_data: Optional[Dict] = None) -> bool:
        """
        Комплексная проверка рыночных условий
        
        Args:
            current_price: Текущая цена
            volume: Объем торгов
            symbol: Торговая пара
            correlation_data: Данные корреляции
            
        Returns:
            True если можно торговать
        """
        # Проверка ценовых аномалий
        if self.price_breaker.check_anomaly(current_price, symbol):
            self.trading_enabled = False
            self.emergency_stop_active = True
            return False
        
        # Проверка объемных аномалий
        if self.volume_breaker.check_volume_anomaly(volume, symbol):
            self.logger.warning(f"⚠️ Volume anomaly detected for {symbol}")
            return False
        
        # Проверка корреляции
        if correlation_data and not self.correlation_breaker.check_correlation(correlation_data):
            self.logger.warning(f"⚠️ Correlation risk detected for {symbol}")
            return False
        
        # Общая проверка
        return self.trading_enabled and not self.emergency_stop_active
    
    def emergency_stop(self):
        """Экстренная остановка торговли"""
        self.trading_enabled = False
        self.emergency_stop_active = True
        self.logger.critical("🚨 EMERGENCY STOP ACTIVATED - ALL TRADING HALTED")
    
    def reset(self):
        """Сброс circuit breaker"""
        self.trading_enabled = True
        self.emergency_stop_active = False
        self.price_breaker = CircuitBreaker()
        self.volume_breaker = VolumeAnomalyDetector()
        self.correlation_breaker = CorrelationMonitor()
        self.logger.info("✅ Circuit breaker reset")

class VolumeAnomalyDetector:
    """🔍 Детектор объемных аномалий"""
    
    def __init__(self, max_volume_spike: float = 5.0):
        """
        Инициализация детектора объемных аномалий
        
        Args:
            max_volume_spike: Максимальный всплеск объема (в разах)
        """
        self.max_volume_spike = max_volume_spike
        self.volume_history = deque(maxlen=100)
        self.logger = logging.getLogger('VolumeAnomalyDetector')
    
    def check_volume_anomaly(self, current_volume: float, symbol: str) -> bool:
        """Проверка объемной аномалии"""
        self.volume_history.append(current_volume)
        
        if len(self.volume_history) < 20:
            return False
        
        # Вычисляем средний объем
        avg_volume = sum(self.volume_history) / len(self.volume_history)
        
        # Проверяем всплеск
        if current_volume > avg_volume * self.max_volume_spike:
            self.logger.warning(
                f"📊 Volume spike detected: {symbol}, "
                f"Current: {current_volume:.0f}, "
                f"Average: {avg_volume:.0f}, "
                f"Spike: {current_volume/avg_volume:.1f}x"
            )
            return True
        
        return False

class CorrelationMonitor:
    """📊 Монитор корреляции между активами"""
    
    def __init__(self, max_correlation: float = 0.8):
        """
        Инициализация монитора корреляции
        
        Args:
            max_correlation: Максимальная допустимая корреляция
        """
        self.max_correlation = max_correlation
        self.logger = logging.getLogger('CorrelationMonitor')
    
    def check_correlation(self, correlation_data: Dict[str, float]) -> bool:
        """Проверка корреляции между активами"""
        for asset1, correlations in correlation_data.items():
            for asset2, correlation in correlations.items():
                if asset1 != asset2 and abs(correlation) > self.max_correlation:
                    self.logger.warning(
                        f"📊 High correlation detected: "
                        f"{asset1} vs {asset2}: {correlation:.2f}"
                    )
                    return False
        
        return True

# Глобальный экземпляр для использования в проекте
advanced_circuit_breaker = AdvancedCircuitBreaker()

# Пример использования
if __name__ == "__main__":
    # Тестирование circuit breaker
    breaker = CircuitBreaker(max_price_move=0.1, time_window=10)
    
    print("Testing circuit breaker...")
    
    # Симуляция нормальных цен
    for i in range(5):
        price = 100 + i * 0.01
        anomaly = breaker.check_anomaly(price, "BTCUSDT")
        print(f"Price: {price:.2f}, Anomaly: {anomaly}")
    
    # Симуляция флеш-крэша
    crash_price = 50  # 50% падение
    anomaly = breaker.check_anomaly(crash_price, "BTCUSDT")
    print(f"Crash Price: {crash_price:.2f}, Anomaly: {anomaly}")
    
    print("Circuit breaker test completed")
