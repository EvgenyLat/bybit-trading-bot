#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📈 Trailing Stop Loss System
Система трейлинг стоп-лосса для максимизации прибыли
"""

import time
import logging
from typing import Optional, Dict, Any
from decimal import Decimal
from enum import Enum

class TrailingStopType(Enum):
    PERCENTAGE = "percentage"
    ATR = "atr"
    FIXED = "fixed"

class TrailingStop:
    """📈 Trailing stop loss для защиты прибыли"""
    
    def __init__(self, 
                 activation_pct: float = 0.01,  # 1% прибыль для активации
                 trail_pct: float = 0.005,      # 0.5% трейлинг
                 trail_type: TrailingStopType = TrailingStopType.PERCENTAGE,
                 max_trail_distance: float = 0.05):  # 5% максимальное расстояние
        """
        Инициализация trailing stop
        
        Args:
            activation_pct: Процент прибыли для активации трейлинга
            trail_pct: Процент трейлинга
            trail_type: Тип трейлинга (percentage, atr, fixed)
            max_trail_distance: Максимальное расстояние трейлинга
        """
        self.activation_pct = activation_pct
        self.trail_pct = trail_pct
        self.trail_type = trail_type
        self.max_trail_distance = max_trail_distance
        
        # Состояние
        self.highest_price = None
        self.lowest_price = None
        self.activated = False
        self.entry_price = None
        self.current_stop_price = None
        
        self.logger = logging.getLogger('TrailingStop')
    
    def initialize(self, entry_price: Decimal, side: str):
        """
        Инициализация trailing stop
        
        Args:
            entry_price: Цена входа
            side: Направление позиции ('buy' или 'sell')
        """
        self.entry_price = entry_price
        self.side = side
        self.activated = False
        self.highest_price = None
        self.lowest_price = None
        self.current_stop_price = None
        
        self.logger.info(f"📈 Trailing stop initialized: {side} @ {entry_price}")
    
    def update(self, current_price: Decimal, atr: Optional[Decimal] = None) -> Optional[Decimal]:
        """
        Обновляет trailing stop
        
        Args:
            current_price: Текущая цена
            atr: ATR для ATR-based трейлинга
            
        Returns:
            Цену стопа если нужно закрыть позицию, иначе None
        """
        if self.entry_price is None:
            self.logger.warning("⚠️ Trailing stop not initialized")
            return None
        
        if self.side == 'buy':
            return self._update_long_position(current_price, atr)
        else:
            return self._update_short_position(current_price, atr)
    
    def _update_long_position(self, current_price: Decimal, atr: Optional[Decimal] = None) -> Optional[Decimal]:
        """Обновление для лонг позиции"""
        profit_pct = (current_price - self.entry_price) / self.entry_price
        
        # Активируем trailing stop при достижении прибыли
        if not self.activated and profit_pct >= self.activation_pct:
            self.activated = True
            self.highest_price = current_price
            self.logger.info(f"✅ Trailing stop activated at {current_price} (profit: {profit_pct:.2%})")
        
        if self.activated:
            # Обновляем максимум
            if current_price > self.highest_price:
                self.highest_price = current_price
            
            # Рассчитываем стоп
            if self.trail_type == TrailingStopType.PERCENTAGE:
                stop_price = self.highest_price * (1 - self.trail_pct)
            elif self.trail_type == TrailingStopType.ATR and atr:
                stop_price = self.highest_price - (atr * Decimal('2'))
            else:  # FIXED
                stop_price = self.highest_price - Decimal(str(self.trail_pct))
            
            # Ограничиваем максимальное расстояние
            max_stop_distance = self.highest_price * self.max_trail_distance
            min_stop_price = self.highest_price - max_stop_distance
            stop_price = max(stop_price, min_stop_price)
            
            self.current_stop_price = stop_price
            
            # Проверяем срабатывание стопа
            if current_price <= stop_price:
                self.logger.info(f"🛑 Trailing stop hit: {current_price} <= {stop_price}")
                return stop_price
        
        return None
    
    def _update_short_position(self, current_price: Decimal, atr: Optional[Decimal] = None) -> Optional[Decimal]:
        """Обновление для шорт позиции"""
        profit_pct = (self.entry_price - current_price) / self.entry_price
        
        if not self.activated and profit_pct >= self.activation_pct:
            self.activated = True
            self.lowest_price = current_price
            self.logger.info(f"✅ Trailing stop activated at {current_price} (profit: {profit_pct:.2%})")
        
        if self.activated:
            if current_price < self.lowest_price:
                self.lowest_price = current_price
            
            if self.trail_type == TrailingStopType.PERCENTAGE:
                stop_price = self.lowest_price * (1 + self.trail_pct)
            elif self.trail_type == TrailingStopType.ATR and atr:
                stop_price = self.lowest_price + (atr * Decimal('2'))
            else:  # FIXED
                stop_price = self.lowest_price + Decimal(str(self.trail_pct))
            
            # Ограничиваем максимальное расстояние
            max_stop_distance = self.lowest_price * self.max_trail_distance
            max_stop_price = self.lowest_price + max_stop_distance
            stop_price = min(stop_price, max_stop_price)
            
            self.current_stop_price = stop_price
            
            if current_price >= stop_price:
                self.logger.info(f"🛑 Trailing stop hit: {current_price} >= {stop_price}")
                return stop_price
        
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Получение статуса trailing stop"""
        return {
            'activated': self.activated,
            'entry_price': float(self.entry_price) if self.entry_price else None,
            'current_stop_price': float(self.current_stop_price) if self.current_stop_price else None,
            'highest_price': float(self.highest_price) if self.highest_price else None,
            'lowest_price': float(self.lowest_price) if self.lowest_price else None,
            'side': getattr(self, 'side', None),
            'trail_type': self.trail_type.value
        }

class AdaptiveTrailingStop(TrailingStop):
    """📈 Адаптивный trailing stop с динамическими параметрами"""
    
    def __init__(self, 
                 base_activation_pct: float = 0.01,
                 base_trail_pct: float = 0.005,
                 volatility_multiplier: float = 1.5):
        """
        Инициализация адаптивного trailing stop
        
        Args:
            base_activation_pct: Базовый процент активации
            base_trail_pct: Базовый процент трейлинга
            volatility_multiplier: Множитель волатильности
        """
        super().__init__(base_activation_pct, base_trail_pct)
        self.base_activation_pct = base_activation_pct
        self.base_trail_pct = base_trail_pct
        self.volatility_multiplier = volatility_multiplier
        self.current_volatility = None
    
    def update_volatility(self, volatility: float):
        """Обновление текущей волатильности"""
        self.current_volatility = volatility
        
        # Адаптируем параметры под волатильность
        if volatility > 0.05:  # Высокая волатильность
            self.activation_pct = self.base_activation_pct * 1.5
            self.trail_pct = self.base_trail_pct * 1.5
        elif volatility < 0.02:  # Низкая волатильность
            self.activation_pct = self.base_activation_pct * 0.7
            self.trail_pct = self.base_trail_pct * 0.7
        else:  # Средняя волатильность
            self.activation_pct = self.base_activation_pct
            self.trail_pct = self.base_trail_pct
        
        self.logger.debug(f"📊 Volatility updated: {volatility:.3f}, "
                         f"activation: {self.activation_pct:.3f}, "
                         f"trail: {self.trail_pct:.3f}")

class TrailingStopManager:
    """📈 Менеджер множественных trailing stops"""
    
    def __init__(self):
        """Инициализация менеджера"""
        self.trailing_stops = {}
        self.logger = logging.getLogger('TrailingStopManager')
    
    def add_trailing_stop(self, 
                         position_id: str,
                         entry_price: Decimal,
                         side: str,
                         trail_type: TrailingStopType = TrailingStopType.PERCENTAGE,
                         **kwargs) -> TrailingStop:
        """
        Добавление trailing stop для позиции
        
        Args:
            position_id: Уникальный ID позиции
            entry_price: Цена входа
            side: Направление позиции
            trail_type: Тип трейлинга
            **kwargs: Дополнительные параметры
            
        Returns:
            Созданный trailing stop
        """
        if trail_type == TrailingStopType.ATR:
            trailing_stop = AdaptiveTrailingStop(**kwargs)
        else:
            trailing_stop = TrailingStop(trail_type=trail_type, **kwargs)
        
        trailing_stop.initialize(entry_price, side)
        self.trailing_stops[position_id] = trailing_stop
        
        self.logger.info(f"📈 Added trailing stop for position {position_id}")
        return trailing_stop
    
    def update_position(self, 
                       position_id: str, 
                       current_price: Decimal,
                       atr: Optional[Decimal] = None,
                       volatility: Optional[float] = None) -> Optional[Decimal]:
        """
        Обновление позиции и проверка trailing stop
        
        Args:
            position_id: ID позиции
            current_price: Текущая цена
            atr: ATR для ATR-based трейлинга
            volatility: Волатильность для адаптивного трейлинга
            
        Returns:
            Цену стопа если нужно закрыть позицию
        """
        if position_id not in self.trailing_stops:
            return None
        
        trailing_stop = self.trailing_stops[position_id]
        
        # Обновляем волатильность для адаптивного трейлинга
        if isinstance(trailing_stop, AdaptiveTrailingStop) and volatility:
            trailing_stop.update_volatility(volatility)
        
        # Обновляем trailing stop
        stop_price = trailing_stop.update(current_price, atr)
        
        if stop_price:
            self.logger.info(f"🛑 Trailing stop triggered for position {position_id}")
            # Удаляем trailing stop после срабатывания
            del self.trailing_stops[position_id]
        
        return stop_price
    
    def remove_trailing_stop(self, position_id: str):
        """Удаление trailing stop"""
        if position_id in self.trailing_stops:
            del self.trailing_stops[position_id]
            self.logger.info(f"📈 Removed trailing stop for position {position_id}")
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Получение статуса всех trailing stops"""
        return {
            position_id: trailing_stop.get_status()
            for position_id, trailing_stop in self.trailing_stops.items()
        }
    
    def get_active_count(self) -> int:
        """Получение количества активных trailing stops"""
        return len(self.trailing_stops)

# Глобальный экземпляр для использования в проекте
trailing_stop_manager = TrailingStopManager()

# Пример использования
if __name__ == "__main__":
    # Тестирование trailing stop
    ts = TrailingStop(activation_pct=0.02, trail_pct=0.01)
    
    # Симуляция лонг позиции
    ts.initialize(Decimal('100'), 'buy')
    
    print("Testing trailing stop for long position...")
    
    # Симуляция роста цены
    prices = [100, 101, 102, 103, 104, 105, 104, 103, 102]
    
    for price in prices:
        stop_price = ts.update(Decimal(str(price)))
        status = ts.get_status()
        print(f"Price: {price}, Stop: {stop_price}, Status: {status['activated']}")
        
        if stop_price:
            print(f"🛑 Stop triggered at {stop_price}")
            break
    
    print("Trailing stop test completed")
