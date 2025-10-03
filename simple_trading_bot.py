#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простой торговый бот для Bybit
Без машинного обучения - только классический технический анализ
"""

import pandas as pd
import numpy as np
import ta
import time
from datetime import datetime
import yaml
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SimpleTradingBot:
    """Простой торговый бот для Bybit без ML"""
    
    def __init__(self):
        """Инициализация бота"""
        self.data = []
        self.position = None
        self.balance = 1000  # Стартовый баланс $1000
        self.last_signal = None
        self.trade_history = []
        
        # Настройки стратегии (можно менять)
        self.params = {
            'rsi_oversold': 30,    # RSI ниже 30 - перепроданность (покупка)
            'rsi_overbought': 70,  # RSI выше 70 - перекупленность (продажа)
            'sma_fast': 5,         # Быстрая скользящая средняя
            'sma_slow': 20,        # Медленная скользящая средняя
            'risk_per_trade': 0.02, # Риск 2% на сделку
            'take_profit': 0.03,   # Целевая прибыль 3%
            'stop_loss': 0.015     # Стоп-лосс 1.5%
        }
        
        print("🤖 Простой торговый бот инициализирован!")
        print(f"💰 Стартовый баланс: ${self.balance}")
        print("📊 Стратегия: RSI + SMA crossover")
        
    def add_price_data(self, price_data):
        """Добавление новых ценовых данных"""
        self.data.append({
            'timestamp': datetime.now(),
            'close': price_data,
            'volume': 1000  # Фиксированный объем для теста
        })
        
        # Храним только последние 100 точек
        if len(self.data) > 100:
            self.data.pop(0)
    
    def calculate_technical_indicators(self):
        """Расчет технических индикаторов"""
        if len(self.data) < self.params['sma_slow']:
            return None
            
        # Создаем DataFrame из наших данных
        df = pd.DataFrame(self.data)
        
        # Технические индикаторы
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['sma_fast'] = ta.trend.sma(df['close'], window=self.params['sma_fast'])
        df['sma_slow'] = ta.trend.sma(df['close'], window=self.params['sma_slow'])
        df['macd'] = ta.trend.macd(df['close'])
        
        return df.iloc[-1]  # Возвращаем последние значения
    
    def generate_signal(self):
        """Генерация торгового сигнала"""
        indicators = self.calculate_technical_indicators()
        if indicators is None:
            return 'WAIT', 'Недостаточно данных'
        
        signal_score = 0
        reasons = []
        
        # Правило 1: RSI сигналы
        if indicators['rsi'] < self.params['rsi_oversold']:
            signal_score += 3
            reasons.append(f"RSI перепроданность ({indicators['rsi']:.1f})")
        elif indicators['rsi'] > self.params['rsi_overbought']:
            signal_score -= 3
            reasons.append(f"RSI перекупленность ({indicators['rsi']:.1f})")
        
        # Правило 2: SMA crossover
        if indicators['sma_fast'] > indicators['sma_slow']:
            signal_score += 2
            reasons.append("Восходящий тренд SMA")
        elif indicators['sma_fast'] < indicators['sma_slow']:
            signal_score -= 2
            reasons.append("Нисходящий тренд SMA")
        
        # Правило 3: MACD сигнал
        if indicators['macd'] > 0:
            signal_score += 1
            reasons.append("MACD положительный")
        else:
            signal_score -= 1
            reasons.append("MACD отрицательный")
        
        # Финальное решение
        if self.position == 'long' and signal_score <= -3:
            return 'CLOSE_LONG', f"Сигнал закрытия LONG: {', '.join(reasons)}"
        elif self.position == 'short' and signal_score >= 3:
            return 'CLOSE_SHORT', f"Сигнал закрытия SHORT: {', '.join(reasons)}"
        elif signal_score >= 4:
            return 'BUY', f"Сильный сигнал покупки: {', '.join(reasons)}"
        elif signal_score >= 2:
            return 'BUY', f"Сигнал покупки: {', '.join(reasons)}"
        elif signal_score <= -4:
            return 'SELL', f"Сильный сигнал продажи: {', '.join(reasons)}"
        elif signal_score <= -2:
            return 'SELL', f"Сигнал продажи: {', '.join(reasons)}"
        else:
            return 'HOLD', f"Нейтральный сигнал (score: {signal_score})"
    
    def calculate_position_size(self, signal, current_price):
        """Расчет размера позиции"""
        if not self.data:
            return 0
            
        # Используем риск-менеджмент
        balance_usd = self.balance
        
        if signal in ['BUY', 'SELL']:
            # Размер позиции на основе риска
            risk_amount = balance_usd * self.params['risk_per_trade']
            stop_distance = current_price * self.params['stop_loss']
            position_size_usd = risk_amount / stop_distance * current_price
            
            # Ограничиваем размер позиции максимум 10% от баланса
            max_position_size = balance_usd * 0.1
            position_size_usd = min(position_size_usd, max_position_size)
            
            return position_size_usd
        
        return 0
    
    def execute_trade(self, signal, current_price):
        """Исполнение торговых сигналов"""
        if signal == 'WAIT':
            return
            
        timestamp = datetime.now()
        
        # Закрытие позиций
        if signal in ['CLOSE_LONG', 'CLOSE_SHORT'] and self.position:
            # Симуляция закрытия позиции
            trade_pnl = current_price - self.position['entry_price']
            if self.position['side'] == 'short':
                trade_pnl = -trade_pnl
            
            self.balance += trade_pnl
            self.trade_history.append({
                'timestamp': timestamp,
                'action': 'CLOSE',
                'side': self.position['side'],
                'price': current_price,
                'pnl': trade_pnl,
                'balance': self.balance
            })
            
            print(f"🚪 Закрыли {self.position['side']} позицию за ${trade_pnl:.2f}")
            self.position = None
            return
        
        # Открытие новых позиций
        if signal in ['BUY', 'SELL'] and self.position is None:
            position_size =(self.calculate_position_size(signal, current_price))
            
            if position_size > 10:  # Минимальная позиция $10
                side = 'long' if signal == 'BUY' else 'short'
                self.position = {
                    'side': side,
                    'entry_price': current_price,
                    'size': position_size,
                    'timestamp': timestamp
                }
                
                self.trade_history.append({
                    'timestamp': timestamp,
                    'action': 'OPEN',
                    'side': side,
                    'price': current_price,
                    'size': position_size,
                    'balance': self.balance
                })
                
                print(f"🚪 Открыли {side} позицию по цене ${current_price:.2f}")
    
    def get_status(self):
        """Получение статуса бота"""
        if self.position:
            current_pnl = self.data[-1]['close'] - self.position['entry_price']
            if self.position['side'] == 'short':
                current_pnl = -current_pnl
            
            return {
                'status': 'ACTIVE',
                'balance': self.balance,
                'position': self.position,
                'current_pnl': current_pnl,
                'total_trades': len(self.trade_history)
            }
        else:
            return {
                'status': 'WAITING',
                'balance': self.balance,
                'position': None,
                'total_trades': len(self.trade_history)
            }
    
    def save_state(self):
        """Сохранение состояния бота"""
        state = {
            'balance': self.balance,
            'position': self.position,
            'trade_history': self.trade_history,
            'last_update': datetime.now().isoformat()
        }
        
        with open('bot_state.yaml', 'w') as f:
            yaml.dump(state, f)
    
    def print_summary(self):
        """Вывод сводки по боту"""
        status = self.get_status()
        
        print("\n" + "="*50)
        print("📊 СВОДКА ТОРГОВОГО БОТА")
        print("="*50)
        print(f"💰 Баланс: ${status['balance']:.2f}")
        print(f"📈 Статус: {status['status']}")
        print(f"🔄 Торгов: {status['total_trades']}")
        
        if status['position']:
            print(f"📋 Позиция: {status['position']['side']} @ ${status['position']['entry_price']:.2f}")
            print(f"💵 Текущий PnL: ${status['current_pnl']:.2f}")
        
        if len(self.trade_history) > 0:
            recent_trades = self.trade_history[-3:]
            print("\n🔄 Последние сделки:")
            for trade in recent_trades:
                print(f"  {trade['timestamp'].strftime('%H:%M')} {trade['action']} {trade['side']} @ ${trade['price']:.2f}")
        
        print("="*50)


def main():
    """Основная функция для тестирования бота"""
    print("🚀 ЗАПУСК ПРОСТОГО ТОРГОВОГО БОТА")
    print("="*50)
    
    # Создаем бота
    bot = SimpleTradingBot()
    
    # Симулируем ценовые данные BTC (пример)
    print("\n📊 Тестирование на исторических данных BTC...")
    btc_prices = [45000, 45150, 44950, 44800, 45200, 45050, 44900, 45100, 45000, 44900, 
                  44750, 44600, 44850, 45000, 45150, 45050, 44900, 44750, 44800, 44950]
    
    for i, price in enumerate(btc_prices):
        print(f"\n⏰ Ценовая свеча {i+1}: ${price:,}")
        
        # Добавляем данные
        bot.add_price_data(price)
        
        # Генерируем сигнал
        signal, reason = bot.generate_signal()
        print(f"📡 Сигнал: {signal} - {reason}")
        
        # Исполняем сделку
        bot.execute_trade(signal, price)
        
        # Просыпаемся
        time.sleep(0.5)
    
    # Выводим финальную сводку
    bot.print_summary()
    
    print("\n🎉 ТЕСТ ЗАВЕРШЕН!")
    print("📝 Следующий шаг: подключение к Bybit API")


if __name__ == "__main__":
    main()
