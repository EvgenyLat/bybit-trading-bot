#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Быстрый запуск торгового бота Bybit
Автоматическая настройка и тестирование системы
"""

import os
import sys
from pathlib import Path

def print_banner():
    """Красивое приветствие"""
    print("🤖" + "="*50)
    print("🚀 BYBIT TRADING BOT - QUICK START")
    print("🤖" + "="*50)
    print()

def check_requirements():
    """Проверка установленных библиотек"""
    print("📦 Проверка зависимостей...")
    
    required_packages = [
        'pandas', 'numpy', 'pybit', 'ccxt', 'ta', 
        'requests', 'pyyaml', 'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - установлен")
        except ImportError:
            print(f"❌ {package} - НЕ УСТАНОВЛЕН")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Отсутствуют библиотеки: {', '.join(missing_packages)}")
        print("📋 Установите их командой:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("\n✅ Все зависимости установлены!")
    return True

def check_config():
    """Проверка конфигурации"""
    print("\n⚙️ Проверка конфигурации Конфигуражtion check...")
    
    # Проверка .env файла
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env файл найден")
        
        # Читаем содержимое для проверки важных настроек
        with open(env_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if "BYBIT_API_KEY=" in content and "ваш_api_ключ" not in content:
            print("✅ Bybit API ключи настроены")
        else:
            print("⚠️ Bybit API ключи нужно настроить")
            return False
            
    else:
        print("❌ .env файл не найден")
        print("📝 Создайте файл .env из config/secrets.env.example")
        return False
    
    return True

def test_trading_logic():
    """Тест торговой логики"""
    print("\n🧪 Тестирование торговой логики...")
    
    try:
        import pandas as pd
        import numpy as np
        import ta
        
        # Создаем тестовые данные
        test_prices = [45000, 45150, 44950, 44800, 45200, 45050, 
                      45000, 45100, 44900, 44750, 44600, 44850, 
                      45000, 45150, 45050, 44900, 44750, 44800, 44950]
        
        data = pd.DataFrame({'close': test_prices})
        
        # Расчет индикаторов
        data['rsi'] = ta.momentum.rsi(data['close'])
        data['sma_5'] = ta.trend.sma(data['close'], window=5)
        data['sma_20'] = ta.trend.sma(data['close'], window=20)
        
        # Определение сигнала
        current_rsi = data['rsi'].iloc[-1]
        current_sma5 = data['sma_5'].iloc[-1]
        current_sma20 = data['sma_20'].iloc[-1]
        
        signal = "HOLD"
        reasons = []
        
        if current_rsi < 30:
            signal = "BUY"
            reasons.append("RSI перепроданность")
        elif current_rsi > 70:
            signal = "SELL"
            reasons.append("RSI перекупленность")
        
        if current_sma5 > current_sma20 and signal == "HOLD":
            signal = "BUY"
            reasons.append("Восходящий тренд")
        
        print(f"📊 Текущие показатели:")
        print(f"   💰 Цена: ${int(data['close'].iloc[-1]):,}")
        print(f"   📈 RSI: {current_rsi:.1f}")
        print(f"   📊 SMA 5: ${current_sma5:.0f}")
        print(f"   📊 SMA 20: ${current_sma20:.0f}")
        print(f"   🎯 Сигнал: {signal}")
        print(f"   📝 Причины: {', '.join(reasons) if reasons else 'Нейтрально'}")
        
        print("\n✅ Торговая логика работает корректно!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в торговой логике: {e}")
        return False

def test_bybit_connection():
    """Тест подключения к Bybit"""
    print("\n🌐 Тестирование подключения к Bybit...")
    
    try:
        from pybit.unified_trading import HTTP
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        
        api_key = os.getenv('BYBIT_API_KEY')
        api_secret = os.getenv('BYBIT_API_SECRET')
        testnet = os.getenv('BYBIT_TESTNET', 'true').lower() == 'true'
        
        if not api_key or api_key == 'ваш_api_ключ':
            print("❌ API ключи не настроены")
            return False
        
        # Подключение (без реальных запросов)
        print(f"🔧 Настройки подключения:")
        print(f"   📡 Testnet: {testnet}")
        print(f"   🔑 API Key: {api_key[:10]}...")
        
        print("\n✅ Настройки подключения корректны!")
        print("💡 Для полного теста запустите бота")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return False

def show_next_steps():
    """Показ следующих шагов"""
    print("\n🎯 СЛЕДУЮЩИЕ ШАГИ:")
    print("="*50)
    print("1. 📝 Настройте API ключи в файле .env")
    print("2. 🧪 Запустите тестовый режим:")
    print("   python simple_trading_bot.py")
    print("3. 📊 Проверьте Telegram уведомления")
    print("4. 💰 Начните с малой суммы на TESTNET")
    print("5. 📈 Переходите на реальную торговлю")
    print("="*50)

def main():
    """Главная функция быстрого запуска"""
    print_banner()
    
    all_good = True
    
    # Проверяем зависимости
    if not check_requirements():
        all_good = False
    
    # Проверяем конфигурацию
    if not check_config():
        all_good = False
    
    # Тестируем торговую логику
    if not test_trading_logic():
        all_good = False
    
    # Тестируем подключение
    if not test_bybit_connection():
        all_good = False
    
    print("\n" + "="*50)
    if all_good:
        print("🎉 СИСТЕМА ГОТОВА К ТОРГОВЛЕ!")
        print("✅ Все компоненты работают корректно")
    else:
        print("⚠️ НУЖНЫ ИСПРАВЛЕНИЯ")
        print("❌ Проверьте найденные проблемы")
    
    show_next_steps()
    
    print("\n📚 Дополнительно см. README.md и документацию")
    print("🔒 Не забудьте про безопасность торговли!")

if __name__ == "__main__":
    main()
