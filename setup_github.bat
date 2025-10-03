@echo off
echo == BYBIT TRADING BOT - GITHUB SETUP ==
echo.

REM Настройка имени пользователя Git
git config --global user.name "EvgenyLat"
git config --global user.email "evgeny@example.com"

REM Инициализация репозитория
echo Инициализация Git репозитория...
git init

echo Добавление файлов проекта...
git add .

echo Создание первого коммита...
git commit -m "🚀 Initial commit: Bybit Trading Bot

✅ Полнофункциональный торговый бот:
- Technical Analysis (RSI, SMA, MACD, Bollinger Bands)
- Risk Management и позиционный контроль  
- Подключение к Bybit REST/WebSocket API
- Telegram уведомления и мониторинг
- Docker контейнеризация
- Comprehensive backtesting с Vectorbt
- Microservices architecture
- Security шифрование API ключей
- Health monitoring и logging

📊 Готов к продакшен торговле на Bybit!"

REM Настройка удаленного репозитория
echo Настройка подключения к GitHub...
git remote add origin https://github.com/EvgenyLat/bybit-trading-bot.git

REM Переключение на main ветку
git branch -M main

echo Загрузка в GitHub...
git push -u origin main

echo.
echo ✅ ПРОЕКТ УСПЕШНО ЗАГРУЖЕН В GITHUB!
echo 🌐 URL: https://github.com/EvgenyLat/bybit-trading-bot
echo.
pause
