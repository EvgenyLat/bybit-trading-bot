#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔒 SECURE Configuration Loader
Безопасная загрузка конфигурации с проверкой секретов
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import yaml
import logging

class SecureConfig:
    """🔒 Безопасная загрузка конфигурации без экспозиции секретов"""
    
    def __init__(self, env_file: str = '.env', config_file: str = 'config/config.yaml'):
        """
        Инициализация конфигурации с проверкой безопасности
        
        Args:
            env_file: Путь к .env файлу
            config_file: Путь к основному конфигу
        """
        self.env_path = Path(env_file)
        self.config_path = Path(config_file)
        
        # Проверяем существование файлов
        self._validate_files()
        
        # Загружаем переменные окружения
        self._load_env()
        
        # Проверяем секреты
        self._validate_secrets()
        
        # Загружаем основной конфиг
        self._load_config()
        
    def _validate_files(self):
        """Проверка существования критических файлов"""
        if not self.env_path.exists():
            raise RuntimeError(f"❌ {self.env_path} не найден - abort!")

        if not self.config_path.exists():
            raise RuntimeError(f"❌ {self.config_path} не найден - abort!")
            
        print("✅ Конфигурационные файлы найдены")
        
    def _load_env(self):
        """Загрузка переменных окружения"""
        # Предупреждаем о коммите .env
        self._check_env_security()
        
        load_dotenv(dotenv_path=self.env_path)
        print("✅ Переменные окружения загружены")
        
    def _check_env_security(self):
        """Проверка безопасности .env файла"""
        # Проверяем что .env не в git (должен быть в .gitignore)
        env_content = self.env_path.read_text()
        
        # Проверяем примеры ключей (не должны быть настоящими)
        test_patterns = [
            'ваш_api_ключ',
            'test_key',
            'example_key'
        ]
        
        for pattern in test_patterns:
            if pattern in env_content.lower():
                raise RuntimeError(
                    f"⚠️ Найдены тестовые значения в .env: {pattern}\n"
                    f"Замените на настоящие API ключи перед запуском!"
                )
                
        print("🔒 .env файл прошел проверку безопасности")
        
    def _validate_secrets(self):
        """Проверка наличия обязательных секретов"""
        required_secrets = [
            'BYBIT_API_KEY',
            'BYBIT_API_SECRET'
        ]
        
        missing_secrets = []
        
        for secret in required_secrets:
            value = os.getenv(secret)
            if not value or value.strip() == '':
                missing_secrets.append(secret)
                
        if missing_secrets:
            raise RuntimeError(
                f"❌ Отсутствуют обязательные переменные:\n"
                f"{chr(10).join(f'  - {s}' for s in missing_secrets)}\n"
                f"Добавьте их в файл .env!"
            )
            
        print("✅ Все секреты найдены")
        
    def _load_config(self):
        """Загрузка основного конфига"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        print("✅ Основной конфиг загружен")
        
    def get_env_var(self, key: str, default: Optional[str] = None) -> str:
        """Безопасное получение переменной окружения"""
        value = os.getenv(key, default)
        if not value:
            raise RuntimeError(f"❌ Переменная {key} не установлена")
        return value
        
    def get_config(self, path: str, default: Any = None) -> Any:
        """Получение значения по пути (например 'trading.symbol')"""
        keys = path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
        
    @property
    def bybit_api_key(self) -> str:
        """🔒 Безопасно получить Bybit API ключ"""
        return self.get_env_var('BYBIT_API_KEY')
        
    @property 
    def bybit_api_secret(self) -> str:
        """🔒 Безопасно получить Bybit API секрет"""
        return self.get_env_var('BYBIT_API_SECRET')
        
    @property
    def bybit_testnet(self) -> bool:
        """Проверка использования тест-сети"""
        testnet = self.get_env_var('BYBIT_TESTNET', 'true').lower()
        return testnet == 'true'
        
    @property
    def telegram_enabled(self) -> bool:
        """Проверка включенности Telegram"""
        token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        return bool(token) and bool(chat_id) and token != 'ваш_токен_от_botfather'
        
    def log_security_warnings(self):
        """Логирование предупреждений безопасности"""
        warnings = []
        
        # Проверка testnet
        if not self.bybit_testnet:
            warnings.append("🔴 ТОРГОВЛЯ В РЕАЛЬНОЙ СЕТИ!")
            
        # Проверка Telegram
        if not self.telegram_enabled:
            warnings.append("⚠️ Telegram уведомления отключены")
            
        # Проверка безопасного пароля
        master_pass = os.getenv('MASTER_PASSWORD')
        if not master_pass or len(master_pass) < 12:
            warnings.append("⚠️ Слишком простой MASTER_PASSWORD")
            
        if warnings:
            print("\n🚨 СИСТЕМНЫЕ ПРЕДУПРЕЖДЕНИЯ:")
            for warning in warnings:
                print(f"   {warning}")
            print()
        else:
            print("✅ Настройки безопасности корректны")


# Singleton instance
_config: Optional[SecureConfig] = None

def get_config() -> SecureConfig:
    """Получить глобальный экземпляр конфигурации"""
    global _config
    if _config is None:
        _config = SecureConfig()
    return _config


if __name__ == "__main__":
    """Тестирование конфигурации"""
    try:
        config = SecureConfig()
        config.log_security_warnings()
        
        print("\n📊 Текущая конфигурация:")
        print(f"💰 Символ торговли: {config.get_config('trading.symbol', 'N/A')}")
        print(f"🧪 Testnet включен: {config.bybit_testnet}")
        print(f"📱 Telegram включен: {config.telegram_enabled}")
        
        print("\n✅ Конфигурация загружена успешно!")
        
    except Exception as e:
        print(f"❌ Ошибка конфигурации: {e}")
        sys.exit(1)
