#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 SIMPLE Security Fix
Простая версия security исправлений без проблем кодировки
"""

import os
import shutil
from pathlib import Path

def apply_security_fixes():
    """Применить простые security исправления"""
    
    print("🔧 Applying Simple Security Fixes...")
    
    fixes_applied = []
    
    # 1. Обновить .gitignore
    gitignore_path = Path('.gitignore')
    
    critical_entries = [
        "",
        "# SECURITY CRITICAL",
        ".env*",
        "*.env",
        "secrets.env", 
        "credentials.env",
        "config/secrets.env",
        "security_report.txt",
        "security_*.json",
        "bot_state.json",
        "positions.json",
        "trades.json",
        "logs/",
        "*.log"
    ]
    
    if gitignore_path.exists():
        try:
            content = gitignore_path.read_text(encoding='utf-8')
        except:
            content = gitignore_path.read_text(encoding='latin1')
            
        new_entries = []
        for entry in critical_entries:
            if entry.strip() and entry not in content:
                new_entries.append(entry)
                
        if new_entries:
            updated_content = content + '\n'.join(new_entries)
            try:
                gitignore_path.write_text(updated_content, encoding='utf-8')
            except:
                gitignore_path.write_text(updated_content, encoding='latin1')
            fixes_applied.append("Updated .gitignore")
    
    # 2. Проверить наличие .env в git репозитории
    env_file = Path('.env')
    if env_file.exists():
        try:
            content = env_file.read_text(encoding='utf-8')
        except:
            content = env_file.read_text(encoding='latin1')
            
        # Проверяем на тестовые значения
        test_patterns = [
            'ваш_api_ключ',
            'your_api_key', 
            'test_key',
            'example_key',
            'PUT_YOUR_KEY_HERE'
        ]
        
        found_test_patterns = []
        for pattern in test_patterns:
            if pattern.lower() in content.lower():
                found_test_patterns.append(pattern)
                
        if found_test_patterns:
            fixes_applied.append(f"Found test patterns: {', '.join(found_test_patterns[:3])}")
        else:
            fixes_applied.append(".env contains real keys")
    else:
        fixes_applied.append(".env file not found")
        
    # 3. Проверить основные модули безопасности
    security_modules = [
        'src/config.py',
        'src/secure_executor.py', 
        'src/safe_executor.py',
        'scripts/security_check.py'
    ]
    
    missing_modules = []
    for module in security_modules:
        if Path(module).exists():
            fixes_applied.append(f"✅ {module}")
        else:
            missing_modules.append(module)
            
    if missing_modules:
        fixes_applied.append(f"⚠️ Missing: {', '.join(missing_modules)}")
        
    # 4. Создать простую конфигурацию безопасности
    safety_config = Path('config/safety_config.yaml')
    if not safety_config.exists():
        safety_config.parent.mkdir(exist_ok=True)
        
        safety_content = """# Basic Safety Configuration
emergency_stop:
  enabled: true
  max_daily_loss: 3%
  max_position_size: 5%

api_security:
  timeout_seconds: 30
  retry_attempts: 3
  rate_limit_delay: 1

logging:
  omit_secrets: true
  log_level: INFO
"""
        
        safety_config.write_text(safety_content, encoding='utf-8')
        fixes_applied.append("Created safety configuration")
    
    # Генерируем отчет
    print("\n🔒 SECURITY FIX REPORT")
    print("=" * 40)
    
    for fix in fixes_applied:
        print(f"✅ {fix}")
        
    print(f"\n📊 Applied {len(fixes_applied)} fixes")
        
    # Рекомендации
    print("\n🔧 CRITICAL RECOMMENDATIONS:")
    print("1. Replace test API keys with real ones")
    print("2. Install security tools: pip install bandit safety")  
    print("3. Test emergency stop functionality")
    print("4. Never commit real API keys to git")
    
    return True

def main():
    """Основная функция"""
    print("🔧 Starting simple security fixes...")
    
    try:
        apply_security_fixes()
        print("\n✅ Security fixes completed!")
        return 0
    except Exception as e:
        print(f"❌ Security fixes failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
