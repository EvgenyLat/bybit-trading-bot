#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔒 SIMPLE Security Check Script 
Простая проверка безопасности без внешних команд
"""

import os
import re
from pathlib import Path
from typing import List

def check_env_file() -> List[str]:
    """Проверка файла .env"""
    print("🔍 Checking .env file...")
    
    issues = []
    env_file = Path('.env')
    gitignore_file = Path('.gitignore')
    
    # Проверяем существование .env
    if not env_file.exists():
        issues.append("❌ .env file not found")
        return issues
    
    # Проверяем .gitignore
    if not gitignore_file.exists():
        issues.append("❌ .gitignore file missing")
    else:
        gitignore_content = gitignore_file.read_text()
        if '.env' not in gitignore_content:
            issues.append("❌ .env not in .gitignore")
    
    # Читаем .env и проверяем
    try:
        env_content = env_file.read_text()
        
        # Тестовые значения
        test_patterns = [
            'ваш_api_ключ',
            'test_key', 
            'example_key',
            'your_key_here'
        ]
        
        for pattern in test_patterns:
            if pattern in env_content.lower():
                issues.append(f"⚠️ Test pattern found in .env: {pattern}")
        
        # Обязательные переменные
        required_vars = ['BYBIT_API_KEY', 'BYBIT_API_SECRET']
        for var in required_vars:
            if f"{var}=" not in env_content:
                issues.append(f"❌ Missing required variable: {var}")
                
        if not issues:
            print("✅ .env file configuration OK")
            
    except Exception as e:
        issues.append(f"❌ Error reading .env: {e}")
        
    return issues

def check_secrets_in_files() -> List[str]:
    """Проверка секретов в файлах проекта"""
    print("🔍 Checking for secrets in project files...")
    
    issues = []
    patterns = [
        r'api_key\s*=\s*["\'][^"\']+["\']',
        r'api_secret\s*=\s*["\'][^"\']+["\']',
        r'ghp_[a-zA-Z0-9]{36}',  # GitHub tokens
    ]
    
    excluded_files = ['.env', '.git/', '__pycache__/', '.vscode/', '.idea/']
    
    for pattern in patterns:
        for py_file in Path('.').rglob('*.py'):
            # Пропускаем исключенные файлы
            if any(exc in str(py_file) for exc in excluded_files):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                matches = re.findall(pattern, content, re.IGNORECASE)
                
                for match in matches:
                    # Исключаем примеры
                    if any(word in match.lower() for word in ['example', 'test', 'demo']):
                        continue
                        
                    issues.append(f"⚠️ Potential secret in {py_file}: {match[:20]}...")
                    
            except Exception:
                pass
                
    if not issues:
        print("✅ No hardcoded secrets found in code")
        
    return issues

def check_config_files() -> List[str]:
    """Проверка конфигурационных файлов"""
    print("🔍 Checking configuration files...")
    
    issues = []
    config_files = [
        'config/config.yaml',
        'config/risk_config.yaml', 
        'config/secrets.env.example'
    ]
    
    for config_file in config_files:
        config_path = Path(config_file)
        if not config_path.exists():
            issues.append(f"❌ Missing config file: {config_file}")
    
    if not issues:
        print("✅ All required config files present")
        
    return issues

def check_dangerous_patterns() -> List[str]:
    """Проверка опасных паттернов в коде"""
    print("🔍 Checking for dangerous code patterns...")
    
    issues = []
    dangerous_patterns = [
        (r'eval\s*\(', 'eval() function'),
        (r'exec\s*\(', 'exec() function'),
        (r'__import__\s*\(', '__import__ function'),
        (r'pickle\.loads', 'pickle deserialization'),
        (r'shell=True', 'shell command execution')
    ]
    
    for pattern, description in dangerous_patterns:
        for py_file in Path('src').rglob('*.py'):
            try:
                content = py_file.read_text(encoding='utf-8')
                if re.search(pattern, content):
                    issues.append(f"⚠️ {description} in {py_file.name}")
            except Exception:
                pass
                
    if not issues:
        print("✅ No dangerous code patterns found")
        
    return issues

def generate_security_report() -> str:
    """Генерация отчета безопасности"""
    print("\n🔒 SIMPLE SECURITY REPORT")
    print("=" * 50)
    
    all_checks = [
        ("Environment File", check_env_file()),
        ("Secret Detection", check_secrets_in_files()),
        ("Configuration Files", check_config_files()),
        ("Code Safety", check_dangerous_patterns())
    ]
    
    critical_count = 0
    warning_count = 0
    report = []
    
    for check_name, issues in all_checks:
        if issues:
            report.append(f"\n📋 {check_name}:")
            for issue in issues:
                if "❌" in issue:
                    critical_count += 1
                else:
                    warning_count += 1
                report.append(f"   {issue}")
        else:
            report.append(f"\n✅ {check_name}: PASSED")
    
    # Итоги
    report.append(f"\n{'='*50}")
    report.append(f"📊 SUMMARY:")
    report.append(f"   🔴 Critical Issues: {critical_count}")
    report.append(f"   ⚠️ Warnings: {warning_count}")
    
    if critical_count == 0:
        report.append(f"   ✅ Overall Status: SECURE")
    else:
        report.append(f"   🚨 Overall Status: REQUIRES ATTENTION")
    
    # Рекомендации
    report.append(f"\n🔧 RECOMMENDATIONS:")
    if critical_count > 0:
        report.append(f"   1. Fix critical issues before deployment")
        report.append(f"   2. Secure your .env file")
    
    report.append(f"   3. Install security tools: pip install bandit safety")
    report.append(f"   4. Use GitHub Advanced Security")
    report.append(f"   5. Regular security audits")
    
    return '\n'.join(report)

def main():
    """Основная функция"""
    print("🔒 Starting simple security check...")
    
    try:
        report = generate_security_report()
        print(report)
        
        # Сохраняем отчет
        with open('security_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 Report saved to: security_report.txt")
        
        return 0
        
    except Exception as e:
        print(f"❌ Security check failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
