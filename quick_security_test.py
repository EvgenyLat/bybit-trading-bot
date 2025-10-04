#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔒 Quick Security Test for CI/CD
Быстрый тест безопасности для GitHub Actions
"""

import os
import sys
from pathlib import Path

def test_security_basics():
    """Базовые тесты безопасности"""
    print("🔒 Running Quick Security Tests")
    print("=" * 40)
    
    issues = []
    
    # 1. Проверка .gitignore
    gitignore = Path('.gitignore')
    if gitignore.exists():
        try:
            content = gitignore.read_text(encoding='utf-8')
            if '.env' in content:
                print("✅ .env in .gitignore")
            else:
                issues.append("❌ .env NOT in .gitignore")
        except Exception as e:
            issues.append(f"❌ Cannot read .gitignore: {e}")
    else:
        issues.append("❌ .gitignore missing")
    
    # 2. Проверка .env файла
    env_file = Path('.env')
    if env_file.exists():
        try:
            content = env_file.read_text(encoding='utf-8')
            if 'ваш_api_ключ' in content.lower():
                issues.append("⚠️ Test API key in .env (replace with real)")
            else:
                print("✅ .env contains real keys")
        except Exception as e:
            issues.append(f"❌ Cannot read .env: {e}")
    else:
        print("ℹ️ .env file not found (expected for CI)")
    
    # 3. Проверка security модулей
    security_modules = [
        'scripts/security_check.py',
        'internal_security_scan.py',
        'src/config.py',
        'src/secure_executor.py'
    ]
    
    for module in security_modules:
        if Path(module).exists():
            print(f"✅ {module} exists")
        else:
            issues.append(f"❌ {module} missing")
    
    # 4. Проверка GitHub Actions
    workflow = Path('.github/workflows/security-and-tests.yml')
    if workflow.exists():
        print("✅ GitHub Actions workflow configured")
    else:
        issues.append("❌ GitHub Actions workflow missing")
    
    # 5. Проверка requirements файлов
    req_files = ['requirements-production.txt', 'requirements-secure.txt']
    for req_file in req_files:
        if Path(req_file).exists():
            print(f"✅ {req_file} exists")
        else:
            issues.append(f"❌ {req_file} missing")
    
    # ИТОГИ
    print("\n" + "=" * 40)
    print("📊 SECURITY TEST RESULTS:")
    
    if not issues:
        print("🎉 ALL SECURITY TESTS PASSED!")
        print("✅ Repository is SECURE")
        return True
    else:
        print(f"⚠️ Found {len(issues)} issues:")
        for issue in issues:
            print(f"   {issue}")
        return False

def test_imports():
    """Тест импорта основных библиотек"""
    print("\n📦 Testing Core Library Imports:")
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import yaml
        print("✅ PyYAML")
    except ImportError as e:
        print(f"❌ PyYAML import failed: {e}")
        return False
    
    return True

def main():
    """Главная функция"""
    print("🚀 Quick Security Test for CI/CD Pipeline")
    print("=" * 50)
    
    # Запускаем тесты
    security_ok = test_security_basics()
    imports_ok = test_imports()
    
    print("\n" + "=" * 50)
    print("🏆 FINAL RESULTS:")
    
    if security_ok and imports_ok:
        print("✅ ALL TESTS PASSED - CI/CD READY!")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())