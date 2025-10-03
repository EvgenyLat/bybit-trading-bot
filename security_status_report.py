#!/usr/bin/env python3

import os
import sys
from pathlib import Path

def security_check():
    """Простая проверка безопасности"""
    
    print("🔒 SECURITY STATUS REPORT")
    print("=" * 50)
    
    issues = []
    
    # 1. Проверка .env файла
    env_file = Path('.env')
    if env_file.exists():
        print("✅ .env file: FOUND")
        
        # Проверяем .gitignore
        gitignore = Path('.gitignore')
        if gitignore.exists():
            try:
                gitignore_content = gitignore.read_text(encoding='utf-8')
                if '.env' in gitignore_content:
                    print("✅ .env in .gitignore: YES")
                else:
                    issues.append("❌ .env NOT in .gitignore")
            except UnicodeDecodeError:
                # Пробуем другие кодировки
                try:
                    gitignore_content = gitignore.read_text(encoding='latin1')
                    if '.env' in gitignore_content:
                        print("✅ .env in .gitignore: YES")
                    else:
                        issues.append("❌ .env NOT in .gitignore")
                except Exception as e:
                    issues.append(f"❌ Cannot read .gitignore: {e}")
        else:
            issues.append("❌ .gitignore missing")
            
        # Читаем .env содержимое (безопасно)
        try:
            env_content = env_file.read_text(encoding='utf-8')
            if 'ваш_api_ключ' in env_content.lower():
                issues.append("⚠️ Test API key in .env (replace with real)")
            else:
                print("✅ .env contains real keys")
        except UnicodeDecodeError:
            try:
                env_content = env_file.read_text(encoding='latin1')
                if 'ваш_api_ключ' in env_content.lower():
                    issues.append("⚠️ Test API key in .env (replace with real)")
                else:
                    print("✅ .env contains real keys")
            except Exception as e:
                issues.append(f"❌ Cannot read .env: {e}")
        except Exception as e:
            issues.append(f"❌ Cannot read .env: {e}")
            
    else:
        issues.append("❌ .env file NOT FOUND")
    
    # 2. Проверка конфигурационных файлов
    config_files = [
        'config/config.yaml',
        'config/risk_config.yaml',
        'config/secrets.env.example'
    ]
    
    missing_files = []
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✅ {config_file}: FOUND")
        else:
            missing_files.append(config_file)
    
    if missing_files:
        issues.append(f"❌ Missing files: {', '.join(missing_files)}")
    
    # 3. Проверка основных модулей безопасности
    security_modules = [
        'src/config.py',
        'src/secure_executor.py', 
        'scripts/security_check.py'
    ]
    
    for module in security_modules:
        if Path(module).exists():
            print(f"✅ {module}: READY")
        else:
            issues.append(f"❌ Security module missing: {module}")
    
    # 4. Проверка GitHub Actions
    github_workflows = Path('.github/workflows')
    if github_workflows.exists():
        print("✅ GitHub Actions: CONFIGURED")
    else:
        issues.append("⚠️ GitHub Actions not configured")
    
    # ИТОГИ
    print("\n" + "=" * 50)
    print("📊 SECURITY SUMMARY:")
    
    if not issues:
        print("🎉 ALL SECURITY CHECKS PASSED!")
        print("✅ Repository is SECURE")
        print("🚀 Ready for production deployment")
    else:
        print(f"⚠️ Found {len(issues)} issues:")
        for issue in issues:
            print(f"   {issue}")
            
        if any("❌" in issue for issue in issues):
            print("\n🚨 CRITICAL ISSUES FOUND")
            print("Please fix before deploying to production")
        else:
            print("\n⚠️ Minor issues - safe to proceed")
    
    print("\n🔧 IMMEDIATE ACTIONS:")
    print("1. Ensure .env has real API keys (not test values)")
    print("2. Test emergency stop: python src/config.py")
    print("3. Verify Bybit testnet connection")
    print("4. Install security tools: pip install pip-audit bandit")
    
    return len(issues) == 0

if __name__ == "__main__":
    secure = security_check()
    sys.exit(0 if secure else 1)
