#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔒 COMPREHENSIVE Security Check Script
Комплексная проверка безопасности торгового бота
"""

import os
import sys
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

class SecurityChecker:
    """🔒 Комплексная проверка безопасности"""
    
    def __init__(self):
        self.base_dir = Path.cwd()
        self.issues = []
        self.critical_issues = []
        
    def check_git_secrets(self) -> List[str]:
        """Проверка секретов в git истории"""
        print("🔍 Checking Git history for secrets...")
        
        secrets_patterns = [
            r'API_KEY\s*=\s*["\']([^"\']{,50})["\']',
            r'API_SECRET\s*=\s*["\']([^"\']{,50})["\']',
            r'BYBIT_API_KEY\s*=\s*["\']([^"\']{,50})["\']',
            r'BYBIT_API_SECRET\s*=\s*["\']([^"\']{,50})["\']',
            r'bot_token\s*=\s*["\']([^"\']{,50})["\']',
            r'private_key\s*=\s*["\']([^"\']{,50})["\']',
            r'ghp_[a-zA-Z0-9]{36}',  # GitHub personal access tokens
            r'gho_[a-zA-Z0-9]{36}',  # GitHub OAuth tokens
        ]
        
        issues = []
        
        try:
            # Проверяем текущие файлы
            for pattern in secrets_patterns:
                result = subprocess.run(
                    ['git', 'grep', '-n', '-i', pattern],
                    capture_output=True,
                    text=True,
                    cwd=self.base_dir
                )
                
                if result.stdout:
                    matches = result.stdout.strip().split('\n')
                    for match in matches:
                        # Исключаем example файлы
                        if 'example' not in match.lower() and 'test' not in match.lower():
                            issues.append(f"⚠️ Potential secret in: {match}")
                            
            # Проверяем индекс git (staged files)
            result = subprocess.run(
                ['git', 'diff', '--cached'],
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                for pattern in secrets_patterns:
                    matches = re.findall(pattern, result.stdout)
                    if matches:
                        issues.append(f"🚨 CRITICAL: Secret in staged files: {pattern}")
                            
        except Exception as e:
            issues.append(f"❌ Error checking secrets: {e}")
            
        return issues
        
    def check_env_file(self) -> List[str]:
        """Проверка файла .env"""
        print("🔍 Checking .env file...")
        
        issues = []
        env_file = self.base_dir / '.env'
        
        if not env_file.exists():
            issues.append("❌ .env file not found")
            return issues
            
        # Проверяем что .env в .gitignore
        gitignore_file = self.base_dir / '.gitignore'
        if gitignore_file.exists():
            gitignore_content = gitignore_file.read_text()
            if '.env' not in gitignore_content:
                issues.append("❌ .env not in .gitignore")
        else:
            issues.append("❌ .gitignore file missing")
            
        # Читаем .env и проверяем наличие тестовых значений
        try:
            env_content = env_file.read_text()
            
            # Проверяем на тестовые значения
            test_patterns = [
                'ваш_api_ключ',
                'test_key',
                'example_key',
                'your_key_here'
            ]
            
            for pattern in test_patterns:
                if pattern in env_content.lower():
                    issues.append(f"⚠️ Test pattern found in .env: {pattern}")
                    
            # Проверяем минимальные требования
            required_vars = ['BYBIT_API_KEY', 'BYBIT_API_SECRET']
            for var in required_vars:
                if f"{var}=" not in env_content:
                    issues.append(f"❌ Missing required variable: {var}")
                    
        except Exception as e:
            issues.append(f"❌ Error reading .env: {e}")
            
        return issues
        
    def check_dependencies(self) -> List[str]:
        """Проверка безопасности зависимостей"""
        print("🔍 Checking dependencies security...")
        
        issues = []
        
        # Проверяем pip-audit если установлен
        try:
            result = subprocess.run(
                ['pip-audit', '--format=json'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("✅ pip-audit passed")
            else:
                issues.append("⚠️ pip-audit found vulnerabilities")
                if result.stdout:
                    issues.append(f"Details: {result.stdout[:200]}...")
                    
        except subprocess.TimeoutExpired:
            issues.append("⚠️ pip-audit timeout")
        except FileNotFoundError:
            issues.append("⚠️ pip-audit not installed (pip install pip-audit)")
            
        # Проверяем safety если установлен  
        try:
            result = subprocess.run(
                ['safety', 'check', '--json'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("✅ safety passed")
            else:
                issues.append("⚠️ safety found vulnerabilities")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Safety не критично
            pass
            
        return issues
        
    def check_code_security(self) -> List[str]:
        """Проверка безопасности кода"""
        print("🔍 Checking code security...")
        
        issues = []
        
        # Проверяем bandit если установлен
        try:
            result = subprocess.run(
                ['bandit', '-r', 'src/', '-f', 'json'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if 'No issues identified' in result.stdout:
                print("✅ bandit passed - no issues")
            else:
                issues.append("⚠️ bandit found potential security issues")
                if result.stdout:
                    issues.append(f"Details: bandit analysis completed")
                    
        except subprocess.TimeoutExpired:
            issues.append("⚠️ bandit timeout")
        except FileNotFoundError:
            issues.append("⚠️ bandit not installed (pip install bandit)")
            
        # Проверяем критические паттерны в коде
        dangerous_patterns = [
            (r'eval\s*\(', 'Use of eval() function'),
            (r'exec\s*\(', 'Use of exec() function'),
            (r'__import__\s*\(', 'Use of __import__'),
            (r'pickle\.loads', 'Deserialization with pickle'),
            (r'shell=True', 'subprocess with shell=True'),
        ]
        
        src_dir = self.base_dir / 'src'
        if src_dir.exists():
            for py_file in src_dir.rglob('*.py'):
                try:
                    content = py_file.read_text()
                    for pattern, description in dangerous_patterns:
                        if re.search(pattern, content):
                            issues.append(f"⚠️ {description} in {py_file.name}")
                except Exception:
                    pass
                    
        return issues
        
    def check_configuration(self) -> List[str]:
        """Проверка конфигурации безопасности"""
        print("🔍 Checking configuration security...")
        
        issues = []
        
        # Проверяем основные конфиги
        config_files = [
            'config/config.yaml',
            'config/risk_config.yaml',
            'config/secrets.env.example'
        ]
        
        for config_file in config_files:
            config_path = self.base_dir / config_file
            if not config_path.exists():
                issues.append(f"❌ Missing config file: {config_file}")
                
        # Проверяем настройки безопасности в config.yaml
        config_path = self.base_dir / 'config/config.yaml'
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Проверяем критические настройки
                if 'emergency_stop_enabled: true' not in content:
                    issues.append("⚠️ Emergency stop might not be enabled")
                    
                if 'max_daily_loss:' in content:
                    if 'max_daily_loss: 1.0' in content:  # Слишком высокий лимит
                        issues.append("⚠️ Daily loss limit might be too high")
                        
            except Exception as e:
                issues.append(f"❌ Error reading config.yaml: {e}")
                
        return issues
        
    def check_file_permissions(self) -> List[str]:
        """Проверка прав доступа к файлам"""
        print("🔍 Checking file permissions...")
        
        issues = []
        
        sensitive_files = [
            '.env',
            'config/secrets.env',
            'logs/',
            '*.log'
        ]
        
        for pattern in sensitive_files:
            matches = list(self.base_dir.rglob(pattern))
            if matches:
                for file_path in matches:
                    if file_path.is_file():
                        # На Windows проверяем через attrib
                        try:
                            result = subprocess.run(
                                ['attrib', str(file_path)],
                                capture_output=True,
                                text=True
                            )
                            
                            if 'R' not in result.stdout:  # R = read-only
                                issues.append(f"⚠️ {file_path} might be world-readable")
                                
                        except Exception:
                            # В Linux можно использовать stat
                            pass
                            
        return issues
        
    def generate_report(self) -> str:
        """Генерация отчета о безопасности"""
        print("\n" + "="*60)
        print("🔒 COMPREHENSIVE SECURITY REPORT")
        print("="*60)
        
        # Запускаем все проверки
        all_checks = [
            ("Git History", self.check_git_secrets()),
            (".env Security", self.check_env_file()),
            ("Dependencies", self.check_dependencies()),
            ("Code Security", self.check_code_security()),
            ("Configuration", self.check_configuration()),
            ("File Permissions", self.check_file_permissions())
        ]
        
        # Категоризируем проблемы
        critical_count = 0
        warning_count = 0
        
        report = []
        
        for check_name, issues in all_checks:
            if issues:
                report.append(f"\n📋 {check_name}:")
                for issue in issues:
                    if "🚨 CRITICAL" in issue or "❌" in issue:
                        critical_count += 1
                        report.append(f"   {issue}")
                    else:
                        warning_count += 1
                        report.append(f"   {issue}")
            else:
                report.append(f"\n✅ {check_name}: PASSED")
                
        # Итоговая статистика
        report.append(f"\n{'='*60}")
        report.append(f"📊 SUMMARY:")
        report.append(f"   🔴 Critical Issues: {critical_count}")
        report.append(f"   ⚠️ Warnings: {warning_count}")
        
        if critical_count == 0:
            report.append(f"   ✅ Overall Status: SECURE")
        elif critical_count <= 2:
            report.append(f"   ⚠️ Overall Status: NEEDS ATTENTION")
        else:
            report.append(f"   🚨 Overall Status: CRITICAL ISSUES")
            
        # Рекомендации
        report.append(f"\n🔧 RECOMMENDATIONS:")
        
        if critical_count > 0:
            report.append(f"   1. Fix all critical issues before production")
            report.append(f"   2. Remove any secrets from git history")
            report.append(f"   3. Secure your .env file")
            
        report.append(f"   4. Install security tools: pip install pip-audit bandit safety")
        report.append(f"   5. Enable GitHub Advanced Security")
        report.append(f"   6. Set up automated security scanning in CI/CD")
        
        return '\n'.join(report)


def main():
    """Основная функция проверки безопасности"""
    print("🔒 Starting comprehensive security check...")
    
    try:
        checker = SecurityChecker()
        report = checker.generate_report()
        
        print(report)
        
        # Сохраняем отчет
        report_file = Path('security_check_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        print(f"\n📄 Report saved to: {report_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Security check failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
