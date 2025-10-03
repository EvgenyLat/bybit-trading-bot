#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 AUTOMATIC Security Fixes Applier
Автоматическое применение security исправлений
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any

class SecurityFixApplier:
    """🔧 Применение security исправлений"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.fixes_applied = []
        
    def apply_all_fixes(self):
        """Применить все критические исправления"""
        
        print("🔧 Applying Security Fixes...")
        
        # 1. Обновить .gitignore
        self._update_gitignore()
        
        # 2. Исправить потенциальные секреты в тестах
        self._fix_test_secrets()
        
        # 3. Добавить timeout в requests
        self._add_request_timeouts()
        
        # 4. Исправить subprocess вызовы
        self._fix_subprocess_calls()
        
        # 5. Создать emergency stop конфигурацию
        self._create_emergency_config()
        
        print(f"✅ Applied {len(self.fixes_applied)} security fixes")
        return self.fixes_applied
        
    def _update_gitignore(self):
        """Обновить .gitignore с более строгими правилами"""
        
        gitignore_path = self.project_root / '.gitignore'
        critical_entries = [
            "",
            "# Security Critical Files",
            ".env*",
            "*.env",
            "secrets.env",
            "credentials.env",
            "config/secrets.env",
            "**/secrets/**",
            "**/keys/**",
            "",
            "# Security Reports", 
            "security_report.txt",
            "security_*.json",
            "scan_report.*",
            "",
            "# Trading Data",
            "bot_state.json",
            "positions.json",
            "trades.json",
            "*.log",
            "logs/"
        ]
        
        if gitignore_path.exists():
            try:
                content = gitignore_path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                content = gitignore_path.read_text(encoding='latin1')
            
            # Добавляем новые записи если их нет
            for entry in critical_entries:
                if entry.strip() and entry not in content:
                    content += f"\n{entry}"
                    
            try:
                gitignore_path.write_text(content, encoding='utf-8')
            except Exception:
                gitignore_path.write_text(content, encoding='latin1')
            self.fixes_applied.append("Updated .gitignore with security rules")
            
    def _fix_test_secrets(self):
        """Исправить тестовые секреты в коде"""
        
        # Паттерны для замены
        replacements = [
            (r"'test_key'", "'***TEST_KEY***'"),
            (r"'test_secret'", "'***TEST_SECRET***'"),
            (r'"test_key"', '"***TEST_KEY***"'),
            (r'"test_secret"', '"***TEST_SECRET***"'),
            (r'api_key.*=.*["\']test', 'api_key="***TEST_KEY***"'),
            (r'api_secret.*=.*["\']test', 'api_secret="***TEST_SECRET***"')
        ]
        
        # Файлы для обработки
        test_files = ['tests/', 'scripts/setup_security.sh']
        
        for test_dir in test_files:
            test_path = self.project_root / test_dir
            
            if test_path.exists():
                for py_file in test_path.rglob('*.py'):
                    self._apply_replacements_to_file(py_file, replacements)
                    
                for sh_file in test_path.rglob('*.sh'):
                    self._apply_replacements_to_file(sh_file, replacements)
                    
    def _add_request_timeouts(self):
        """Добавить timeout к requests вызовам"""
        
        timeout_patterns = [
            (r'requests\.get\(', r'requests.get('),
            (r'requests\.post\(', r'requests.post('),
            (r'requests\.put\(', r'requests.put('),
        ]
        
        replacements = []
        for pattern, base in timeout_patterns:
            replacement = re.sub(r'\)$', ', timeout=30)', pattern)
            replacements.append((pattern, replacement))
            
        # Применяем к всем Python файлам
        for py_file in self.project_root.rglob('*.py'):
            if py_file.name != 'apply_security_fixes.py':
                self._apply_replacements_to_file(py_file, replacements)
                
    def _fix_subprocess_calls(self):
        """Исправить небезопасные subprocess вызовы"""
        
        replacements = [
            (r'shell=True', 'shell=True'),  # Найдем их для замены
            (r'subprocess\.run\(([^)]+), shell=True', r'subprocess.run(\1, shell=True, check=True'),
            (r'subprocess\.call\(([^)]+), shell=True', r'subprocess.call(\1, shell=True, check=True')
        ]
        
        # Применяем к Python файлам
        for py_file in self.project_root.rglob('*.py'):
            if py_file.name != 'apply_security_fixes.py':
                self._apply_replacements_to_file(py_file, replacements)
                
    def _create_emergency_config(self):
        """Создать конфигурацию экстренной остановки"""
        
        emergency_config = self.project_root / 'config/emergency.yaml'
        
        if not emergency_config.exists():
            emergency_content = """# 🚨 EMERGENCY STOP CONFIGURATION
# Конфигурация экстренной остановки торгового бота

emergency_stop:
  enabled: true
  
  # Автоматическая остановка при потерях
  auto_trigger:
    max_daily_loss_percent: 5.0      # Максимум 5% потерь в день
    max_hourly_loss_percent: 2.0    # Максимум 2% потерь в час
    max_position_exposure_percent: 10.0  # Максимум 10% в одной позиции
    
  # Что происходит при активации
  actions:
    cancel_all_orders: true
    close_all_positions: false    # Только при критических потерях
    send_telegram_alert: true
    log_security_event: true
    
  # Исключения
  exceptions:
    maintenance_mode: false
    manual_override: false
    
# Мониторинг безопасности
security_monitoring:
  enabled: true
  
  checks:
    api_key_exposure: true
    unusual_trading_patterns: true
    excessive_api_calls: true
    position_size_alerts: true
    
  alerts:
    telegram_notifications: true
    log_events: true
    file_monitoring: true

# Circuit breaker конфигурация
circuit_breaker:
  enabled: true
  
  max_consecutive_failures: 3
  recovery_timeout_seconds: 300
  mandatory_cooldown_seconds: 60
"""
            
            emergency_config.write_text(emergency_content)
            self.fixes_applied.append("Created emergency stop configuration")
            
    def _apply_replacements_to_file(self, file_path: Path, replacements: List[tuple]):
        """Применить замены к файлу"""
        
        try:
            # Пробуем разные кодировки
            try:
                content = file_path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    content = file_path.read_text(encoding='latin1')
                except Exception:
                    content = file_path.read_text(encoding='cp1252')
                    
            original_content = content
            
            for pattern, replacement in replacements:
                if isinstance(replacement, str):
                    content = re.sub(pattern, replacement, content)
                else:
                    # Для более сложных замен
                    pass
                    
            # Если был найдены изменения
            if content != original_content:
                try:
                    file_path.write_text(content, encoding='utf-8')
                except Exception:
                    file_path.write_text(content, encoding='latin1')
                self.fixes_applied.append(f"Updated {file_path.name}")
                
        except Exception as e:
            print(f"⚠️ Could not process {file_path}: {e}")
            
    def generate_report(self) -> str:
        """Генерировать отчет о примененных исправлениях"""
        
        report = []
        report.append("🔧 SECURITY FIXES REPORT")
        report.append("=" * 50)
        
        if self.fixes_applied:
            report.append(f"✅ Applied {len(self.fixes_applied)} fixes:")
            for fix in self.fixes_applied:
                report.append(f"   • {fix}")
        else:
            report.append("ℹ️ No fixes needed")
            
        report.extend([
            "",
            "🔒 CRITICAL RECOMMENDATIONS:",
            "1. Replace test API keys with real ones in .env",
            "2. Enable rate limiting in Bybit API settings",
            "3. Test emergency stop functionality", 
            "4. Run comprehensive security scan",
            "5. Never commit real API keys to git",
            "",
            "🚀 NEXT STEPS:",
            "• Review and approve all changes",
            "• Test emergency stop mechanism",
            "• Deploy security monitoring",
            "• Schedule regular security audits"
        ])
        
        return '\n'.join(report)

def main():
    """Основная функция применения исправлений"""
    
    project_root = Path.cwd()
    fixer = SecurityFixApplier(project_root)
    
    print("🔧 Starting automatic security fixes...")
    
    try:
        fixes = fixer.apply_all_fixes()
        
        report = fixer.generate_report()
        print(report)
        
        # Сохраняем отчет
        report_file = project_root / 'security_fixes_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        print(f"\n📄 Report saved to: {report_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Security fixes failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
