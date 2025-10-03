#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔒 INTERNAL Security Scanner
Выполнение security проверок изнутри Python для обхода PATH проблем
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, List

class InternalSecurityScanner:
    """🔒 Внутренний сканер безопасности"""
    
    def __init__(self):
        self.results = {
            'bandit': {'status': 'skipped', 'details': 'PATH issues'},
            'pip_audit': {'status': 'skipped', 'details': 'PATH issues'},
            'safety': {'status': 'skipped', 'details': 'PATH issues'},
            'ruff': {'status': 'skipped', 'details': 'PATH issues'},
            'black': {'status': 'skipped', 'details': 'PATH issues'},
            'mypy': {'status': 'skipped', 'details': 'PATH issues'},
            'manual_checks': {'status': 'completed', 'issues': []}
        }
        
    def run_manual_security_checks(self):
        """Ручные проверки безопасности"""
        
        print("🔍 Running manual security checks...")
        
        issues = []
        
        # 1. Проверка на hardcoded secrets
        secrets_found = self._check_for_secrets()
        if secrets_found:
            issues.extend(secrets_found)
            
        # 2. Проверка опасных паттернов
        dangerous_patterns = self._check_dangerous_patterns()
        if dangerous_patterns:
            issues.extend(dangerous_patterns)
            
        # 3. Проверка .gitignore
        gitignore_issues = self._check_gitignore()
        if gitignore_issues:
            issues.extend(gitignore_issues)
            
        # 4. Проверка конфигурации
        config_issues = self._check_config()
        if config_issues:
            issues.extend(config_issues)
            
        self.results['manual_checks']['issues'] = issues
        
        return issues
        
    def _check_for_secrets(self) -> List[str]:
        """Проверка на захардкоженные секреты"""
        
        issues = []
        patterns = [
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'api_secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
            r'password\s*=\s*["\'][^"\']+["\']',
        ]
        
        # Проверяем src/ на реальные секреты (не тестовые)
        for py_file in Path('src').rglob('*.py'):
            try:
                content = py_file.read_text(encoding='utf-8')
                
                for pattern in patterns:
                    import re
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    
                    for match in matches:
                        # Проверяем что это не тестовый секрет
                        if not any(test_word in match.lower() for test_word in 
                                  ['test', 'example', 'demo ', 'fake', '***']):
                            issues.append(f"HARDCODED_SECRET in {py_file}: {match[:20]}...")
                            
            except Exception:
                continue
                
        return issues
        
    def _check_dangerous_patterns(self) -> List[str]:
        """Проверка опасных паттернов"""
        
        issues = []
        dangerous_patterns = [
            (r'eval\s*\(', 'eval() function detected'),
            (r'exec\s*\(', 'exec() function detected'),
            (r'shell=True', 'subprocess with shell=True'),
            (r'requests\.get\([^)]*\)(?!.*timeout)', 'requests without timeout'),
            (r'pickle\.loads', 'pickle deserialization'),
        ]
        
        for py_file in Path('src').rglob('*.py'):
            try:
                content = py_file.read_text(encoding='utf-8')
                
                for pattern, description in dangerous_patterns:
                    import re
                    if re.search(pattern, content):
                        issues.append(f"DANGEROUS_PATTERN in {py_file}: {description}")
                        
            except Exception:
                continue
                
        return issues
        
    def _check_gitignore(self) -> List[str]:
        """Проверка .gemignore"""
        
        issues = []
        gitignore_path = Path('.gitignore')
        
        if not gitignore_path.exists():
            issues.append("Missing .gitignore file")
            return issues
            
        try:
            content = gitignore_path.read_text(encoding='utf-8')
            
            required_patterns = [
                '.env',
                '*.env',
                'secrets.env',
                'logs/',
                '*.log'
            ]
            
            for pattern in required_patterns:
                if pattern not in content:
                    issues.append(f".gitignore missing pattern: {pattern}")
                    
        except Exception as e:
            issues.append(f"Cannot read .gitignore: {e}")
            
        return issues
        
    def _check_config(self) -> List[str]:
        """Проверка конфигурации"""
        
        issues = []
        config_files = [
            'config/config.yaml',
            'config/secrets.env.example',
            'README.md'
        ]
        
        for config_file in config_files:
            config_path = Path(config_file)
            if not config_path.exists():
                issues.append(f"Missing config file: {config_file}")
                
        return issues
        
    def try_external_tools(self):
        """Попытка запуска внешних инструментов"""
        
        print("🔧 Attempting to run external security tools...")
        
        tools = [
            ('bandit', ['bandit', '-r', 'src/', '-f', 'json']),
            ('pip_audit', ['pip-audit', '--format=json']),
            ('safety', ['safety', 'check', '--json'])
        ]
        
        for tool_name, cmd in tools:
            try:
                print(f"   Trying {tool_name}...")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    self.results[tool_name] = {
                        'status': 'completed',
                        'output': 'Success'
                    }
                    print(f"   ✅ {tool_name}: SUCCESS")
                else:
                    self.results[tool_name] = {
                        'status': 'error',
                        'output': result.stderr or 'Unknown error'
                    }
                    print(f"   ❌ {tool_name}: FAILED")
                    
            except subprocess.TimeoutExpired:
                self.results[tool_name] = {
                    'status': 'timeout',
                    'output': 'Command timed out'
                }
                print(f"   ⏰ {tool_name}: TIMEOUT")
            except FileNotFoundError:
                self.results[tool_name] = {
                    'status': 'not_found',
                    'output': f'{tool_name} not found in PATH'
                }
                print(f"   ❌ {tool_name}: NOT FOUND")
            except Exception as e:
                self.results[tool_name] = {
                    'status': 'error',
                    'output': str(e)
                }
                print(f"   ❌ {tool_name}: ERROR - {e}")
                
    def generate_report(self) -> str:
        """Генерация отчета"""
        
        report = []
        report.append("🔒 INTERNAL SECURITY SCAN REPORT")
        report.append("=" * 50)
        
        # Manual checks
        manual_issues = self.results['manual_checks']['issues']
        if manual_issues:
            report.append(f"\n⚠️ MANUAL CHECKS FOUND {len(manual_issues)} ISSUES:")
            for issue in manual_issues:
                report.append(f"   • {issue}")
        else:
            report.append(f"\n✅ MANUAL CHECKS: NO ISSUES FOUND")
            
        # Tool status
        report.append(f"\n📊 SECURITY TOOL STATUS:")
        for tool, result in self.results.items():
            if tool != 'manual_checks':
                status_symbol = {
                    'completed': '✅',
                    'skipped': '⏭️',
                    'error': '❌',
                    'timeout': '⏰',
                    'not_found': '❌'
                }.get(result['status'], '❓')
                
                report.append(f"   {status_symbol} {tool}: {result['status'].upper()}")
                
        # Summary
        completed_tools = sum(1 for r in self.results.values() 
                             if isinstance(r, dict) and r.get('status') == 'completed')
        total_tools = len([k for k in self.results.keys() if k != 'manual_checks'])
        
        report.append(f"\n📊 SUMMARY:")
        report.append(f"   🛠️ Tools completed: {completed_tools}/{total_tools}")
        report.append(f"   🔍 Manual issues: {len(manual_issues)}")
        
        if len(manual_issues) == 0 and completed_tools > 0:
            report.append(f"   ✅ Overall Status: SECURE")
        elif len(manual_issues) <= 2:
            report.append(f"   ⚠️ Overall Status: MOSTLY SECURE")
        else:
            report.append(f"   🚨 Overall Status: NEEDS ATTENTION")
            
        # Recommendations
        report.append(f"\n🔧 RECOMMENDATIONS:")
        
        if len(manual_issues) > 0:
            report.append(f"   1. Fix manual check issues")
            
        if completed_tools < total_tools:
            report.append(f"   2. Install security tools to PATH or use virtual environment")
            
        report.extend([
            f"   3. Replace test API keys with real ones",
            f"   4. Test emergency stop functionality",
            f"   5. Regular security audits"
        ])
        
        return '\n'.join(report)
        
    def save_results(self):
        """Сохранение результатов в файл"""
        
        report_content = self.generate_report()
        
        # Сохраняем человекочитаемый отчет
        with open('internal_security_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        # Сохраняем JSON с деталями
        json_report = {
            'scan_timestamp': str(Path.cwd()),
            'results': self.results,
            'summary': {
                'manual_issues_count': len(self.results['manual_checks']['issues']),
                'tools_completed': sum(1 for r in self.results.values() 
                                     if isinstance(r, dict) and r.get('status') == 'completed'),
                'total_tools': len([k for k in self.results.keys() if k != 'manual_checks'])
            }
        }
        
        with open('internal_security_report.json', 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False)

def main():
    """Основная функция сканирования"""
    
    print("🔒 Starting internal security scan...")
    
    scanner = InternalSecurityScanner()
    
    # Выполняем ручные проверки
    issues = scanner.run_manual_security_checks()
    
    # Пытаемся запустить внешние инструменты
    scanner.try_external_tools()
    
    # Генерируем и выводим отчет
    report = scanner.generate_report()
    print("\n" + report)
    
    # Сохраняем результаты
    scanner.save_results()
    
    print(f"\n📄 Reports saved:")
    print(f"   • internal_security_report.txt")
    print(f"   • internal_security_report.json")
    
    # Возвращаем код выхода
    critical_issues = len(scanner.results['manual_checks']['issues'])
    return 0 if critical_issues <= 2 else 1

if __name__ == "__main__":
    exit(main())
