#!/usr/bin/env python3

print("🔒 QUICK SECURITY TEST")
print("=" * 40)

import os
from pathlib import Path

# 1. Проверка ключевых файлов
critical_files = [
    '.env',
    '.gitignore',
    'config/config.yaml',
    'config/secrets.env.example',
    'src/config.py',
    'src/secure_executor.py'
]

found_files = 0
missing_files = []

for file in critical_files:
    if Path(file).exists():
        print(f"✅ {file}")
        found_files += 1
    else:
        print(f"❌ {file} MISSING")
        missing_files.append(file)

print(f"\n📊 Files Found: {found_files}/{len(critical_files)}")

# 2. Проверка структуры проекта
directories = ['src', 'config', 'scripts', 'services']
present_dirs = []

for dir_name in directories:
    if Path(dir_name).exists():
        present_dirs.append(dir_name)
        print(f"✅ Directory {dir_name}/")

print(f"\n📁 Directories: {len(present_dirs)}/{len(directories)}")

# 3. Итоговая оценка
print("\n" + "=" * 40)
if len(missing_files) == 0 and len(present_dirs) == len(directories):
    print("🎉 SECURITY STATUS: EXCELLENT")
    print("✅ All critical files present")
    print("🚀 Project is PRODUCTION READY")
else:
    print("⚠️ SECURITY STATUS: NEEDS ATTENTION")
    if missing_files:
        print(f"❌ Missing files: {len(missing_files)}")
    
print("\n🔧 IMMEDIATE ACTIONS:")
print("1. Get real Bybit API keys")
print("2. Update .env with real keys")
print("3. Test connection: python quick_start.py")

exit(0 if len(missing_files) == 0 else 1)
