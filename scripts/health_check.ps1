# System Health Check Script (PowerShell)
# Checks system components without requiring Python execution

Write-Host "🔍 SYSTEM HEALTH CHECK" -ForegroundColor Cyan
Write-Host "======================" -ForegroundColor Cyan

# Check project structure
Write-Host "📁 Checking project structure..." -ForegroundColor Yellow

$RequiredDirs = @("config", "src", "services", "tests", "scripts", "infra")
$RequiredFiles = @("README.md", "requirements.txt", "config/config.yaml", "config/risk_config.yaml")

foreach ($dir in $RequiredDirs) {
    if (Test-Path $dir) {
        Write-Host "✅ Directory $dir exists" -ForegroundColor Green
    } else {
        Write-Host "❌ Directory $dir missing" -ForegroundColor Red
    }
}

foreach ($file in $RequiredFiles) {
    if (Test-Path $file) {
        Write-Host "✅ File $file exists" -ForegroundColor Green
    } else {
        Write-Host "❌ File $file missing" -ForegroundColor Red
    }
}

Write-Host ""

# Check services structure
Write-Host "🔧 Checking services structure..." -ForegroundColor Yellow

$Services = @("data_collector", "feature_engineering", "model_training", "signal_service", "risk_manager", "executor", "backtester", "security", "error_handling", "concurrency", "security_monitoring", "portfolio_manager", "fundamental_analysis", "reinforcement_learning")

foreach ($service in $Services) {
    $servicePath = "services/$service"
    if (Test-Path $servicePath) {
        if (Test-Path "$servicePath/__init__.py") {
            Write-Host "✅ Service $service properly structured" -ForegroundColor Green
        } else {
            Write-Host "⚠️  Service $service missing __init__.py" -ForegroundColor Yellow
        }
    } else {
        Write-Host "❌ Service $service missing" -ForegroundColor Red
    }
}

Write-Host ""

# Check configuration files
Write-Host "⚙️  Checking configuration files..." -ForegroundColor Yellow

if (Test-Path "config/config.yaml") {
    Write-Host "✅ Main config file exists" -ForegroundColor Green
    
    $configContent = Get-Content "config/config.yaml" -Raw
    
    if ($configContent -match "api:") {
        Write-Host "✅ API configuration section found" -ForegroundColor Green
    } else {
        Write-Host "❌ API configuration section missing" -ForegroundColor Red
    }
    
    if ($configContent -match "trading:") {
        Write-Host "✅ Trading configuration section found" -ForegroundColor Green
    } else {
        Write-Host "❌ Trading configuration section missing" -ForegroundColor Red
    }
    
    if ($configContent -match "risk:") {
        Write-Host "✅ Risk configuration section found" -ForegroundColor Green
    } else {
        Write-Host "❌ Risk configuration section missing" -ForegroundColor Red
    }
} else {
    Write-Host "❌ Main config file missing" -ForegroundColor Red
}

if (Test-Path "config/risk_config.yaml") {
    Write-Host "✅ Risk config file exists" -ForegroundColor Green
    
    $riskConfigContent = Get-Content "config/risk_config.yaml" -Raw
    
    if ($riskConfigContent -match "risk_per_trade:") {
        Write-Host "✅ Risk per trade parameter found" -ForegroundColor Green
    } else {
        Write-Host "❌ Risk per trade parameter missing" -ForegroundColor Red
    }
    
    if ($riskConfigContent -match "emergency_stop_enabled:") {
        Write-Host "✅ Emergency stop parameter found" -ForegroundColor Green
    } else {
        Write-Host "❌ Emergency stop parameter missing" -ForegroundColor Red
    }
} else {
    Write-Host "❌ Risk config file missing" -ForegroundColor Red
}

if (Test-Path "config/secrets.env.example") {
    Write-Host "✅ Secrets template exists" -ForegroundColor Green
    
    $secretsContent = Get-Content "config/secrets.env.example" -Raw
    $RequiredEnvVars = @("BYBIT_API_KEY", "BYBIT_API_SECRET", "MASTER_PASSWORD", "TELEGRAM_BOT_TOKEN")
    
    foreach ($var in $RequiredEnvVars) {
        if ($secretsContent -match "$var=") {
            Write-Host "✅ Environment variable $var found" -ForegroundColor Green
        } else {
            Write-Host "❌ Environment variable $var missing" -ForegroundColor Red
        }
    }
} else {
    Write-Host "❌ Secrets template missing" -ForegroundColor Red
}

Write-Host ""

# Check Docker configuration
Write-Host "🐳 Checking Docker configuration..." -ForegroundColor Yellow

if (Test-Path "infra/docker-compose.yml") {
    Write-Host "✅ Docker Compose file exists" -ForegroundColor Green
    
    $dockerComposeContent = Get-Content "infra/docker-compose.yml" -Raw
    $RequiredServices = @("timescaledb", "redis", "prometheus", "grafana", "mlflow", "trading_bot")
    
    foreach ($service in $RequiredServices) {
        if ($dockerComposeContent -match "$service:") {
            Write-Host "✅ Docker service $service configured" -ForegroundColor Green
        } else {
            Write-Host "❌ Docker service $service missing" -ForegroundColor Red
        }
    }
} else {
    Write-Host "❌ Docker Compose file missing" -ForegroundColor Red
}

if (Test-Path "infra/Dockerfile") {
    Write-Host "✅ Dockerfile exists" -ForegroundColor Green
    
    $dockerfileContent = Get-Content "infra/Dockerfile" -Raw
    
    if ($dockerfileContent -match "python:") {
        Write-Host "✅ Python base image specified" -ForegroundColor Green
    } else {
        Write-Host "❌ Python base image not specified" -ForegroundColor Red
    }
    
    if ($dockerfileContent -match "ta-lib") {
        Write-Host "✅ TA-Lib installation found" -ForegroundColor Green
    } else {
        Write-Host "❌ TA-Lib installation missing" -ForegroundColor Red
    }
} else {
    Write-Host "❌ Dockerfile missing" -ForegroundColor Red
}

Write-Host ""

# Check requirements.txt
Write-Host "📦 Checking dependencies..." -ForegroundColor Yellow

if (Test-Path "requirements.txt") {
    Write-Host "✅ Requirements file exists" -ForegroundColor Green
    
    $requirementsContent = Get-Content "requirements.txt"
    $depCount = ($requirementsContent | Where-Object { $_ -match "^[a-zA-Z]" }).Count
    Write-Host "📊 Total dependencies: $depCount" -ForegroundColor Cyan
    
    $CriticalDeps = @("pandas", "numpy", "pybit", "ccxt", "scikit-learn", "tensorflow", "torch", "xgboost", "redis", "timescaledb", "mlflow", "prometheus-client")
    
    foreach ($dep in $CriticalDeps) {
        if ($requirementsContent -match $dep) {
            Write-Host "✅ Critical dependency $dep found" -ForegroundColor Green
        } else {
            Write-Host "❌ Critical dependency $dep missing" -ForegroundColor Red
        }
    }
} else {
    Write-Host "❌ Requirements file missing" -ForegroundColor Red
}

Write-Host ""

# Check test files
Write-Host "🧪 Checking test files..." -ForegroundColor Yellow

if (Test-Path "tests/test_security.py") {
    Write-Host "✅ Security tests exist" -ForegroundColor Green
    
    $securityTestContent = Get-Content "tests/test_security.py"
    $testCount = ($securityTestContent | Where-Object { $_ -match "def test_" }).Count
    Write-Host "📊 Security test methods: $testCount" -ForegroundColor Cyan
} else {
    Write-Host "❌ Security tests missing" -ForegroundColor Red
}

if (Test-Path "tests/test_system_comprehensive.py") {
    Write-Host "✅ Comprehensive system tests exist" -ForegroundColor Green
    
    $systemTestContent = Get-Content "tests/test_system_comprehensive.py"
    $testCount = ($systemTestContent | Where-Object { $_ -match "def test_" }).Count
    Write-Host "📊 System test methods: $testCount" -ForegroundColor Cyan
} else {
    Write-Host "❌ Comprehensive system tests missing" -ForegroundColor Red
}

Write-Host ""

# Check scripts
Write-Host "📜 Checking scripts..." -ForegroundColor Yellow

if (Test-Path "scripts/setup.sh") {
    Write-Host "✅ Setup script exists" -ForegroundColor Green
} else {
    Write-Host "❌ Setup script missing" -ForegroundColor Red
}

if (Test-Path "scripts/setup_security.sh") {
    Write-Host "✅ Security setup script exists" -ForegroundColor Green
} else {
    Write-Host "❌ Security setup script missing" -ForegroundColor Red
}

Write-Host ""

# Check README
Write-Host "📖 Checking documentation..." -ForegroundColor Yellow

if (Test-Path "README.md") {
    Write-Host "✅ README file exists" -ForegroundColor Green
    
    $readmeContent = Get-Content "README.md"
    
    if ($readmeContent -match "## ") {
        $sectionCount = ($readmeContent | Where-Object { $_ -match "## " }).Count
        Write-Host "📊 README sections: $sectionCount" -ForegroundColor Cyan
    }
    
    if ($readmeContent -match -i "install") {
        Write-Host "✅ Installation instructions found" -ForegroundColor Green
    } else {
        Write-Host "❌ Installation instructions missing" -ForegroundColor Red
    }
    
    if ($readmeContent -match -i "usage") {
        Write-Host "✅ Usage instructions found" -ForegroundColor Green
    } else {
        Write-Host "❌ Usage instructions missing" -ForegroundColor Red
    }
} else {
    Write-Host "❌ README file missing" -ForegroundColor Red
}

Write-Host ""

# Generate summary
Write-Host "📋 SYSTEM HEALTH SUMMARY" -ForegroundColor Cyan
Write-Host "=======================" -ForegroundColor Cyan

# Count files and directories
$totalFiles = (Get-ChildItem -Recurse -File).Count
$totalDirs = (Get-ChildItem -Recurse -Directory).Count

Write-Host "📊 Project statistics:" -ForegroundColor Cyan
Write-Host "  - Total files: $totalFiles" -ForegroundColor White
Write-Host "  - Total directories: $totalDirs" -ForegroundColor White
Write-Host "  - Services: $($Services.Count)" -ForegroundColor White
Write-Host "  - Dependencies: $depCount" -ForegroundColor White
Write-Host "  - Test methods: $testCount" -ForegroundColor White

Write-Host ""
Write-Host "🎯 Next steps:" -ForegroundColor Yellow
Write-Host "1. Install Python 3.10+ if not already installed" -ForegroundColor White
Write-Host "2. Install dependencies: pip install -r requirements.txt" -ForegroundColor White
Write-Host "3. Copy config/secrets.env.example to .env and fill in your API keys" -ForegroundColor White
Write-Host "4. Run security setup: ./scripts/setup_security.sh" -ForegroundColor White
Write-Host "5. Run comprehensive tests: python tests/test_system_comprehensive.py" -ForegroundColor White
Write-Host "6. Start with Docker: docker-compose up -d" -ForegroundColor White

Write-Host ""
Write-Host "✅ System health check completed!" -ForegroundColor Green
