#!/bin/bash

# System Health Check Script
# Checks system components without requiring Python execution

echo "🔍 SYSTEM HEALTH CHECK"
echo "======================"

# Check project structure
echo "📁 Checking project structure..."

REQUIRED_DIRS=("config" "src" "services" "tests" "scripts" "infra")
REQUIRED_FILES=("README.md" "requirements.txt" "config/config.yaml" "config/risk_config.yaml")

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "✅ Directory $dir exists"
    else
        echo "❌ Directory $dir missing"
    fi
done

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ File $file exists"
    else
        echo "❌ File $file missing"
    fi
done

echo ""

# Check services structure
echo "🔧 Checking services structure..."

SERVICES=("data_collector" "feature_engineering" "model_training" "signal_service" "risk_manager" "executor" "backtester" "security" "error_handling" "concurrency" "security_monitoring" "portfolio_manager" "fundamental_analysis" "reinforcement_learning")

for service in "${SERVICES[@]}"; do
    if [ -d "services/$service" ]; then
        if [ -f "services/$service/__init__.py" ]; then
            echo "✅ Service $service properly structured"
        else
            echo "⚠️  Service $service missing __init__.py"
        fi
    else
        echo "❌ Service $service missing"
    fi
done

echo ""

# Check configuration files
echo "⚙️  Checking configuration files..."

if [ -f "config/config.yaml" ]; then
    echo "✅ Main config file exists"
    
    # Check for required sections
    if grep -q "api:" config/config.yaml; then
        echo "✅ API configuration section found"
    else
        echo "❌ API configuration section missing"
    fi
    
    if grep -q "trading:" config/config.yaml; then
        echo "✅ Trading configuration section found"
    else
        echo "❌ Trading configuration section missing"
    fi
    
    if grep -q "risk:" config/config.yaml; then
        echo "✅ Risk configuration section found"
    else
        echo "❌ Risk configuration section missing"
    fi
else
    echo "❌ Main config file missing"
fi

if [ -f "config/risk_config.yaml" ]; then
    echo "✅ Risk config file exists"
    
    # Check for critical risk parameters
    if grep -q "risk_per_trade:" config/risk_config.yaml; then
        echo "✅ Risk per trade parameter found"
    else
        echo "❌ Risk per trade parameter missing"
    fi
    
    if grep -q "emergency_stop_enabled:" config/risk_config.yaml; then
        echo "✅ Emergency stop parameter found"
    else
        echo "❌ Emergency stop parameter missing"
    fi
else
    echo "❌ Risk config file missing"
fi

if [ -f "config/secrets.env.example" ]; then
    echo "✅ Secrets template exists"
    
    # Check for required environment variables
    REQUIRED_ENV_VARS=("BYBIT_API_KEY" "BYBIT_API_SECRET" "MASTER_PASSWORD" "TELEGRAM_BOT_TOKEN")
    
    for var in "${REQUIRED_ENV_VARS[@]}"; do
        if grep -q "$var=" config/secrets.env.example; then
            echo "✅ Environment variable $var found"
        else
            echo "❌ Environment variable $var missing"
        fi
    done
else
    echo "❌ Secrets template missing"
fi

echo ""

# Check Docker configuration
echo "🐳 Checking Docker configuration..."

if [ -f "infra/docker-compose.yml" ]; then
    echo "✅ Docker Compose file exists"
    
    # Check for required services
    REQUIRED_SERVICES=("timescaledb" "redis" "prometheus" "grafana" "mlflow" "trading_bot")
    
    for service in "${REQUIRED_SERVICES[@]}"; do
        if grep -q "$service:" infra/docker-compose.yml; then
            echo "✅ Docker service $service configured"
        else
            echo "❌ Docker service $service missing"
        fi
    done
else
    echo "❌ Docker Compose file missing"
fi

if [ -f "infra/Dockerfile" ]; then
    echo "✅ Dockerfile exists"
    
    # Check for Python installation
    if grep -q "python:" infra/Dockerfile; then
        echo "✅ Python base image specified"
    else
        echo "❌ Python base image not specified"
    fi
    
    # Check for TA-Lib installation
    if grep -q "ta-lib" infra/Dockerfile; then
        echo "✅ TA-Lib installation found"
    else
        echo "❌ TA-Lib installation missing"
    fi
else
    echo "❌ Dockerfile missing"
fi

echo ""

# Check requirements.txt
echo "📦 Checking dependencies..."

if [ -f "requirements.txt" ]; then
    echo "✅ Requirements file exists"
    
    # Count dependencies
    DEP_COUNT=$(grep -c "^[a-zA-Z]" requirements.txt)
    echo "📊 Total dependencies: $DEP_COUNT"
    
    # Check for critical dependencies
    CRITICAL_DEPS=("pandas" "numpy" "pybit" "ccxt" "scikit-learn" "tensorflow" "torch" "xgboost" "redis" "timescaledb" "mlflow" "prometheus-client")
    
    for dep in "${CRITICAL_DEPS[@]}"; do
        if grep -q "$dep" requirements.txt; then
            echo "✅ Critical dependency $dep found"
        else
            echo "❌ Critical dependency $dep missing"
        fi
    done
else
    echo "❌ Requirements file missing"
fi

echo ""

# Check test files
echo "🧪 Checking test files..."

if [ -f "tests/test_security.py" ]; then
    echo "✅ Security tests exist"
    
    # Count test methods
    TEST_COUNT=$(grep -c "def test_" tests/test_security.py)
    echo "📊 Security test methods: $TEST_COUNT"
else
    echo "❌ Security tests missing"
fi

if [ -f "tests/test_system_comprehensive.py" ]; then
    echo "✅ Comprehensive system tests exist"
    
    # Count test methods
    TEST_COUNT=$(grep -c "def test_" tests/test_system_comprehensive.py)
    echo "📊 System test methods: $TEST_COUNT"
else
    echo "❌ Comprehensive system tests missing"
fi

echo ""

# Check scripts
echo "📜 Checking scripts..."

if [ -f "scripts/setup.sh" ]; then
    echo "✅ Setup script exists"
else
    echo "❌ Setup script missing"
fi

if [ -f "scripts/setup_security.sh" ]; then
    echo "✅ Security setup script exists"
else
    echo "❌ Security setup script missing"
fi

echo ""

# Check README
echo "📖 Checking documentation..."

if [ -f "README.md" ]; then
    echo "✅ README file exists"
    
    # Check for key sections
    if grep -q "## " README.md; then
        SECTION_COUNT=$(grep -c "## " README.md)
        echo "📊 README sections: $SECTION_COUNT"
    fi
    
    # Check for installation instructions
    if grep -qi "install" README.md; then
        echo "✅ Installation instructions found"
    else
        echo "❌ Installation instructions missing"
    fi
    
    # Check for usage instructions
    if grep -qi "usage" README.md; then
        echo "✅ Usage instructions found"
    else
        echo "❌ Usage instructions missing"
    fi
else
    echo "❌ README file missing"
fi

echo ""

# Check file permissions
echo "🔐 Checking file permissions..."

# Check for sensitive files with proper permissions
SENSITIVE_FILES=("config/secrets.env.example" ".env" "logs/" "config/encrypted/")

for file in "${SENSITIVE_FILES[@]}"; do
    if [ -e "$file" ]; then
        PERMS=$(stat -c "%a" "$file" 2>/dev/null || stat -f "%OLp" "$file" 2>/dev/null || echo "unknown")
        echo "📊 Permissions for $file: $PERMS"
        
        # Check if permissions are restrictive enough
        if [[ "$PERMS" == "600" ]] || [[ "$PERMS" == "700" ]]; then
            echo "✅ Secure permissions for $file"
        else
            echo "⚠️  Consider more restrictive permissions for $file"
        fi
    fi
done

echo ""

# Generate summary
echo "📋 SYSTEM HEALTH SUMMARY"
echo "======================="

# Count issues
ISSUES=0
WARNINGS=0

# Count files and directories
TOTAL_FILES=$(find . -type f | wc -l)
TOTAL_DIRS=$(find . -type d | wc -l)

echo "📊 Project statistics:"
echo "  - Total files: $TOTAL_FILES"
echo "  - Total directories: $TOTAL_DIRS"
echo "  - Services: ${#SERVICES[@]}"
echo "  - Dependencies: $DEP_COUNT"
echo "  - Test methods: $TEST_COUNT"

echo ""
echo "🎯 Next steps:"
echo "1. Install Python 3.10+ if not already installed"
echo "2. Install dependencies: pip install -r requirements.txt"
echo "3. Copy config/secrets.env.example to .env and fill in your API keys"
echo "4. Run security setup: ./scripts/setup_security.sh"
echo "5. Run comprehensive tests: python tests/test_system_comprehensive.py"
echo "6. Start with Docker: docker-compose up -d"

echo ""
echo "✅ System health check completed!"
