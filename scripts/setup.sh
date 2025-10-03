#!/bin/bash

# Bybit Trading Bot Setup Script
# This script sets up the complete trading bot environment

set -e

echo "🚀 Setting up Bybit Trading Bot..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs data models config infra/timescaledb infra/prometheus infra/grafana

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📋 Creating .env file from template..."
    cp config/secrets.env.example .env
    echo "⚠️  Please edit .env file with your actual API keys and passwords!"
fi

# Create TimescaleDB initialization script
echo "🗄️  Creating TimescaleDB initialization script..."
cat > infra/timescaledb/init.sql << 'EOF'
-- Create extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create OHLCV data table
CREATE TABLE IF NOT EXISTS ohlcv_data (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    open DECIMAL(20,8) NOT NULL,
    high DECIMAL(20,8) NOT NULL,
    low DECIMAL(20,8) NOT NULL,
    close DECIMAL(20,8) NOT NULL,
    volume DECIMAL(20,8) NOT NULL,
    turnover DECIMAL(20,8),
    PRIMARY KEY (timestamp, symbol, timeframe)
);

-- Create hypertable
SELECT create_hypertable('ohlcv_data', 'timestamp', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe 
ON ohlcv_data (symbol, timeframe, timestamp DESC);

-- Create trades table
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    pnl DECIMAL(20,8),
    commission DECIMAL(20,8),
    order_id VARCHAR(50),
    strategy VARCHAR(50)
);

-- Create hypertable for trades
SELECT create_hypertable('trades', 'timestamp', if_not_exists => TRUE);

-- Create signals table
CREATE TABLE IF NOT EXISTS signals (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(20) NOT NULL,
    signal_strength DECIMAL(10,4) NOT NULL,
    confidence DECIMAL(10,4) NOT NULL,
    technical_signal DECIMAL(10,4),
    ml_signal DECIMAL(10,4),
    features JSONB
);

-- Create hypertable for signals
SELECT create_hypertable('signals', 'timestamp', if_not_exists => TRUE);

-- Create performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    metric_value DECIMAL(20,8) NOT NULL,
    metric_type VARCHAR(20) NOT NULL
);

-- Create hypertable for performance metrics
SELECT create_hypertable('performance_metrics', 'timestamp', if_not_exists => TRUE);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trading_user;
EOF

# Create Prometheus configuration
echo "📊 Creating Prometheus configuration..."
cat > infra/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'trading-bot'
    static_configs:
      - targets: ['trading_bot:8000']
    scrape_interval: 5s
    metrics_path: '/metrics'
EOF

# Create Grafana provisioning
echo "📈 Creating Grafana provisioning..."
mkdir -p infra/grafana/provisioning/datasources
mkdir -p infra/grafana/provisioning/dashboards

cat > infra/grafana/provisioning/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

cat > infra/grafana/provisioning/dashboards/dashboard.yml << 'EOF'
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF

# Create basic Grafana dashboard
mkdir -p infra/grafana/dashboards
cat > infra/grafana/dashboards/trading-bot.json << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "Bybit Trading Bot",
    "tags": ["trading", "crypto"],
    "style": "dark",
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Equity Curve",
        "type": "graph",
        "targets": [
          {
            "expr": "equity_usd",
            "legendFormat": "Equity (USD)"
          }
        ],
        "yAxes": [
          {
            "label": "USD",
            "min": 0
          }
        ],
        "xAxes": [
          {
            "type": "time"
          }
        ]
      },
      {
        "id": 2,
        "title": "Daily P&L",
        "type": "graph",
        "targets": [
          {
            "expr": "daily_pnl",
            "legendFormat": "Daily P&L"
          }
        ]
      },
      {
        "id": 3,
        "title": "Open Positions",
        "type": "stat",
        "targets": [
          {
            "expr": "open_positions",
            "legendFormat": "Positions"
          }
        ]
      },
      {
        "id": 4,
        "title": "Win Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "win_rate",
            "legendFormat": "Win Rate %"
          }
        ],
        "unit": "percent"
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
EOF

echo "✅ Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your API keys and passwords"
echo "2. Run: docker-compose up -d"
echo "3. Access Grafana at http://localhost:3000 (admin/admin)"
echo "4. Access Prometheus at http://localhost:9090"
echo "5. Access MLflow at http://localhost:5000"
echo ""
echo "⚠️  Remember to:"
echo "- Use testnet first (BYBIT_TESTNET=true)"
echo "- Start with small amounts"
echo "- Monitor the bot closely"
echo "- Set up proper risk limits"
echo ""
echo "🔧 To start the bot:"
echo "docker-compose up trading_bot"
echo ""
echo "🔧 To stop everything:"
echo "docker-compose down"

