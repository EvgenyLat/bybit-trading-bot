#!/bin/bash

# Security Setup Script
# Sets up security configurations and validates system security

set -e

echo "🔒 Setting up security configurations..."

# Create security directories
mkdir -p logs/security
mkdir -p config/encrypted
mkdir -p backups/security

# Set proper permissions
chmod 700 config/encrypted
chmod 600 config/secrets.env.example
chmod 700 logs/security

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp config/secrets.env.example .env
    echo "⚠️  Please edit .env file with your actual API keys and passwords!"
    echo "⚠️  Generate a strong master password for encryption!"
fi

# Generate master password if not set
if ! grep -q "MASTER_PASSWORD=" .env || grep -q "your_strong_master_password_here" .env; then
    echo "Generating strong master password..."
    MASTER_PASS=$(openssl rand -base64 32)
    sed -i "s/MASTER_PASSWORD=.*/MASTER_PASSWORD=$MASTER_PASS/" .env
    echo "✅ Master password generated and saved to .env"
fi

# Create security audit log
touch logs/security/security_audit.log
chmod 600 logs/security/security_audit.log

# Create emergency stop file
touch config/emergency_stop
chmod 600 config/emergency_stop
echo "false" > config/emergency_stop

# Create IP whitelist
touch config/ip_whitelist.txt
chmod 600 config/ip_whitelist.txt
echo "# Add trusted IP addresses here" > config/ip_whitelist.txt
echo "127.0.0.1" >> config/ip_whitelist.txt

echo "✅ Security setup completed!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your actual API keys"
echo "2. Review security settings"
echo "3. Add trusted IPs to config/ip_whitelist.txt"
echo "4. Test emergency stop functionality"
echo ""
echo "⚠️  IMPORTANT: Keep your .env file secure and never commit it to version control!"

