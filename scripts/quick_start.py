#!/usr/bin/env python3
"""
Quick Start Script for Strategy Management
Demonstrates how to use the strategy management system
"""

import sys
import os
import asyncio
import yaml
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("🚀 BYBIT TRADING BOT - QUICK START")
    print("=" * 60)
    print("This script will guide you through setting up and using")
    print("the strategy management system.")
    print("=" * 60)

def check_requirements():
    """Check if requirements are met"""
    print("\n🔍 Checking requirements...")
    
    # Check if config files exist
    required_files = [
        "config/config.yaml",
        "config/advanced_strategies.yaml",
        "config/risk_config.yaml"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files found")
    return True

def load_strategies():
    """Load and display available strategies"""
    print("\n📋 Loading available strategies...")
    
    try:
        with open("config/advanced_strategies.yaml", 'r') as f:
            strategies_config = yaml.safe_load(f)
        
        strategies = strategies_config.get('strategies', {})
        enabled_strategies = []
        
        for name, config in strategies.items():
            if config.get('enabled', False):
                enabled_strategies.append(name)
        
        print(f"✅ Found {len(enabled_strategies)} enabled strategies:")
        for i, strategy in enumerate(enabled_strategies, 1):
            print(f"  {i}. {strategy}")
        
        return enabled_strategies
        
    except Exception as e:
        print(f"❌ Error loading strategies: {e}")
        return []

def demonstrate_strategy_modes():
    """Demonstrate different strategy modes"""
    print("\n🎮 STRATEGY MODES DEMONSTRATION")
    print("-" * 40)
    
    # Import strategy manager
    try:
        from services.strategy_manager import StrategyManager
        
        # Load config
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        strategy_manager = StrategyManager(config)
        
        print("1. 🔄 SINGLE MODE - One strategy at a time")
        print("   Best for: Testing, debugging, focused trading")
        print("   Example: switch_single momentum")
        print("")
        
        print("2. ⚡ PARALLEL MODE - Multiple strategies simultaneously")
        print("   Best for: Diversification, different market conditions")
        print("   Example: enable_parallel momentum,mean_reversion")
        print("")
        
        print("3. 📊 PORTFOLIO MODE - Capital allocation across strategies")
        print("   Best for: Risk management, professional trading")
        print("   Example: setup_portfolio momentum:0.6,mean_reversion:0.4")
        print("")
        
        print("4. 🤖 ADAPTIVE MODE - Automatic strategy switching")
        print("   Best for: Hands-off trading, optimization")
        print("   Example: enable_adaptive")
        print("")
        
        return strategy_manager
        
    except ImportError as e:
        print(f"❌ Error importing strategy manager: {e}")
        return None

def interactive_demo():
    """Run interactive demo"""
    print("\n🎯 INTERACTIVE DEMO")
    print("-" * 40)
    print("Let's try some strategy management commands!")
    print("")
    
    try:
        from services.strategy_manager import StrategyManager
        
        # Load config
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        strategy_manager = StrategyManager(config)
        
        # Demo commands
        demo_commands = [
            ("list_strategies", "Show available strategies"),
            ("status", "Show current status"),
            ("switch_single", ["momentum"], "Switch to single strategy mode"),
            ("status", "Show updated status"),
            ("enable_parallel", ["momentum,mean_reversion"], "Enable parallel strategies"),
            ("status", "Show final status"),
        ]
        
        for command, args, description in demo_commands:
            print(f"🔧 {description}")
            print(f"   Command: {command} {' '.join(args) if args else ''}")
            
            result = strategy_manager.execute_command(command, args)
            print(f"   Result: {result}")
            print("")
            
            # Ask user to continue
            try:
                input("   Press Enter to continue...")
            except KeyboardInterrupt:
                print("\n   Demo interrupted by user.")
                break
        
        print("✅ Demo completed!")
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")

def show_next_steps():
    """Show next steps for the user"""
    print("\n🎯 NEXT STEPS")
    print("-" * 40)
    print("1. 📚 Read the full guide:")
    print("   docs/STRATEGY_MANAGEMENT_GUIDE.md")
    print("")
    print("2. 🚀 Start the interactive manager:")
    print("   python scripts/strategy_manager.py")
    print("")
    print("3. 🎮 Try demo mode:")
    print("   python scripts/strategy_manager.py --demo")
    print("")
    print("4. ⚙️  Configure your strategies:")
    print("   Edit config/advanced_strategies.yaml")
    print("")
    print("5. 🔒 Set up security:")
    print("   ./scripts/setup_security.sh")
    print("")
    print("6. 🐳 Start with Docker:")
    print("   docker-compose up -d")
    print("")
    print("7. 📊 Monitor performance:")
    print("   Visit http://localhost:3000 (Grafana)")
    print("")

def main():
    """Main function"""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements not met. Please check your setup.")
        return
    
    # Load strategies
    strategies = load_strategies()
    if not strategies:
        print("\n❌ No strategies found. Please check your configuration.")
        return
    
    # Demonstrate modes
    strategy_manager = demonstrate_strategy_modes()
    if not strategy_manager:
        print("\n❌ Could not initialize strategy manager.")
        return
    
    # Ask if user wants interactive demo
    try:
        response = input("\n🎮 Would you like to try an interactive demo? (y/n): ").lower()
        if response in ['y', 'yes']:
            interactive_demo()
    except KeyboardInterrupt:
        print("\n\n👋 Demo skipped.")
    
    # Show next steps
    show_next_steps()
    
    print("\n🎉 Quick start completed!")
    print("Happy trading! 🚀")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
