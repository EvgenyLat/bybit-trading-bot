"""
Interactive Strategy Management Interface
Command-line interface for managing trading strategies
"""

import asyncio
import logging
import json
import sys
from typing import Dict, List, Optional
from datetime import datetime
import yaml

# Import our strategy manager
from services.strategy_manager import StrategyManager, StrategyMode

logger = logging.getLogger(__name__)


class StrategyManagementInterface:
    """Interactive interface for strategy management"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.strategy_manager = StrategyManager(self.config)
        self.running = True
        
    def _load_config(self) -> Dict:
        """Load configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def print_welcome(self):
        """Print welcome message"""
        print("=" * 60)
        print("🤖 BYBIT TRADING BOT - STRATEGY MANAGEMENT")
        print("=" * 60)
        print("Welcome to the interactive strategy management system!")
        print("You can manage multiple trading strategies here.")
        print("")
        print("📋 AVAILABLE STRATEGIES:")
        strategies = self.strategy_manager.get_available_strategies()
        for i, strategy in enumerate(strategies, 1):
            print(f"  {i}. {strategy}")
        print("")
        print("💡 STRATEGY MODES:")
        print("  • Single: Only one strategy active at a time")
        print("  • Parallel: Multiple strategies running simultaneously")
        print("  • Portfolio: Allocate capital across strategies")
        print("  • Adaptive: Automatically switch based on performance")
        print("")
        print("Type 'help' for available commands or 'quit' to exit.")
        print("=" * 60)
    
    def print_help(self):
        """Print help information"""
        print("\n📖 AVAILABLE COMMANDS:")
        print("-" * 40)
        
        commands = self.strategy_manager.get_strategy_commands()
        for command, description in commands.items():
            print(f"  {command:<20} - {description}")
        
        print("\n🎯 STRATEGY MODES EXPLAINED:")
        print("-" * 40)
        print("1. SINGLE MODE:")
        print("   • Only one strategy runs at a time")
        print("   • Best for: Testing individual strategies")
        print("   • Command: switch_single <strategy_name>")
        print("")
        print("2. PARALLEL MODE:")
        print("   • Multiple strategies run simultaneously")
        print("   • Best for: Diversification, different market conditions")
        print("   • Command: enable_parallel <strategy1,strategy2,...>")
        print("")
        print("3. PORTFOLIO MODE:")
        print("   • Capital allocated across strategies")
        print("   • Best for: Risk management, balanced approach")
        print("   • Command: setup_portfolio <strategy1:0.4,strategy2:0.6>")
        print("")
        print("4. ADAPTIVE MODE:")
        print("   • Automatically switches based on performance")
        print("   • Best for: Hands-off trading, optimization")
        print("   • Command: enable_adaptive")
        print("")
        print("📊 MONITORING COMMANDS:")
        print("-" * 40)
        print("  status              - Show current strategy status")
        print("  performance_report - Show detailed performance metrics")
        print("  list_strategies    - List all available strategies")
        print("")
    
    def print_current_status(self):
        """Print current strategy status"""
        print("\n📊 CURRENT STATUS:")
        print("-" * 40)
        
        status = {
            'mode': self.strategy_manager.strategy_mode.value,
            'active_strategies': list(self.strategy_manager.active_strategies.keys()),
            'available_strategies': self.strategy_manager.get_available_strategies()
        }
        
        print(f"Mode: {status['mode'].upper()}")
        print(f"Active strategies: {', '.join(status['active_strategies']) if status['active_strategies'] else 'None'}")
        print(f"Available strategies: {', '.join(status['available_strategies'])}")
        
        # Show strategy details
        if status['active_strategies']:
            print("\n🔍 ACTIVE STRATEGY DETAILS:")
            for strategy_name in status['active_strategies']:
                strategy_status = self.strategy_manager.get_strategy_status(strategy_name)
                print(f"  • {strategy_name}: {strategy_status.value}")
    
    def print_performance_report(self):
        """Print performance report"""
        print("\n📈 PERFORMANCE REPORT:")
        print("-" * 40)
        
        report = self.strategy_manager.get_strategy_performance_report()
        
        if not report:
            print("No performance data available yet.")
            return
        
        print(f"Mode: {report['mode'].upper()}")
        print(f"Active strategies: {', '.join(report['active_strategies'])}")
        
        if report['strategy_performance']:
            print("\n📊 STRATEGY PERFORMANCE:")
            for strategy_name, performance in report['strategy_performance'].items():
                print(f"\n  {strategy_name.upper()}:")
                print(f"    Total trades: {performance['total_trades']}")
                print(f"    Win rate: {performance['win_rate']:.1%}")
                print(f"    Total PnL: {performance['total_pnl']:.2f}")
                print(f"    Max drawdown: {performance['max_drawdown']:.2f}")
                print(f"    Sharpe ratio: {performance['sharpe_ratio']:.2f}")
                print(f"    Profit factor: {performance['profit_factor']:.2f}")
        
        if report['recommendations']:
            print("\n💡 RECOMMENDATIONS:")
            for recommendation in report['recommendations']:
                print(f"  • {recommendation}")
    
    def handle_command(self, command: str, args: List[str] = None) -> bool:
        """Handle user command"""
        try:
            if command == 'help':
                self.print_help()
                return True
            
            elif command == 'status':
                self.print_current_status()
                return True
            
            elif command == 'performance_report':
                self.print_performance_report()
                return True
            
            elif command == 'list_strategies':
                strategies = self.strategy_manager.get_available_strategies()
                print(f"\n📋 Available strategies: {', '.join(strategies)}")
                return True
            
            elif command == 'quit' or command == 'exit':
                print("\n👋 Goodbye! Shutting down strategy management...")
                return False
            
            else:
                # Execute strategy manager command
                result = self.strategy_manager.execute_command(command, args)
                print(f"\n✅ {result}")
                return True
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return True
    
    def parse_command(self, user_input: str) -> tuple:
        """Parse user input into command and arguments"""
        parts = user_input.strip().split()
        if not parts:
            return '', []
        
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        return command, args
    
    def run_interactive_mode(self):
        """Run interactive command mode"""
        self.print_welcome()
        
        while self.running:
            try:
                user_input = input("\n🤖 Strategy Manager > ").strip()
                
                if not user_input:
                    continue
                
                command, args = self.parse_command(user_input)
                
                if not self.handle_command(command, args):
                    self.running = False
                    
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user. Goodbye!")
                break
            except EOFError:
                print("\n\n👋 End of input. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
    
    def run_demo_mode(self):
        """Run demo mode with example commands"""
        print("\n🎮 DEMO MODE - Example Strategy Management")
        print("=" * 50)
        
        demo_commands = [
            ("list_strategies", []),
            ("status", []),
            ("switch_single", ["momentum"]),
            ("status", []),
            ("enable_parallel", ["momentum,mean_reversion"]),
            ("status", []),
            ("setup_portfolio", ["momentum:0.6,mean_reversion:0.4"]),
            ("status", []),
            ("performance_report", []),
        ]
        
        for command, args in demo_commands:
            print(f"\n🔧 Executing: {command} {' '.join(args)}")
            self.handle_command(command, args)
            input("\nPress Enter to continue...")
        
        print("\n✅ Demo completed!")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Strategy Management Interface')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    try:
        interface = StrategyManagementInterface(args.config)
        
        if args.demo:
            interface.run_demo_mode()
        else:
            interface.run_interactive_mode()
            
    except Exception as e:
        print(f"❌ Error starting interface: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
