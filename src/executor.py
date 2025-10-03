"""
Bybit Trade Executor Module for Trading Bot
Handles order placement and trade execution using Bybit API
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
import pandas as pd
from pybit.unified_trading import HTTP

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Bybit order types"""
    MARKET = "Market"
    LIMIT = "Limit"
    STOP = "Stop"
    STOP_LIMIT = "StopLimit"


class OrderSide(Enum):
    """Order sides"""
    BUY = "Buy"
    SELL = "Sell"


class OrderStatus(Enum):
    """Order statuses"""
    NEW = "New"
    PARTIALLY_FILLED = "PartiallyFilled"
    FILLED = "Filled"
    CANCELLED = "Cancelled"
    REJECTED = "Rejected"


class BybitTradeExecutor:
    """Handles trade execution and order management using Bybit API"""
    
    def __init__(self, config: Dict):
        """
        Initialize BybitTradeExecutor with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.symbol = config['trading']['symbol']
        self.category = config['trading']['category']
        self.position_size = config['trading']['position_size']
        self.max_positions = config['trading']['max_positions']
        self.leverage = config['trading']['leverage']
        self.margin_mode = config['trading']['margin_mode']
        
        # Initialize Bybit connection
        self._init_bybit_connection()
        
        # Track positions and orders
        self.positions = {}
        self.orders = {}
        self.order_counter = 0
        
        # Set leverage and margin mode
        self._set_leverage_and_margin()
        
    def _init_bybit_connection(self):
        """Initialize Bybit HTTP connection"""
        try:
            api_config = self.config['api']
            
            self.session = HTTP(
                api_key=api_config['api_key'],
                api_secret=api_config['api_secret'],
                testnet=api_config['testnet']
            )
            
            logger.info("Bybit executor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Bybit connection: {e}")
            raise
    
    def _set_leverage_and_margin(self):
        """Set leverage and margin mode for the symbol"""
        try:
            # Set leverage
            leverage_response = self.session.set_leverage(
                category=self.category,
                symbol=self.symbol,
                buyLeverage=str(self.leverage),
                sellLeverage=str(self.leverage)
            )
            
            if leverage_response['retCode'] != 0:
                logger.warning(f"Failed to set leverage: {leverage_response['retMsg']}")
            
            # Set margin mode
            margin_response = self.session.switch_margin_mode(
                category=self.category,
                symbol=self.symbol,
                tradeMode=self.margin_mode
            )
            
            if margin_response['retCode'] != 0:
                logger.warning(f"Failed to set margin mode: {margin_response['retMsg']}")
            
            logger.info(f"Set leverage to {self.leverage}x and margin mode to {self.margin_mode}")
            
        except Exception as e:
            logger.error(f"Error setting leverage and margin: {e}")
    
    def place_order(self, side: OrderSide, order_type: OrderType, 
                   quantity: float, price: Optional[float] = None,
                   stop_price: Optional[float] = None,
                   time_in_force: str = "GTC") -> Dict:
        """
        Place a new order on Bybit
        
        Args:
            side: Order side (Buy/Sell)
            order_type: Order type
            quantity: Order quantity
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Time in force (GTC, IOC, FOK)
            
        Returns:
            Order information dictionary
        """
        try:
            self.order_counter += 1
            order_id = f"order_{self.order_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Validate order parameters
            if not self._validate_order(side, order_type, quantity, price, stop_price):
                raise ValueError("Invalid order parameters")
            
            # Check position limits
            if not self._check_position_limits(side):
                raise ValueError("Position limit exceeded")
            
            # Prepare order parameters
            order_params = {
                'category': self.category,
                'symbol': self.symbol,
                'side': side.value,
                'orderType': order_type.value,
                'qty': str(quantity),
                'timeInForce': time_in_force
            }
            
            # Add price for limit orders
            if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and price:
                order_params['price'] = str(price)
            
            # Add stop price for stop orders
            if order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and stop_price:
                order_params['stopPrice'] = str(stop_price)
            
            # Place order
            response = self.session.place_order(**order_params)
            
            if response['retCode'] != 0:
                raise Exception(f"Bybit API error: {response['retMsg']}")
            
            # Create order record
            order = {
                'id': order_id,
                'bybit_order_id': response['result']['orderId'],
                'symbol': self.symbol,
                'side': side.value,
                'type': order_type.value,
                'quantity': quantity,
                'price': price,
                'stop_price': stop_price,
                'status': OrderStatus.NEW.value,
                'timestamp': datetime.now(),
                'filled_quantity': 0,
                'average_price': None,
                'response': response
            }
            
            # Store order
            self.orders[order_id] = order
            
            logger.info(f"Order placed: {order_id} - {side.value} {quantity} {self.symbol}")
            return order
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise
    
    def _validate_order(self, side: OrderSide, order_type: OrderType, 
                       quantity: float, price: Optional[float], 
                       stop_price: Optional[float]) -> bool:
        """Validate order parameters"""
        try:
            # Check quantity
            if quantity <= 0:
                logger.error("Invalid quantity")
                return False
            
            # Check price for limit orders
            if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and price is None:
                logger.error("Price required for limit orders")
                return False
            
            # Check stop price for stop orders
            if order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and stop_price is None:
                logger.error("Stop price required for stop orders")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating order: {e}")
            return False
    
    def _check_position_limits(self, side: OrderSide) -> bool:
        """Check if position limits allow new orders"""
        try:
            # Get current positions
            positions = self.get_positions()
            
            # Count active positions
            active_positions = len([p for p in positions if float(p.get('size', 0)) != 0])
            
            if active_positions >= self.max_positions:
                logger.warning("Maximum positions limit reached")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking position limits: {e}")
            return False
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if order was cancelled successfully
        """
        try:
            if order_id not in self.orders:
                logger.error(f"Order not found: {order_id}")
                return False
            
            order = self.orders[order_id]
            bybit_order_id = order.get('bybit_order_id')
            
            if not bybit_order_id:
                logger.error(f"No Bybit order ID found for {order_id}")
                return False
            
            # Cancel order on Bybit
            response = self.session.cancel_order(
                category=self.category,
                symbol=self.symbol,
                orderId=bybit_order_id
            )
            
            if response['retCode'] != 0:
                logger.error(f"Failed to cancel order: {response['retMsg']}")
                return False
            
            # Update order status
            order['status'] = OrderStatus.CANCELLED.value
            order['cancel_timestamp'] = datetime.now()
            
            logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    def get_positions(self) -> List[Dict]:
        """Get current positions from Bybit"""
        try:
            response = self.session.get_positions(
                category=self.category,
                symbol=self.symbol
            )
            
            if response['retCode'] != 0:
                raise Exception(f"Bybit API error: {response['retMsg']}")
            
            positions = response['result']['list']
            
            # Update local positions cache
            for position in positions:
                symbol = position['symbol']
                self.positions[symbol] = {
                    'symbol': symbol,
                    'side': position['side'],
                    'size': float(position['size']),
                    'entry_price': float(position['avgPrice']),
                    'mark_price': float(position['markPrice']),
                    'unrealized_pnl': float(position['unrealisedPnl']),
                    'leverage': position['leverage'],
                    'margin': float(position['positionBalance']),
                    'timestamp': datetime.now()
                }
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Dict]:
        """
        Get orders with optional status filter
        
        Args:
            status: Optional order status filter
            
        Returns:
            List of orders
        """
        try:
            response = self.session.get_open_orders(
                category=self.category,
                symbol=self.symbol
            )
            
            if response['retCode'] != 0:
                raise Exception(f"Bybit API error: {response['retMsg']}")
            
            orders = response['result']['list']
            
            # Filter by status if specified
            if status:
                orders = [o for o in orders if o['orderStatus'] == status.value]
            
            return orders
            
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []
    
    def get_account_balance(self) -> Dict:
        """Get account balance from Bybit"""
        try:
            response = self.session.get_wallet_balance(
                accountType="UNIFIED"
            )
            
            if response['retCode'] != 0:
                raise Exception(f"Bybit API error: {response['retMsg']}")
            
            balance_info = response['result']['list'][0]
            
            return {
                'total_balance': float(balance_info['totalWalletBalance']),
                'available_balance': float(balance_info['availableToWithdraw']),
                'used_balance': float(balance_info['totalWalletBalance']) - float(balance_info['availableToWithdraw']),
                'currency': 'USDT',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return {}
    
    def place_stop_loss_order(self, side: OrderSide, quantity: float, 
                             stop_price: float) -> Dict:
        """
        Place stop loss order
        
        Args:
            side: Order side (opposite to position side)
            quantity: Order quantity
            stop_price: Stop price
            
        Returns:
            Order information dictionary
        """
        try:
            return self.place_order(
                side=side,
                order_type=OrderType.STOP,
                quantity=quantity,
                stop_price=stop_price,
                time_in_force="GTC"
            )
            
        except Exception as e:
            logger.error(f"Error placing stop loss order: {e}")
            raise
    
    def place_take_profit_order(self, side: OrderSide, quantity: float, 
                              price: float) -> Dict:
        """
        Place take profit order
        
        Args:
            side: Order side (opposite to position side)
            quantity: Order quantity
            price: Take profit price
            
        Returns:
            Order information dictionary
        """
        try:
            return self.place_order(
                side=side,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=price,
                time_in_force="GTC"
            )
            
        except Exception as e:
            logger.error(f"Error placing take profit order: {e}")
            raise
    
    def close_position(self, side: OrderSide, quantity: float) -> Dict:
        """
        Close position with market order
        
        Args:
            side: Order side (opposite to position side)
            quantity: Quantity to close
            
        Returns:
            Order information dictionary
        """
        try:
            return self.place_order(
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity,
                time_in_force="IOC"
            )
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            raise
    
    def get_position_info(self, symbol: str) -> Optional[Dict]:
        """Get specific position information"""
        try:
            positions = self.get_positions()
            
            for position in positions:
                if position['symbol'] == symbol:
                    return {
                        'symbol': position['symbol'],
                        'side': position['side'],
                        'size': float(position['size']),
                        'entry_price': float(position['avgPrice']),
                        'mark_price': float(position['markPrice']),
                        'unrealized_pnl': float(position['unrealisedPnl']),
                        'leverage': position['leverage'],
                        'margin': float(position['positionBalance']),
                        'timestamp': datetime.now()
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting position info: {e}")
            return None
    
    def update_order_status(self, order_id: str) -> bool:
        """Update order status from Bybit"""
        try:
            if order_id not in self.orders:
                return False
            
            order = self.orders[order_id]
            bybit_order_id = order.get('bybit_order_id')
            
            if not bybit_order_id:
                return False
            
            # Get order status from Bybit
            response = self.session.get_open_orders(
                category=self.category,
                symbol=self.symbol,
                orderId=bybit_order_id
            )
            
            if response['retCode'] != 0:
                return False
            
            orders = response['result']['list']
            if not orders:
                # Order not found in open orders, might be filled or cancelled
                order['status'] = OrderStatus.FILLED.value
                return True
            
            bybit_order = orders[0]
            order['status'] = bybit_order['orderStatus']
            order['filled_quantity'] = float(bybit_order['cumExecQty'])
            order['average_price'] = float(bybit_order['avgPrice']) if bybit_order['avgPrice'] else None
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating order status: {e}")
            return False
    
    def cancel_all_orders(self) -> int:
        """Cancel all open orders"""
        try:
            response = self.session.cancel_all_orders(
                category=self.category,
                symbol=self.symbol
            )
            
            if response['retCode'] != 0:
                logger.error(f"Failed to cancel all orders: {response['retMsg']}")
                return 0
            
            cancelled_count = len(response['result']['list'])
            logger.info(f"Cancelled {cancelled_count} orders")
            
            # Update local order cache
            for order_id, order in self.orders.items():
                if order['status'] == OrderStatus.NEW.value:
                    order['status'] = OrderStatus.CANCELLED.value
                    order['cancel_timestamp'] = datetime.now()
            
            return cancelled_count
            
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            return 0
