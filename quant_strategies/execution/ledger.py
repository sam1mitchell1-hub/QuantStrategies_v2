"""
Order Ledger for OMS

Provides append-only audit trail for all trading activity:
- Signals generated
- Orders placed
- Fills received
- Position changes

Uses SQLite for persistence with full history tracking.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from .models import Signal, Order, Fill, OrderStatus, SignalType, OrderSide, OrderType

logger = logging.getLogger(__name__)


class Ledger:
    """
    Append-only ledger for tracking all trading activity.
    
    All operations are INSERT only - no UPDATEs or DELETEs.
    Status changes are recorded as new rows.
    """
    
    def __init__(self, db_path: str = "data/ledger.db"):
        """
        Initialize ledger with SQLite database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Create directory if it doesn't exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Ledger initialized at {db_path}")
    
    def _init_database(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                signal_id TEXT PRIMARY KEY,
                strategy_name TEXT NOT NULL,
                ticker TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                strength REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        
        # Orders table (append-only, status changes create new rows)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT NOT NULL,
                signal_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                order_type TEXT NOT NULL,
                status TEXT NOT NULL,
                limit_price REAL,
                submitted_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                filled_quantity INTEGER DEFAULT 0,
                FOREIGN KEY(signal_id) REFERENCES signals(signal_id)
            )
        """)
        
        # Create index for efficient order lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_orders_order_id 
            ON orders(order_id, updated_at DESC)
        """)
        
        # Fills table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fills (
                fill_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                fees REAL NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY(order_id) REFERENCES orders(order_id)
            )
        """)
        
        # Positions history table (tracks all position changes)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS position_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                avg_cost REAL NOT NULL,
                market_value REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
        
        logger.debug("Database tables initialized")
    
    def record_signal(self, signal: Signal):
        """
        Record a trading signal.
        
        Args:
            signal: Signal object to record
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO signals (signal_id, strategy_name, ticker, signal_type, strength, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                signal.signal_id,
                signal.strategy_name,
                signal.ticker,
                signal.signal_type.value,
                signal.strength,
                signal.timestamp.isoformat()
            ))
            conn.commit()
            logger.info(f"Recorded signal: {signal.signal_id} - {signal.ticker} {signal.signal_type.value}")
        except sqlite3.IntegrityError as e:
            logger.warning(f"Signal {signal.signal_id} already exists: {e}")
        finally:
            conn.close()
    
    def record_order(self, order: Order):
        """
        Record an order (or order status change).
        
        This is append-only - each status change creates a new row.
        
        Args:
            order: Order object to record
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO orders 
                (order_id, signal_id, ticker, side, quantity, order_type, status, 
                 limit_price, submitted_at, updated_at, filled_quantity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order.order_id,
                order.signal_id,
                order.ticker,
                order.side.value,
                order.quantity,
                order.order_type.value,
                order.status.value,
                order.limit_price,
                order.submitted_at.isoformat() if order.submitted_at else None,
                order.updated_at.isoformat() if order.updated_at else None,
                order.filled_quantity
            ))
            conn.commit()
            logger.info(f"Recorded order: {order.order_id} - {order.ticker} {order.side.value} {order.quantity} @ {order.status.value}")
        finally:
            conn.close()
    
    def update_order_status(self, order_id: str, status: OrderStatus, 
                          filled_quantity: int = 0):
        """
        Update order status by inserting a new row.
        
        Args:
            order_id: Order ID to update
            status: New status
            filled_quantity: Total filled quantity
        """
        # Get the latest order record
        latest = self.get_latest_order_status(order_id)
        if not latest:
            logger.error(f"Cannot update status for unknown order: {order_id}")
            return
        
        # Create updated order record
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO orders 
                (order_id, signal_id, ticker, side, quantity, order_type, status, 
                 limit_price, submitted_at, updated_at, filled_quantity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order_id,
                latest['signal_id'],
                latest['ticker'],
                latest['side'],
                latest['quantity'],
                latest['order_type'],
                status.value,
                latest['limit_price'],
                latest['submitted_at'],
                datetime.now().isoformat(),
                filled_quantity
            ))
            conn.commit()
            logger.info(f"Updated order {order_id} status to {status.value}")
        finally:
            conn.close()
    
    def record_fill(self, fill: Fill):
        """
        Record an order fill.
        
        Args:
            fill: Fill object to record
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO fills (fill_id, order_id, quantity, price, fees, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                fill.fill_id,
                fill.order_id,
                fill.quantity,
                fill.price,
                fill.fees,
                fill.timestamp.isoformat()
            ))
            conn.commit()
            logger.info(f"Recorded fill: {fill.fill_id} - {fill.quantity} @ ${fill.price:.2f}")
        except sqlite3.IntegrityError as e:
            logger.warning(f"Fill {fill.fill_id} already exists: {e}")
        finally:
            conn.close()
    
    def record_position_snapshot(self, ticker: str, quantity: int, avg_cost: float,
                                market_value: float, unrealized_pnl: float):
        """
        Record a position snapshot.
        
        Args:
            ticker: Stock ticker
            quantity: Position size
            avg_cost: Average cost per share
            market_value: Current market value
            unrealized_pnl: Unrealized P&L
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO position_history 
                (ticker, quantity, avg_cost, market_value, unrealized_pnl, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                quantity,
                avg_cost,
                market_value,
                unrealized_pnl,
                datetime.now().isoformat()
            ))
            conn.commit()
            logger.debug(f"Recorded position snapshot: {ticker} x{quantity}")
        finally:
            conn.close()
    
    def get_order_history(self, order_id: str) -> List[Dict[str, Any]]:
        """
        Get complete history for an order.
        
        Args:
            order_id: Order ID to query
            
        Returns:
            List of order records in chronological order
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT * FROM orders 
                WHERE order_id = ?
                ORDER BY updated_at ASC
            """, (order_id,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
    
    def get_latest_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest status for an order.
        
        Args:
            order_id: Order ID to query
            
        Returns:
            Latest order record or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT * FROM orders 
                WHERE order_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
            """, (order_id,))
            
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
    
    def get_fills_for_order(self, order_id: str) -> List[Dict[str, Any]]:
        """
        Get all fills for an order.
        
        Args:
            order_id: Order ID to query
            
        Returns:
            List of fill records
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT * FROM fills 
                WHERE order_id = ?
                ORDER BY timestamp ASC
            """, (order_id,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
    
    def get_signal_by_id(self, signal_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a signal by ID.
        
        Args:
            signal_id: Signal ID to query
            
        Returns:
            Signal record or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT * FROM signals 
                WHERE signal_id = ?
            """, (signal_id,))
            
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
    
    def get_all_signals(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent signals.
        
        Args:
            limit: Maximum number of signals to return
            
        Returns:
            List of signal records
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT * FROM signals 
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
    
    def get_position_history(self, ticker: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get position history for a ticker.
        
        Args:
            ticker: Stock ticker
            limit: Maximum number of records to return
            
        Returns:
            List of position history records
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT * FROM position_history 
                WHERE ticker = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (ticker, limit))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

