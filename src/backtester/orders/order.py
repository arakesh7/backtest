from dataclasses import dataclass, field
from datetime import datetime
import secrets


@dataclass
class Trade:
    asset: str
    size: int
    price: float
    side: str
    commission: float
    pnl: float 
    executed_at: str
    order_id: str 
    id: str = field(default_factory=lambda: secrets.token_hex(16))


@dataclass(slots=True)
class Order:
    size: int
    asset: str
    side: str
    price: float = field(default=0.0)  # Default for market orders
    type: str = field(default='MARKET')
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())  # Default to current timestamp
    trigger_price: float = field(default=None)  # Optional for market orders
    stop_price: float = field(default=None)  # Optional stop-loss
    status: str = field(default='CREATED')  # Default order status
    last_modified: str = field(default=None)  # Execution timestamp
    id: str = field(default_factory=lambda: secrets.token_hex(16))  # Unique ID for each order
    parent_id: str = field(default=None)  # Optional parent order ID. Used for Cover Order
    oco_group_id: str = field(default=None)  # Optional OCO group ID for linked bracket orders
    pnl: float = field(default=None)
    is_triggered: bool = field(default=False)
    remaining_size: int = field(init=False)
    filled_size: int = field(default=0, init=False)
    trades: list = field(default_factory=list, init=False)

    def __post_init__(self):
        self.remaining_size = self.size
        self._validate()

    def _validate(self):
        if self.type == 'MARKET':
            self.price = None
        elif self.type == 'LIMIT':
            if self.price is None:
                raise ValueError("Price must be specified for LIMIT orders.")
        elif self.type == 'STOPLIMIT':
            if self.price is None or self.trigger_price is None:
                raise ValueError("Price and trigger price must be specified for STOPLIMIT orders.")
        elif self.type == 'SL':
            if self.stop_price is None:
                raise ValueError("Stop price must be specified for SL orders.")
        else:
            raise ValueError(f"Invalid order type. {self.type}")

    def execute(self, price, pnl=0, dt=None):
        if self.status != 'CREATED':
            raise RuntimeError(f"Order: {self.id} cannot be executed as its status is {self.status}.")
        self.status = 'EXECUTED'
        self.last_modified = dt
        self.price = price
        self.pnl = pnl
        return self.size

    @property
    def commission(self):
        return sum([trade.commission for trade in self.trades])

    def update_status(self):
        if self.remaining_size <= 0:
            self.status = 'EXECUTED'
        else:
            self.status = 'PARTIALLY_FILLED'

    def add_fill(self, trade: Trade):
        self.trades.append(trade)
        fill_qty = abs(trade.size)  # trade.size is negative for SELL; use absolute qty
        self.filled_size += fill_qty
        self.remaining_size -= fill_qty
        self.last_modified = trade.executed_at
        self.pnl = sum(t.pnl for t in self.trades)
        self.update_status()

    def cancel(self):
        if self.status in ('EXECUTED', 'CANCELLED'):
            raise RuntimeError(f"Order: {self.id} is already {self.status} and cannot be cancelled.")
        self.status = 'CANCELLED'

    def is_buy(self):
        return self.side == 'BUY'


@dataclass
class BuyOrder(Order):
    side: str = 'BUY'


@dataclass
class SellOrder(Order):
    side: str = 'SELL'


@dataclass
class Position:
    asset: str
    size: int = field(default=0)
    avg_price: float = field(default=0)
    total_commission: float = field(default=0.0)

    @property
    def value(self):
        return self.avg_price * self.size

    def pseudo_update(self, size, price):
        """Doesn't update the actual Position object, but returns the new position
        to check whether the position should be updated or not based on the available cash.
        """
        return Position(self.asset, self.size, self.avg_price).update(size, price)

    def update(self, size, price):
        old_size = self.size
        new_size = old_size + size

        if old_size == 0:
            # Case 1: Fresh position opened from flat
            opened = abs(size)
            closed = 0
            self.avg_price = price

        elif (old_size > 0 and size > 0) or (old_size < 0 and size < 0):
            # Case 2: Adding to existing position in the same direction
            opened = abs(size)
            closed = 0
            self.avg_price = ((self.avg_price * abs(old_size)) + (price * abs(size))) / abs(new_size)

        elif (old_size > 0 and new_size >= 0) or (old_size < 0 and new_size <= 0):
            # Case 3: Partially or fully closing existing position (no flip)
            opened = 0
            closed = abs(size)
            if new_size == 0:
                self.avg_price = 0.0

        else:
            # Case 4: Position flip (e.g. Long -> Short or Short -> Long)
            closed = abs(old_size)
            opened = abs(new_size)
            self.avg_price = price

        self.size = new_size
        return opened, closed, self.size, self.avg_price
