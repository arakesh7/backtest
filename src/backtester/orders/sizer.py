from abc import ABC, abstractmethod


class Sizer(ABC):
    """Abstract base class for position sizers."""

    @abstractmethod
    def get_size(
        self,
        broker,
        asset: str,
        entry_price: float,
        stop_loss_price: float,
        side: str,
        **kwargs,
    ) -> int:
        """Abstract method to calculate position size."""
        raise NotImplementedError()


class FixedRiskPercentSizer(Sizer):
    """
    Calculates position size based on a fixed percentage of total portfolio equity
    risked per trade based on the stop-loss distance:
        size = (equity * (risk_percent / 100)) / |entry_price - stop_loss_price|

    If stop_loss_price is not provided or equals entry_price, it falls back to
    allocating a fixed percentage of equity to the position:
        size = (equity * (risk_percent / 100)) / entry_price
    """

    def __init__(self, risk_percent: float = 1.0, max_allocation_pct: float = None):
        """
        Initializes the sizer with a fixed risk percentage.
        :param risk_percent: The percentage of portfolio equity to risk (e.g., 1.0 for 1%).
        :param max_allocation_pct: Optional cap on maximum capital allocated to a single position
                                   as a percentage of equity (e.g. 50.0 for 50%).
        """
        if not 0 < risk_percent <= 100:
            raise ValueError("risk_percent must be between 0 and 100.")
        self.risk_percent = risk_percent
        self.max_allocation_pct = max_allocation_pct

    def get_size(
        self,
        broker,
        asset: str,
        entry_price: float,
        stop_loss_price: float = None,
        side: str = "BUY",
        **kwargs,
    ) -> int:
        if entry_price <= 0:
            return 0

        # Base capital from portfolio equity (fallback to cash if flat or equity <= 0)
        equity = getattr(broker, "equity", None)
        if equity is None or equity <= 0:
            equity = broker.cash

        if equity <= 0:
            return 0

        capital_to_risk = equity * (self.risk_percent / 100.0)

        # Risk-based sizing when stop_loss_price is provided
        if stop_loss_price is not None and stop_loss_price > 0:
            risk_per_share = abs(entry_price - stop_loss_price)
            if risk_per_share > 0:
                raw_size = int(capital_to_risk / risk_per_share)
            else:
                raw_size = int(capital_to_risk / entry_price)
        else:
            raw_size = int(capital_to_risk / entry_price)

        # Apply optional single-position maximum allocation cap
        if self.max_allocation_pct is not None and self.max_allocation_pct > 0:
            max_capital = equity * (self.max_allocation_pct / 100.0)
            max_size_by_allocation = int(max_capital / entry_price)
            raw_size = min(raw_size, max_size_by_allocation)

        # Safety cap: cannot exceed total affordable shares with available purchasing power
        if hasattr(broker, "get_available_cash"):
            available_funds = broker.get_available_cash()
        else:
            available_funds = broker.cash

        if available_funds > 0:
            margin_ratio = getattr(broker, "margin_ratio", 1.0)
            cost_per_share = entry_price if side == "BUY" else (entry_price * margin_ratio)
            if cost_per_share > 0:
                max_affordable = int(available_funds / cost_per_share)
                raw_size = min(raw_size, max_affordable)

        return max(0, raw_size)


class FixedSizeSizer(Sizer):
    """
    Returns a fixed, pre-defined position size for every trade.
    """

    def __init__(self, default_size: int = 1):
        self.default_size = default_size

    def get_size(
        self,
        broker,
        asset: str,
        entry_price: float,
        stop_loss_price: float = None,
        side: str = "BUY",
        **kwargs,
    ) -> int:
        return self.default_size


class AllInSizer(Sizer):
    """
    Calculates position size to use all available purchasing power for a trade.
    """

    def get_size(
        self,
        broker,
        asset: str,
        entry_price: float,
        stop_loss_price: float = None,
        side: str = "BUY",
        **kwargs,
    ) -> int:
        if entry_price <= 0:
            return 0
        available_funds = (
            broker.get_available_cash()
            if hasattr(broker, "get_available_cash")
            else broker.cash
        )
        if available_funds <= 0:
            return 0
        margin_ratio = getattr(broker, "margin_ratio", 1.0)
        cost_per_share = entry_price if side == "BUY" else (entry_price * margin_ratio)
        return int(available_funds / cost_per_share) if cost_per_share > 0 else 0
