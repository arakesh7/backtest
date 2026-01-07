from abc import ABC, abstractmethod

class Sizer(ABC):
    """Abstract base class for position sizers."""
    @abstractmethod
    def get_size(self, broker, asset: str, entry_price: float, stop_loss_price: float, side: str, **kwargs) -> int:
        """Abstract method to calculate position size."""
        raise NotImplementedError()

class FixedRiskPercentSizer(Sizer):
    """
    Calculates position size based on a fixed percentage of available cash.
    """
    def __init__(self, risk_percent: float = 1.0):
        """
        Initializes the sizer with a fixed risk percentage.
        :param risk_percent: The percentage of available cash to risk (e.g., 1.0 for 1%).
        """
        if not 0 < risk_percent <= 100:
            raise ValueError("risk_percent must be between 0 and 100.")
        self.risk_percent = risk_percent

    def get_size(self, broker, asset: str, entry_price: float, stop_loss_price: float, side: str, **kwargs) -> int:
        """
        Calculates the position size based on a fixed percentage of cash to risk.
        """
        # Determine if the stop loss is valid for the trade direction
        if side == 'BUY':
            if entry_price <= stop_loss_price: # For long, SL must be below entry
                return 0
        elif side == 'SELL':
            if entry_price >= stop_loss_price: # For short, SL must be above entry
                return 0

        cash_to_risk = broker.cash * (self.risk_percent / 100)
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share == 0:
            return 0

        return int(cash_to_risk / risk_per_share)

class FixedSizeSizer(Sizer):
    """
    Returns a fixed, pre-defined position size for every trade.
    """
    def __init__(self, default_size: int = 1):
        self.default_size = default_size

    def get_size(self, broker, asset: str, entry_price: float, stop_loss_price: float, side: str, **kwargs) -> int:
        return self.default_size

class AllInSizer(Sizer):
    """
    Calculates position size to use all available cash for a purchase.
    For sell orders, it does not apply any logic and would typically be used
    with a `close()` or `sell(size=position.size)` call in the strategy.
    """
    def get_size(self, broker, asset: str, entry_price: float, stop_loss_price: float = None, side: str = 'BUY', **kwargs) -> int:
        """
        Calculates the maximum position size possible with the available cash.
        """
        if entry_price <= 0:
            return 0
        return int(broker.cash / entry_price)