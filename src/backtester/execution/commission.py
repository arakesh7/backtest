from abc import ABC, abstractmethod


class Commission(ABC):
    """Abstract base class for commission and transaction fee models."""

    def __init__(self, rate: float = 0.0):
        self.rate = float(rate)

    @abstractmethod
    def calculate(self, size: int, price: float) -> float:
        """Calculate commission fee for a given order fill size and price."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_commission(self, size: int, price: float) -> float:
        return self.calculate(size, price)


class NoCommission(Commission):
    """Zero commission model."""

    def __init__(self):
        super().__init__(rate=0.0)

    def calculate(self, size: int, price: float) -> float:
        return 0.0

    def get_commission(self, position_or_size=None, price=None) -> float:
        return 0.0


class FlatCommission(Commission):
    """Fixed fee per executed order (e.g. ₹20 or $1 per trade)."""

    def __init__(self, fee: float = 20.0):
        super().__init__(rate=0.0)
        self.fee = float(fee)

    def calculate(self, size: int, price: float) -> float:
        return self.fee


class PercentageCommission(Commission):
    """Percentage of turnover commission model (e.g. 0.0003 for 0.03%)."""

    def __init__(self, rate: float = 0.0003, min_fee: float = 0.0):
        super().__init__(rate=rate)
        self.min_fee = float(min_fee)

    def calculate(self, size: int, price: float) -> float:
        turnover = abs(size) * float(price)
        return max(self.min_fee, turnover * self.rate)
