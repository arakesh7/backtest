from abc import abstractmethod, ABC


class Slippage(ABC):
    """
    Abstract base class for slippage models.
    """
    @abstractmethod
    def calculate_slippage(self, order, price: float) -> float:
        """
        Apply slippage to the given price.

        Args:
            order: The Order being executed.
            price (float): The original price before slippage.

        Returns:
            float: The price after applying slippage.
        """
        raise NotImplementedError("Subclasses must implement the calculate_slippage method.")


class DefaultSlippage(Slippage):
    def calculate_slippage(self, order, price: float) -> float:
        """
        Default slippage model that does not modify the price.
        """
        return price
