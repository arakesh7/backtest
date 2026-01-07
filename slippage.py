from abc import abstractmethod

class Slippage:
    """
    Abstract base class for slippage models."""
    @abstractmethod
    def calculate_slippage(self, order, price) -> float:
        """
        Apply slippage to the given price.

        Args:
            price (float): The original price before slippage.

        Returns:
            float: The price after applying slippage.
        """
        raise NotImplementedError("Subclasses must implement the apply_slippage method.")
    

class DefaultSlippage(Slippage):

    def calculate_slippage(self, order, price) -> float:
       """
        Default slippage model that does not modify the price.
       """
       return price