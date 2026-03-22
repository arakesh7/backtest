from slippage import DefaultSlippage
from abc import abstractmethod, ABC
from typing import Tuple


class ExecutionModel(ABC):
    """
    Abstract base class for execution models using the Template Method Pattern.

    An execution model determines if an order should be executed based on the
    current data bar, and at what price. It can also incorporate a slippage model.
    """
    def __init__(self, slippage_model=None):
        self.slippage_model = slippage_model if slippage_model else DefaultSlippage()

    @abstractmethod
    def execute(self, price, order, data_bar: tuple) -> Tuple[bool, float]:
      raise NotImplementedError("Subclasses must implement the execute method.")


class SimpleExecutionModel(ExecutionModel):
    """
    A simple execution model that fills the entire order immediately 
    at the given price adjusted for slippage.
    """
    def execute(self, price, order, data_bar):
        fill_price = self.slippage_model.calculate_slippage(order, price)
        return fill_price, order.size
    

class PercentVolumeExecutionModel(ExecutionModel):
    def __init__(self, volume_pct=0.1, slippage_model=None):
        super().__init__(slippage_model)
        if not (0 < volume_pct <=1 ):
            raise ValueError("volume_pct must be between 0 and 1")
        self.volume_pct = volume_pct


    def execute(self, price, order, data_bar):
        _, _, _, _, _, v = data_bar
        max_volume = int(v * self.volume_pct)

        if max_volume <= 0:
            return None, 0
        
        exec_size = min(order.remaining_size, max_volume)
        fill_price = self.slippage_model.calculate_slippage(order, price)  # Use close price with slippage
        return fill_price, exec_size
 

        
        
      
      
      
