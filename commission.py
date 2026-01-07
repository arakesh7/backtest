class Commission:
    def __init__(self, rate):
        self.rate = rate

    def get_value(self, position, price):
        return position.size * price
    
    def calculate(self, sales):
        raise NotImplementedError("This method should be implemented in subclasses.")


class NoCommission:
    def get_commission(self, position, price):
        return 0