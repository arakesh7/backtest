import pandas as pd


class TradingCalendar:
    def get_trading_days(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
        pass


class WorkingDayTradingCalendar(TradingCalendar):
    """
    WorkingDayTradingCalendar generates trading days for a given date range, excluding weekends and holidays.
    """

    market_open_time = "09:15:00"
    market_close_time = "15:30:00"

    # def _get_1min_bars(self, start_date: str, end_date: str):
    #     business_days = pd.bdate_range(start=start_date, end=end_date)
    #     bars = []
    #     for day in business_days:
    #         day_bars = pd.date_range(
    #             f"{day.date()} {self.market_open_time}",
    #             f"{day.date()} {self.market_close_time}",
    #             freq="1min",
    #         )
    #         bars.extend(day_bars)
    #     print(type(bars[0]), ": ", bars[0])
    #     return bars

    def _get_1min_bars(self, trading_days):
        day_bars = [
            pd.date_range(
                f"{day.date()} {self.market_open_time}",
                f"{day.date()} {self.market_close_time}",
                freq="1min",
            )
            for day in trading_days
        ]

        bars = []
        for bar in day_bars:
            bars.extend(bar)
        return bars

    def get_trading_bars(self, start_date, end_date, freq="D"):
        trading_days = pd.bdate_range(start=start_date, end=end_date)
        if freq == "D":
            return trading_days
        elif freq == "1min":
            return self._get_1min_bars(trading_days)
        else:
            raise ValueError(f"Unsupported frequency: {freq}")
