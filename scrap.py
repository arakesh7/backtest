
import pandas as pd

business_days = pd.bdate_range(start="2024-01-01", end="2024-01-05")
start_time = "09:15"
end_time = "15:30"

bars = []
for day in business_days:
    intraday_times = pd.date_range(
        f"{day.date()} {start_time}",
        f"{day.date()} {end_time}",
        freq="1min"
    )
    
    bars.extend(intraday_times)

print(bars)

    