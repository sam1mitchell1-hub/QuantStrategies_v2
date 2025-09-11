# %%
from quant_strategies.data import MarketstackClient
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

# %%
# Set your ticker and date range
TICKER = "AAPL"
END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=5*365)
START_STR = START_DATE.strftime('%Y-%m-%d')
END_STR = END_DATE.strftime('%Y-%m-%d')

# %%
# Fetch the data
data_client = MarketstackClient()
df = data_client.get_daily_prices(TICKER, START_STR, END_STR)

# %%
# Plot the opening price
df = df.sort_values("date")
plt.figure(figsize=(12, 6))
plt.plot(df["date"], df["open"], label=f"{TICKER} Opening Price")
plt.xlabel("Date")
plt.ylabel("Opening Price (USD)")
plt.title(f"{TICKER} Opening Price Over Last 5 Years")
plt.legend()
plt.tight_layout()
plt.show() 