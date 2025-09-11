# Example: Fetch 5 years of daily price data for AAPL using YFinanceClient
from quant_strategies.data import YFinanceClient
from datetime import datetime, timedelta
import os

# Set ticker and date range
TICKER = "AAPL"
END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=5*365)
START_STR = START_DATE.strftime('%Y-%m-%d')
END_STR = END_DATE.strftime('%Y-%m-%d')

# Fetch the data
yf_client = YFinanceClient()
df = yf_client.get_daily_prices(TICKER, START_STR, END_STR)

# Display the first few rows
print(f"Fetched {len(df)} rows for {TICKER}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Columns: {list(df.columns)}")
df.head()

# Save the data to CSV
output_file = f"notebooks/{TICKER}_5year_data.csv"
df.to_csv(output_file, index=False)
print(f"\nData saved to: {output_file}")
print(f"File size: {os.path.getsize(output_file) / 1024:.1f} KB") 