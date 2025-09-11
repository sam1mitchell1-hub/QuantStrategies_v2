import pandas as pd
import yfinance as yf
from datetime import datetime
import os

class YFinanceClient:
    def __init__(self):
        pass  # No API key required

    def get_daily_prices(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch daily price data for a given ticker and date range using yfinance.
        Dates should be in 'YYYY-MM-DD' format.
        Returns a pandas DataFrame with the results.
        """
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            return df
        df = df.reset_index()
        df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        }, inplace=True)
        return df

    def get_daily_prices_from_start(self, ticker: str, start_date: str) -> pd.DataFrame:
        """
        Fetch daily price data from a start date until today.
        Start date should be in 'YYYY-MM-DD' format.
        Returns a pandas DataFrame with the results.
        """
        end_date = datetime.today().strftime('%Y-%m-%d')
        return self.get_daily_prices(ticker, start_date, end_date)

    def save_daily_prices(self, ticker: str, start_date: str, end_date: str, 
                         output_dir: str = "notebooks", filename: str = None) -> str:
        """
        Fetch daily price data and save it to a CSV file.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            output_dir: Directory to save the CSV file (default: 'notebooks')
            filename: Custom filename (default: '{ticker}_{start_date}_to_{end_date}.csv')
        
        Returns:
            Path to the saved CSV file
        """
        # Fetch the data
        df = self.get_daily_prices(ticker, start_date, end_date)
        
        if df.empty:
            raise ValueError(f"No data found for {ticker} between {start_date} and {end_date}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            filename = f"{ticker}_{start_date}_to_{end_date}.csv"
        
        # Ensure filename has .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        # Full path to output file
        output_path = os.path.join(output_dir, filename)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        print(f"Saved {len(df)} rows of {ticker} data to: {output_path}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")
        
        return output_path

    def save_daily_prices_from_start(self, ticker: str, start_date: str, 
                                   output_dir: str = "notebooks", filename: str = None) -> str:
        """
        Fetch daily price data from a start date until today and save it to a CSV file.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in 'YYYY-MM-DD' format
            output_dir: Directory to save the CSV file (default: 'notebooks')
            filename: Custom filename (default: '{ticker}_{start_date}_to_today.csv')
        
        Returns:
            Path to the saved CSV file
        """
        end_date = datetime.today().strftime('%Y-%m-%d')
        
        # Fetch the data
        df = self.get_daily_prices(ticker, start_date, end_date)
        
        if df.empty:
            raise ValueError(f"No data found for {ticker} from {start_date} to today")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            filename = f"{ticker}_{start_date}_to_today.csv"
        
        # Ensure filename has .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        # Full path to output file
        output_path = os.path.join(output_dir, filename)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        print(f"Saved {len(df)} rows of {ticker} data to: {output_path}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")
        
        return output_path 