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
        
        # Handle MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten MultiIndex columns by taking the first level (OHLCV names)
            df.columns = df.columns.get_level_values(0)
        
        # Reset index to bring date into a column
        df = df.reset_index()
        
        # Normalize column names to lowercase for consistency
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        # Ensure we have a 'date' column, regardless of how yfinance names it
        if 'date' not in df.columns:
            # Common cases: 'index' or unknown first column that is datetime-like
            if 'index' in df.columns:
                df = df.rename(columns={'index': 'date'})
            else:
                # If first column looks like a datetime, rename it to 'date'
                first_col = df.columns[0]
                if pd.api.types.is_datetime64_any_dtype(df[first_col]):
                    df = df.rename(columns={first_col: 'date'})
                else:
                    # As a fallback, try 'datetime' or 'timestamp'
                    for cand in ('datetime', 'timestamp', 'Date'):
                        if cand.lower() in df.columns:
                            df = df.rename(columns={cand.lower(): 'date'})
                            break
        
        # Map adjusted close naming - handle both 'adj close' and 'adj_close'
        rename_map = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'adj close': 'adj_close',
            'adj_close': 'adj_close',
            'volume': 'volume',
        }
        # Apply mapping where keys exist
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns and k != v})
        
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
        if 'date' in df.columns:
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
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
        if 'date' in df.columns:
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        return output_path 