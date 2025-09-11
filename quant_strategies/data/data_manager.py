import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from .yfinance_client import YFinanceClient

class DataManager:
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataManager for orchestrating data collection.
        
        Args:
            data_dir: Directory to store all data files
        """
        self.data_dir = data_dir
        self.raw_data_dir = os.path.join(data_dir, "raw")
        self.processed_data_dir = os.path.join(data_dir, "processed")
        self.metadata_dir = os.path.join(data_dir, "metadata")
        
        # Create directories if they don't exist
        for directory in [self.raw_data_dir, self.processed_data_dir, self.metadata_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self.client = YFinanceClient()
    
    def get_ticker_data(self, ticker: str, start_date: str, 
                       output_dir: Optional[str] = None) -> str:
        """
        Fetch and save data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in 'YYYY-MM-DD' format
            output_dir: Directory to save data (defaults to raw_data_dir)
        
        Returns:
            Path to the saved CSV file
        """
        if output_dir is None:
            output_dir = self.raw_data_dir
        
        filename = f"{ticker}_{start_date}_to_today.csv"
        return self.client.save_daily_prices_from_start(
            ticker, start_date, output_dir, filename
        )
    
    def get_ticker_data_range(self, ticker: str, start_date: str, end_date: str,
                             output_dir: Optional[str] = None) -> str:
        """
        Fetch and save data for a single ticker with specific date range.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            output_dir: Directory to save data (defaults to raw_data_dir)
        
        Returns:
            Path to the saved CSV file
        """
        if output_dir is None:
            output_dir = self.raw_data_dir
        
        filename = f"{ticker}_{start_date}_to_{end_date}.csv"
        return self.client.save_daily_prices(
            ticker, start_date, end_date, output_dir, filename
        )
    
    def get_multiple_tickers(self, tickers: List[str], start_date: str) -> Dict[str, str]:
        """
        Fetch and save data for multiple tickers.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
        
        Returns:
            Dictionary mapping ticker to file path
        """
        results = {}
        
        print(f"Fetching data for {len(tickers)} tickers from {start_date} to today...")
        
        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{len(tickers)}] Processing {ticker}...")
            try:
                file_path = self.get_ticker_data(ticker, start_date)
                results[ticker] = file_path
                print(f"✓ {ticker} data saved successfully")
            except Exception as e:
                print(f"✗ Error processing {ticker}: {str(e)}")
                results[ticker] = None
        
        # Save metadata
        self._save_metadata(tickers, start_date, results)
        
        return results
    
    def get_multiple_tickers_range(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, str]:
        """
        Fetch and save data for multiple tickers with specific date range.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        
        Returns:
            Dictionary mapping ticker to file path
        """
        results = {}
        
        print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}...")
        
        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{len(tickers)}] Processing {ticker}...")
            try:
                file_path = self.get_ticker_data_range(ticker, start_date, end_date)
                results[ticker] = file_path
                print(f"✓ {ticker} data saved successfully")
            except Exception as e:
                print(f"✗ Error processing {ticker}: {str(e)}")
                results[ticker] = None
        
        # Save metadata
        self._save_metadata(tickers, start_date, results, end_date)
        
        return results
    
    def _save_metadata(self, tickers: List[str], start_date: str, results: Dict[str, str], 
                      end_date: Optional[str] = None):
        """Save metadata about the data collection process."""
        metadata = {
            "collection_date": datetime.now().isoformat(),
            "start_date": start_date,
            "end_date": end_date or datetime.now().strftime('%Y-%m-%d'),
            "tickers": tickers,
            "results": results,
            "successful_tickers": [t for t, path in results.items() if path is not None],
            "failed_tickers": [t for t, path in results.items() if path is None]
        }
        
        metadata_file = os.path.join(
            self.metadata_dir, 
            f"data_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nMetadata saved to: {metadata_file}")
        print(f"Successful: {len(metadata['successful_tickers'])} tickers")
        print(f"Failed: {len(metadata['failed_tickers'])} tickers")
    
    def get_5year_data(self, tickers: List[str]) -> Dict[str, str]:
        """
        Convenience method to get 5 years of data for multiple tickers.
        Uses yesterday as the end date to avoid issues with today's data.
        
        Args:
            tickers: List of stock ticker symbols
        
        Returns:
            Dictionary mapping ticker to file path
        """
        end_date = datetime.now() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=5*365)  # 5 years ago
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        return self.get_multiple_tickers_range(tickers, start_str, end_str) 
        