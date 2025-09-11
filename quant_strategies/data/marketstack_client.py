import os
import requests
import pandas as pd
from datetime import datetime

class MarketstackClient:
    BASE_URL = "http://api.marketstack.com/v1/"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("MARKETSTACK_API_KEY")
        if not self.api_key:
            raise ValueError("Marketstack API key must be provided or set as the MARKETSTACK_API_KEY environment variable.")

    def get_daily_prices(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch daily price data for a given ticker and date range.
        Dates should be in 'YYYY-MM-DD' format.
        Returns a pandas DataFrame with the results.
        """
        endpoint = f"{self.BASE_URL}eod"
        params = {
            "access_key": self.api_key,
            "symbols": ticker,
            "date_from": start_date,
            "date_to": end_date,
            "limit": 1000  # Marketstack paginates, so we may need to loop
        }
        all_data = []
        while True:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            all_data.extend(data.get("data", []))
            # Pagination: check if there's a next page
            if "pagination" in data and data["pagination"].get("next"):
                endpoint = data["pagination"]["next"]
                params = {}  # Already included in the next URL
            else:
                break
        if not all_data:
            return pd.DataFrame()
        df = pd.DataFrame(all_data)
        # Convert date column to datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df 