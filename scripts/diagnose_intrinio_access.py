#!/usr/bin/env python3
"""
Intrinio API Access Diagnostic Script

This script helps diagnose what data is available with your Intrinio API key
and identifies potential access issues.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntrinioDiagnostic:
    """Diagnostic tool for Intrinio API access."""
    
    def __init__(self, api_key: str):
        """Initialize with API key."""
        self.api_key = api_key
        self.base_url = "https://api-v2.intrinio.com"
        self.session = requests.Session()
        self.session.auth = (api_key, '')
    
    def test_basic_connection(self):
        """Test basic API connection."""
        print("=== Testing Basic API Connection ===")
        
        try:
            # Test with a simple endpoint
            url = f"{self.base_url}/securities"
            response = self.session.get(url, params={'page_size': 1})
            
            if response.status_code == 200:
                print("✅ Basic API connection successful")
                return True
            elif response.status_code == 401:
                print("❌ API key is invalid or expired")
                return False
            else:
                print(f"❌ Unexpected response: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def check_ftse_symbol(self):
        """Check if UKX symbol is available."""
        print("\n=== Checking FTSE 100 Symbol (UKX) ===")
        
        try:
            url = f"{self.base_url}/securities/UKX"
            response = self.session.get(url)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ UKX symbol found")
                print(f"   Name: {data.get('name', 'N/A')}")
                print(f"   Type: {data.get('type', 'N/A')}")
                print(f"   Currency: {data.get('currency', 'N/A')}")
                return True
            elif response.status_code == 404:
                print("❌ UKX symbol not found")
                return False
            elif response.status_code == 401:
                print("❌ Unauthorized - API key doesn't have access to this symbol")
                return False
            else:
                print(f"❌ Unexpected response: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error checking UKX: {e}")
            return False
    
    def check_alternative_ftse_symbols(self):
        """Check for alternative FTSE 100 symbols."""
        print("\n=== Checking Alternative FTSE 100 Symbols ===")
        
        # Common FTSE 100 symbols
        symbols = ['UKX', 'FTSE', 'FTSE100', '^FTSE', 'FTSE.UK', 'UKX.L']
        
        available_symbols = []
        
        for symbol in symbols:
            try:
                url = f"{self.base_url}/securities/{symbol}"
                response = self.session.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ {symbol}: {data.get('name', 'N/A')}")
                    available_symbols.append(symbol)
                elif response.status_code == 404:
                    print(f"❌ {symbol}: Not found")
                elif response.status_code == 401:
                    print(f"❌ {symbol}: Unauthorized")
                else:
                    print(f"⚠️  {symbol}: Status {response.status_code}")
                    
            except Exception as e:
                print(f"❌ {symbol}: Error - {e}")
        
        return available_symbols
    
    def check_data_endpoints(self):
        """Check what data endpoints are available."""
        print("\n=== Checking Data Endpoints ===")
        
        endpoints = [
            ("securities", "Basic securities info"),
            ("securities/UKX/prices", "UKX historical prices"),
            ("securities/UKX/prices/intraday", "UKX intraday prices"),
            ("options/prices", "Options data"),
            ("securities/UKX/options", "UKX options")
        ]
        
        available_endpoints = []
        
        for endpoint, description in endpoints:
            try:
                url = f"{self.base_url}/{endpoint}"
                response = self.session.get(url, params={'page_size': 1})
                
                if response.status_code == 200:
                    print(f"✅ {endpoint}: {description}")
                    available_endpoints.append(endpoint)
                elif response.status_code == 401:
                    print(f"❌ {endpoint}: Unauthorized - {description}")
                elif response.status_code == 404:
                    print(f"❌ {endpoint}: Not found - {description}")
                else:
                    print(f"⚠️  {endpoint}: Status {response.status_code} - {description}")
                    
            except Exception as e:
                print(f"❌ {endpoint}: Error - {e}")
        
        return available_endpoints
    
    def check_plan_limits(self):
        """Check API plan limits and usage."""
        print("\n=== Checking API Plan Limits ===")
        
        try:
            # Try to get account info
            url = f"{self.base_url}/account"
            response = self.session.get(url)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Account information retrieved")
                print(f"   Plan: {data.get('plan_name', 'N/A')}")
                print(f"   Monthly calls: {data.get('monthly_calls', 'N/A')}")
                print(f"   Calls used: {data.get('calls_used', 'N/A')}")
                return True
            elif response.status_code == 401:
                print("❌ Cannot access account info - API key issue")
                return False
            else:
                print(f"⚠️  Account info status: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error checking account: {e}")
            return False
    
    def suggest_alternatives(self):
        """Suggest alternative data sources."""
        print("\n=== Alternative Data Sources ===")
        
        print("If Intrinio doesn't provide FTSE 100 data with your plan:")
        print()
        print("1. **Yahoo Finance** (Free):")
        print("   - Symbol: ^FTSE")
        print("   - No API key required")
        print("   - Limited historical data")
        print()
        print("2. **Alpha Vantage** (Free tier available):")
        print("   - Symbol: FTSE")
        print("   - Requires API key")
        print("   - 5 calls per minute limit")
        print()
        print("3. **Quandl** (Paid):")
        print("   - Various FTSE 100 datasets")
        print("   - Professional data quality")
        print()
        print("4. **Interactive Brokers** (If you have account):")
        print("   - Real-time data")
        print("   - Options data available")
        print()
        print("5. **Upgrade Intrinio Plan**:")
        print("   - Contact Intrinio support")
        print("   - Request FTSE 100 data access")
    
    def run_full_diagnostic(self):
        """Run complete diagnostic."""
        print("Intrinio API Access Diagnostic")
        print("=" * 50)
        print(f"API Key: {self.api_key[:8]}...")
        print()
        
        # Run all checks
        basic_ok = self.test_basic_connection()
        
        if not basic_ok:
            print("\n❌ Basic connection failed. Check your API key.")
            return False
        
        ukx_ok = self.check_ftse_symbol()
        alternatives = self.check_alternative_ftse_symbols()
        endpoints = self.check_data_endpoints()
        account_ok = self.check_plan_limits()
        
        # Summary
        print("\n=== DIAGNOSTIC SUMMARY ===")
        print(f"Basic connection: {'✅' if basic_ok else '❌'}")
        print(f"UKX symbol access: {'✅' if ukx_ok else '❌'}")
        print(f"Alternative symbols found: {len(alternatives)}")
        print(f"Available endpoints: {len(endpoints)}")
        print(f"Account info: {'✅' if account_ok else '❌'}")
        
        if not ukx_ok and not alternatives:
            print("\n⚠️  FTSE 100 data not available with current API key")
            self.suggest_alternatives()
        
        return ukx_ok or len(alternatives) > 0


def main():
    """Main function."""
    # Get API key from environment
    api_key = os.getenv('INTRINIO_API_KEY')
    
    if not api_key:
        print("❌ INTRINIO_API_KEY not found in environment variables")
        print("Please set your API key:")
        print("export INTRINIO_API_KEY='your_api_key_here'")
        return False
    
    # Run diagnostic
    diagnostic = IntrinioDiagnostic(api_key)
    success = diagnostic.run_full_diagnostic()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
