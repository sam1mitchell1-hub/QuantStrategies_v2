#!/usr/bin/env python3
"""
Basic Intrinio API Test

This script tests basic Intrinio API connectivity and shows what data
is actually available with your API key.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import json
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_api_key_access():
    """Test if API key is accessible and working."""
    print("=== Testing Intrinio API Key Access ===")
    
    # Get API key from environment
    api_key = os.getenv('INTRINIO_API_KEY')
    
    if not api_key:
        print("❌ INTRINIO_API_KEY not found in environment variables")
        print("Please set your API key:")
        print("export INTRINIO_API_KEY='your_api_key_here'")
        return False
    
    print(f"✅ API Key found: {api_key[:8]}...")
    
    # Test basic API connection
    base_url = "https://api-v2.intrinio.com"
    session = requests.Session()
    session.auth = (api_key, '')
    
    try:
        # Test with a simple endpoint that should work with any plan
        url = f"{base_url}/securities"
        print(f"Testing endpoint: {url}")
        
        response = session.get(url, params={'page_size': 1})
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Basic API connection successful!")
            data = response.json()
            print(f"Response data keys: {list(data.keys())}")
            return True
        elif response.status_code == 401:
            print("❌ API key is invalid or expired")
            print("Response:", response.text[:200])
            return False
        else:
            print(f"❌ Unexpected response: {response.status_code}")
            print("Response:", response.text[:200])
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


def test_available_securities():
    """Test what securities are available."""
    print("\n=== Testing Available Securities ===")
    
    api_key = os.getenv('INTRINIO_API_KEY')
    if not api_key:
        print("❌ No API key available")
        return False
    
    base_url = "https://api-v2.intrinio.com"
    session = requests.Session()
    session.auth = (api_key, '')
    
    try:
        # Get some securities
        url = f"{base_url}/securities"
        response = session.get(url, params={'page_size': 10})
        
        if response.status_code == 200:
            data = response.json()
            securities = data.get('securities', [])
            
            print(f"✅ Found {len(securities)} securities")
            print("Sample securities:")
            for i, sec in enumerate(securities[:5]):
                print(f"  {i+1}. {sec.get('ticker', 'N/A')} - {sec.get('name', 'N/A')}")
            
            return True
        else:
            print(f"❌ Failed to get securities: {response.status_code}")
            print("Response:", response.text[:200])
            return False
            
    except Exception as e:
        print(f"❌ Error getting securities: {e}")
        return False


def test_ftse_specific_access():
    """Test FTSE 100 specific access."""
    print("\n=== Testing FTSE 100 Specific Access ===")
    
    api_key = os.getenv('INTRINIO_API_KEY')
    if not api_key:
        print("❌ No API key available")
        return False
    
    base_url = "https://api-v2.intrinio.com"
    session = requests.Session()
    session.auth = (api_key, '')
    
    # Test different FTSE symbols
    ftse_symbols = ['UKX', 'FTSE', '^FTSE', 'FTSE.L', 'UKX.L']
    
    for symbol in ftse_symbols:
        try:
            print(f"Testing symbol: {symbol}")
            url = f"{base_url}/securities/{symbol}"
            response = session.get(url)
            
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ {symbol}: {data.get('name', 'N/A')}")
                print(f"     Type: {data.get('type', 'N/A')}")
                print(f"     Currency: {data.get('currency', 'N/A')}")
                return True
            elif response.status_code == 404:
                print(f"  ❌ {symbol}: Not found")
            elif response.status_code == 401:
                print(f"  ❌ {symbol}: Unauthorized")
            else:
                print(f"  ⚠️  {symbol}: Status {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ {symbol}: Error - {e}")
    
    return False


def test_data_endpoints():
    """Test what data endpoints are available."""
    print("\n=== Testing Data Endpoints ===")
    
    api_key = os.getenv('INTRINIO_API_KEY')
    if not api_key:
        print("❌ No API key available")
        return False
    
    base_url = "https://api-v2.intrinio.com"
    session = requests.Session()
    session.auth = (api_key, '')
    
    # Test different endpoints
    endpoints = [
        ("securities", "Basic securities info"),
        ("securities/AAPL", "Apple stock info"),
        ("securities/AAPL/prices", "Apple historical prices"),
        ("securities/AAPL/prices/intraday", "Apple intraday prices"),
        ("options/prices", "Options data"),
        ("account", "Account information")
    ]
    
    available_endpoints = []
    
    for endpoint, description in endpoints:
        try:
            url = f"{base_url}/{endpoint}"
            response = session.get(url, params={'page_size': 1})
            
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
    
    return len(available_endpoints) > 0


def test_account_info():
    """Test account information access."""
    print("\n=== Testing Account Information ===")
    
    api_key = os.getenv('INTRINIO_API_KEY')
    if not api_key:
        print("❌ No API key available")
        return False
    
    base_url = "https://api-v2.intrinio.com"
    session = requests.Session()
    session.auth = (api_key, '')
    
    try:
        url = f"{base_url}/account"
        response = session.get(url)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Account information retrieved:")
            print(f"   Plan: {data.get('plan_name', 'N/A')}")
            print(f"   Monthly calls: {data.get('monthly_calls', 'N/A')}")
            print(f"   Calls used: {data.get('calls_used', 'N/A')}")
            print(f"   Calls remaining: {data.get('calls_remaining', 'N/A')}")
            return True
        elif response.status_code == 401:
            print("❌ Cannot access account info - API key issue")
            return False
        else:
            print(f"⚠️  Account info status: {response.status_code}")
            print("Response:", response.text[:200])
            return False
            
    except Exception as e:
        print(f"❌ Error checking account: {e}")
        return False


def main():
    """Main test function."""
    print("Intrinio API Basic Connectivity Test")
    print("=" * 50)
    
    # Run all tests
    tests = [
        test_api_key_access,
        test_available_securities,
        test_ftse_specific_access,
        test_data_endpoints,
        test_account_info
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{len(tests)} tests")
    
    if passed > 0:
        print("✅ API key is working! The issue is likely data access permissions.")
        print("\nNext steps:")
        print("1. Check your Intrinio plan level")
        print("2. Contact Intrinio support about FTSE 100 access")
        print("3. Consider using alternative data sources")
    else:
        print("❌ API key is not working. Check your key and try again.")
    
    return passed > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
