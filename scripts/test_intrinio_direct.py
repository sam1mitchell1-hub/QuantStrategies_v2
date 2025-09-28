#!/usr/bin/env python3
"""
Direct Intrinio API Test

This script tests Intrinio API connectivity by asking for your API key directly.
Use this to test if your API key works and what data is available.
"""

import requests
import json
from datetime import datetime, timedelta


def test_intrinio_api(api_key):
    """Test Intrinio API with provided key."""
    print("=== Testing Intrinio API ===")
    print(f"API Key: {api_key[:8]}...")
    
    base_url = "https://api-v2.intrinio.com"
    session = requests.Session()
    session.auth = (api_key, '')
    
    # Test 1: Basic connection
    print("\n1. Testing basic connection...")
    try:
        url = f"{base_url}/securities"
        response = session.get(url, params={'page_size': 1})
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ Basic connection successful!")
            data = response.json()
            print(f"   Response keys: {list(data.keys())}")
        elif response.status_code == 401:
            print("   ❌ Unauthorized - API key invalid or expired")
            return False
        else:
            print(f"   ❌ Unexpected status: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
        return False
    
    # Test 2: Available securities
    print("\n2. Testing available securities...")
    try:
        url = f"{base_url}/securities"
        response = session.get(url, params={'page_size': 5})
        
        if response.status_code == 200:
            data = response.json()
            securities = data.get('securities', [])
            print(f"   ✅ Found {len(securities)} securities")
            print("   Sample securities:")
            for i, sec in enumerate(securities[:3]):
                print(f"     {i+1}. {sec.get('ticker', 'N/A')} - {sec.get('name', 'N/A')}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 3: FTSE 100 symbols
    print("\n3. Testing FTSE 100 symbols...")
    ftse_symbols = ['UKX', 'FTSE', '^FTSE', 'FTSE.L', 'UKX.L']
    ftse_found = False
    
    for symbol in ftse_symbols:
        try:
            url = f"{base_url}/securities/{symbol}"
            response = session.get(url)
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ {symbol}: {data.get('name', 'N/A')}")
                print(f"      Type: {data.get('type', 'N/A')}")
                print(f"      Currency: {data.get('currency', 'N/A')}")
                ftse_found = True
            elif response.status_code == 404:
                print(f"   ❌ {symbol}: Not found")
            elif response.status_code == 401:
                print(f"   ❌ {symbol}: Unauthorized")
            else:
                print(f"   ⚠️  {symbol}: Status {response.status_code}")
        except Exception as e:
            print(f"   ❌ {symbol}: Error - {e}")
    
    # Test 4: Data endpoints
    print("\n4. Testing data endpoints...")
    endpoints = [
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
                print(f"   ✅ {endpoint}: {description}")
                available_endpoints.append(endpoint)
            elif response.status_code == 401:
                print(f"   ❌ {endpoint}: Unauthorized - {description}")
            elif response.status_code == 404:
                print(f"   ❌ {endpoint}: Not found - {description}")
            else:
                print(f"   ⚠️  {endpoint}: Status {response.status_code} - {description}")
        except Exception as e:
            print(f"   ❌ {endpoint}: Error - {e}")
    
    # Test 5: Account information
    print("\n5. Testing account information...")
    try:
        url = f"{base_url}/account"
        response = session.get(url)
        
        if response.status_code == 200:
            data = response.json()
            print("   ✅ Account information retrieved:")
            print(f"      Plan: {data.get('plan_name', 'N/A')}")
            print(f"      Monthly calls: {data.get('monthly_calls', 'N/A')}")
            print(f"      Calls used: {data.get('calls_used', 'N/A')}")
            print(f"      Calls remaining: {data.get('calls_remaining', 'N/A')}")
        elif response.status_code == 401:
            print("   ❌ Cannot access account info - API key issue")
        else:
            print(f"   ⚠️  Account info status: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error checking account: {e}")
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Basic connection: {'✅' if True else '❌'}")
    print(f"FTSE 100 access: {'✅' if ftse_found else '❌'}")
    print(f"Available endpoints: {len(available_endpoints)}")
    
    if ftse_found:
        print("\n🎉 Great! Your API key has FTSE 100 access!")
        print("The original error might have been a temporary issue.")
    else:
        print("\n⚠️  Your API key doesn't have FTSE 100 access.")
        print("This confirms the 401 errors you were seeing.")
        print("Consider upgrading your Intrinio plan or using alternative data sources.")
    
    return True


def main():
    """Main function."""
    print("Direct Intrinio API Test")
    print("=" * 40)
    print("This script will test your Intrinio API key directly.")
    print()
    
    # Get API key from user
    api_key = input("Enter your Intrinio API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided")
        return False
    
    # Test the API
    success = test_intrinio_api(api_key)
    
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
    
    return success


if __name__ == "__main__":
    main()
