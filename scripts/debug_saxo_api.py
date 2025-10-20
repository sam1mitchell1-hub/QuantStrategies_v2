#!/usr/bin/env python3
"""
Debug script to test Saxo API endpoints and understand the response format.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_providers.saxo_data_provider import SaxoDataProvider
import logging
import requests

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_api_endpoints():
    """Test various Saxo API endpoints to understand what's available."""
    print("Testing Saxo API endpoints...")
    
    try:
        provider = SaxoDataProvider()
        
        # Test account endpoint
        print("\n1. Testing account endpoint...")
        account_data = provider._make_request("/port/v1/accounts/me")
        if account_data:
            print(f"✅ Account data: {account_data}")
        else:
            print("❌ No account data")
        
        # Test instruments search
        print("\n2. Testing instruments search...")
        instruments = provider.search_instruments("Microsoft", "Stock")
        if instruments:
            print(f"✅ Found {len(instruments)} instruments")
            for i, inst in enumerate(instruments[:2]):
                print(f"   {i+1}. {inst}")
        else:
            print("❌ No instruments found")
        
        # Test instrument details
        if instruments:
            print("\n3. Testing instrument details...")
            uic = instruments[0].get('Uic') or instruments[0].get('Identifier')
            if uic:
                details = provider.get_instrument_details(uic)
                if details:
                    print(f"✅ Instrument details: {details}")
                else:
                    print("❌ No instrument details")
        
        # Test different chart endpoints
        if instruments:
            print("\n4. Testing chart endpoints...")
            uic = instruments[0].get('Uic') or instruments[0].get('Identifier')
            if uic:
                # Try different chart endpoints
                endpoints_to_try = [
                    "/chart/v1/charts",
                    "/chart/v1/charts/",
                    "/chart/v1/charts/data",
                    "/chart/v1/charts/price"
                ]
                
                for endpoint in endpoints_to_try:
                    print(f"   Trying {endpoint}...")
                    params = {'Uic': uic, 'Horizon': 1440, 'Count': 10}
                    data = provider._make_request(endpoint, params)
                    if data:
                        print(f"   ✅ {endpoint} returned data: {len(data) if isinstance(data, list) else 'non-list'}")
                        if isinstance(data, list) and len(data) > 0:
                            print(f"   Sample: {data[0]}")
                    else:
                        print(f"   ❌ {endpoint} failed")
        
        # Test direct API call
        print("\n5. Testing direct API call...")
        try:
            response = provider.session.get(f"{provider.config['base_url']}/ref/v1/instruments", 
                                          params={'AssetTypes': 'Stock', 'Keywords': 'AAPL'})
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Direct call successful: {len(data.get('Data', []))} instruments")
            else:
                print(f"   ❌ Direct call failed: {response.text}")
        except Exception as e:
            print(f"   ❌ Direct call error: {e}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_api_endpoints()
