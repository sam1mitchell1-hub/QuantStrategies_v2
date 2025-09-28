#!/usr/bin/env python3
"""
Quick Intrinio API Test

One-liner test to check if your Intrinio API key works.
"""

import requests
import sys

def quick_test(api_key):
    """Quick test of Intrinio API."""
    try:
        url = "https://api-v2.intrinio.com/securities"
        response = requests.get(url, auth=(api_key, ''), params={'page_size': 1})
        
        if response.status_code == 200:
            print("✅ API key works! Status:", response.status_code)
            data = response.json()
            print("Response keys:", list(data.keys()))
            return True
        elif response.status_code == 401:
            print("❌ API key invalid or expired. Status:", response.status_code)
            return False
        else:
            print("⚠️  Unexpected status:", response.status_code)
            print("Response:", response.text[:200])
            return False
    except Exception as e:
        print("❌ Error:", e)
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        api_key = input("Enter your Intrinio API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided")
        sys.exit(1)
    
    success = quick_test(api_key)
    sys.exit(0 if success else 1)
