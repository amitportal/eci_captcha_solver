#!/usr/bin/env python3
"""
Test Script for ECI CAPTCHA Solver API
This script tests the CAPTCHA solver API using the test_captcha_2.png file.
"""

import requests
import sys
import os
import json

def test_captcha_solver(api_url):
    """
    Test the CAPTCHA solver API with test_captcha_2.png
    
    Args:
        api_url (str): The URL of the deployed API
    """
    # Check if the test image exists
    image_path = "test_captcha_2.png"
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found in the current directory")
        return False
        
    print(f"Testing CAPTCHA solver API at: {api_url}")
    print(f"Using test image: {image_path}")
    
    try:
        # Test health endpoint first
        print("\n1. Testing health endpoint...")
        health_response = requests.get(f"{api_url}/health", timeout=30)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"   ✓ Health check passed: {health_data['status']}")
            print(f"   Device info: {health_data['device_info']}")
        else:
            print(f"   ✗ Health check failed with status code: {health_response.status_code}")
            return False
            
        # Test stats endpoint
        print("\n2. Testing stats endpoint...")
        stats_response = requests.get(f"{api_url}/stats", timeout=30)
        if stats_response.status_code == 200:
            stats_data = stats_response.json()
            print(f"   ✓ Stats endpoint working")
            print(f"   Total requests so far: {stats_data['total_requests']}")
        else:
            print(f"   ✗ Stats endpoint failed with status code: {stats_response.status_code}")
            # Continue anyway as this might be a new deployment
            
        # Test solve endpoint
        print("\n3. Testing solve endpoint...")
        with open(image_path, 'rb') as image_file:
            files = {'image': image_file}
            solve_response = requests.post(f"{api_url}/solve", files=files, timeout=60)
            
        if solve_response.status_code == 200:
            solve_data = solve_response.json()
            print(f"   ✓ CAPTCHA solve request successful")
            
            if solve_data.get('success'):
                print(f"   Solution: {solve_data['solution']}")
                print(f"   Confidence: {solve_data['confidence']:.3f}")
                print(f"   Total time: {solve_data['total_time']:.3f}s")
                print(f"   Device used: {solve_data['device']}")
            else:
                print(f"   ✗ Failed to solve CAPTCHA: {solve_data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"   ✗ Solve endpoint failed with status code: {solve_response.status_code}")
            print(f"   Response: {solve_response.text}")
            return False
            
        # Test stats endpoint again to see updated statistics
        print("\n4. Checking updated stats...")
        stats_response = requests.get(f"{api_url}/stats", timeout=30)
        if stats_response.status_code == 200:
            stats_data = stats_response.json()
            print(f"   ✓ Stats updated successfully")
            print(f"   Total requests: {stats_data['total_requests']}")
            print(f"   Successful solutions: {stats_data['successful_solutions']}")
            print(f"   Success rate: {stats_data['success_rate']*100:.1f}%")
        else:
            print(f"   ⚠ Could not retrieve updated stats")
            
        print("\n✓ All tests completed successfully!")
        return True
        
    except requests.exceptions.Timeout:
        print("   ✗ Request timed out. If using Render.com free tier, this might be due to cold start.")
        print("     Please try again in a few seconds.")
        return False
    except requests.exceptions.ConnectionError:
        print("   ✗ Could not connect to the API. Please check:")
        print("     1. The API URL is correct")
        print("     2. The API is running")
        print("     3. Your network connection")
        return False
    except Exception as e:
        print(f"   ✗ An error occurred: {str(e)}")
        return False

def main():
    """Main function"""
    print("ECI CAPTCHA Solver API Test")
    print("=" * 30)
    
    # Check if API URL is provided as command line argument
    if len(sys.argv) < 2:
        print("Usage: python test_captcha_api.py <api_url>")
        print("Example: python test_captcha_api.py https://your-api.onrender.com")
        print("\nIf you haven't deployed the API yet, please follow the deployment instructions in DEPLOYMENT.md")
        return 1
        
    api_url = sys.argv[1].rstrip('/')  # Remove trailing slash if present
    
    # Test the API
    success = test_captcha_solver(api_url)
    
    if success:
        print("\n🎉 API test completed successfully!")
        return 0
    else:
        print("\n❌ API test failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())