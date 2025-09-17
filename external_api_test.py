#!/usr/bin/env python3
"""
External User Test Script for ECI CAPTCHA Solver API
This script allows external users to test the CAPTCHA solver API by sending CAPTCHA images for solving.
"""

import requests
import sys
import os
import time
from pathlib import Path

class CaptchaSolverClient:
    def __init__(self):
        # API endpoints - you can modify these based on your deployment
        self.endpoints = [
            # Add your deployed API URLs here
            # Example: "https://your-app.onrender.com",
            # Example: "https://your-ngrok-url.ngrok.io",
        ]
        
        # Timeout for requests (increased to handle cold starts on free tiers)
        self.timeout = 45

    def add_endpoint(self, url):
        """Add an API endpoint to test"""
        self.endpoints.append(url)
        
    def solve_captcha(self, image_path, api_url=None):
        """
        Solve a CAPTCHA using the API
        
        Args:
            image_path (str): Path to the CAPTCHA image file
            api_url (str, optional): Specific API URL to use
            
        Returns:
            dict: API response with solution or error
        """
        # If a specific URL is provided, use only that
        if api_url:
            endpoints_to_try = [api_url]
        else:
            # Otherwise, try all configured endpoints
            endpoints_to_try = self.endpoints
            
        if not endpoints_to_try:
            raise ValueError("No API endpoints configured. Please add at least one endpoint.")
            
        # Try each endpoint
        for endpoint in endpoints_to_try:
            try:
                print(f"Trying to solve CAPTCHA using endpoint: {endpoint}")
                
                # Check if file exists
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"CAPTCHA image file not found: {image_path}")
                
                # Send the request
                with open(image_path, 'rb') as image_file:
                    files = {'image': image_file}
                    response = requests.post(
                        f"{endpoint}/solve",
                        files=files,
                        timeout=self.timeout
                    )
                
                # Check if request was successful
                if response.status_code == 200:
                    result = response.json()
                    return result
                else:
                    print(f"API returned status code {response.status_code}")
                    print(f"Response: {response.text}")
                    
            except requests.exceptions.Timeout:
                print(f"Request to {endpoint} timed out after {self.timeout} seconds")
            except requests.exceptions.ConnectionError:
                print(f"Could not connect to {endpoint}")
            except Exception as e:
                print(f"Error when trying {endpoint}: {str(e)}")
                
        # If we get here, all endpoints failed
        raise Exception("All API endpoints failed. Please check your configuration and network connection.")

    def health_check(self, api_url=None):
        """
        Perform a health check on the API
        
        Args:
            api_url (str, optional): Specific API URL to check
            
        Returns:
            dict: Health check response
        """
        # If a specific URL is provided, use only that
        if api_url:
            endpoints_to_try = [api_url]
        else:
            # Otherwise, try all configured endpoints
            endpoints_to_try = self.endpoints
            
        if not endpoints_to_try:
            raise ValueError("No API endpoints configured. Please add at least one endpoint.")
            
        # Try each endpoint
        for endpoint in endpoints_to_try:
            try:
                print(f"Performing health check on: {endpoint}")
                
                response = requests.get(
                    f"{endpoint}/health",
                    timeout=self.timeout
                )
                
                # Check if request was successful
                if response.status_code == 200:
                    result = response.json()
                    return result
                else:
                    print(f"Health check returned status code {response.status_code}")
                    print(f"Response: {response.text}")
                    
            except requests.exceptions.Timeout:
                print(f"Health check to {endpoint} timed out after {self.timeout} seconds")
            except requests.exceptions.ConnectionError:
                print(f"Could not connect to {endpoint}")
            except Exception as e:
                print(f"Error when trying health check on {endpoint}: {str(e)}")
                
        # If we get here, all endpoints failed
        raise Exception("Health check failed on all endpoints.")

    def get_stats(self, api_url=None):
        """
        Get statistics from the API
        
        Args:
            api_url (str, optional): Specific API URL to check
            
        Returns:
            dict: Statistics response
        """
        # If a specific URL is provided, use only that
        if api_url:
            endpoints_to_try = [api_url]
        else:
            # Otherwise, try all configured endpoints
            endpoints_to_try = self.endpoints
            
        if not endpoints_to_try:
            raise ValueError("No API endpoints configured. Please add at least one endpoint.")
            
        # Try each endpoint
        for endpoint in endpoints_to_try:
            try:
                print(f"Getting statistics from: {endpoint}")
                
                response = requests.get(
                    f"{endpoint}/stats",
                    timeout=self.timeout
                )
                
                # Check if request was successful
                if response.status_code == 200:
                    result = response.json()
                    return result
                else:
                    print(f"Stats request returned status code {response.status_code}")
                    print(f"Response: {response.text}")
                    
            except requests.exceptions.Timeout:
                print(f"Stats request to {endpoint} timed out after {self.timeout} seconds")
            except requests.exceptions.ConnectionError:
                print(f"Could not connect to {endpoint}")
            except Exception as e:
                print(f"Error when trying to get stats from {endpoint}: {str(e)}")
                
        # If we get here, all endpoints failed
        raise Exception("Stats request failed on all endpoints.")

def print_usage():
    """Print usage instructions"""
    print("ECI CAPTCHA Solver API Test Script")
    print("=" * 40)
    print("Usage:")
    print("  python test_external_api.py <captcha_image_path> [api_url]")
    print("  python test_external_api.py --health [api_url]")
    print("  python test_external_api.py --stats [api_url]")
    print("")
    print("Examples:")
    print("  python test_external_api.py captcha.png")
    print("  python test_external_api.py captcha.png https://your-api.onrender.com")
    print("  python test_external_api.py --health https://your-api.onrender.com")
    print("  python test_external_api.py --stats https://your-api.onrender.com")
    print("")

def main():
    """Main function"""
    # Create client instance
    client = CaptchaSolverClient()
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print_usage()
        return 1
        
    # Parse arguments
    if sys.argv[1] == "--help" or sys.argv[1] == "-h":
        print_usage()
        return 0
    elif sys.argv[1] == "--health":
        # Health check
        api_url = sys.argv[2] if len(sys.argv) > 2 else None
        try:
            result = client.health_check(api_url)
            print("\nHealth Check Result:")
            print("-" * 20)
            print(f"Status: {result.get('status', 'Unknown')}")
            print(f"Device Info: {result.get('device_info', 'Unknown')}")
            print(f"Timestamp: {result.get('timestamp', 'Unknown')}")
            return 0
        except Exception as e:
            print(f"Error: {e}")
            return 1
    elif sys.argv[1] == "--stats":
        # Get statistics
        api_url = sys.argv[2] if len(sys.argv) > 2 else None
        try:
            result = client.get_stats(api_url)
            print("\nStatistics:")
            print("-" * 15)
            print(f"Total Requests: {result.get('total_requests', 0)}")
            print(f"Successful Solutions: {result.get('successful_solutions', 0)}")
            print(f"Failed Solutions: {result.get('failed_solutions', 0)}")
            print(f"Average Solve Time: {result.get('average_solve_time', 0)}s")
            print(f"Success Rate: {result.get('success_rate', 0)*100:.1f}%")
            print(f"Timestamp: {result.get('timestamp', 'Unknown')}")
            return 0
        except Exception as e:
            print(f"Error: {e}")
            return 1
    else:
        # Solve CAPTCHA
        image_path = sys.argv[1]
        api_url = sys.argv[2] if len(sys.argv) > 2 else None
        
        try:
            result = client.solve_captcha(image_path, api_url)
            print("\nCAPTCHA Solution Result:")
            print("-" * 25)
            if result.get('success'):
                print(f"Solution: {result.get('solution')}")
                print(f"Confidence: {result.get('confidence', 0):.3f}")
                print(f"Total Time: {result.get('total_time', 0):.3f}s")
                print(f"Device: {result.get('device', 'Unknown')}")
            else:
                print("Failed to solve CAPTCHA")
                print(f"Error: {result.get('error', 'Unknown error')}")
            print(f"Timestamp: {result.get('timestamp', 'Unknown')}")
            return 0
        except Exception as e:
            print(f"Error: {e}")
            return 1

if __name__ == "__main__":
    sys.exit(main())