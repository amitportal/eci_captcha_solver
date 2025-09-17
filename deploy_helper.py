#!/usr/bin/env python3
"""
Deployment helper script for CAPTCHA Solver API
"""

import os
import subprocess
import sys

def deploy_to_render():
    """Deploy the API to Render.com"""
    print("Deploying CAPTCHA Solver API to Render.com...")
    print("Make sure you have:")
    print("1. Forked the repository to your GitHub account")
    print("2. Created a Render.com account")
    print("3. Connected your GitHub account to Render")
    print("\nDeployment steps:")
    print("1. Go to https://dashboard.render.com/")
    print("2. Click 'New' -> 'Web Service'")
    print("3. Connect to your forked repository")
    print("4. Fill in the following settings:")
    print("   - Name: eci-captcha-solver")
    print("   - Environment: Python 3")
    print("   - Build Command: pip install -r api_requirements.txt")
    print("   - Start Command: python captcha_api.py")
    print("5. Add Environment Variable:")
    print("   - Key: PYTHON_VERSION")
    print("   - Value: 3.11.0")
    print("6. Click 'Create Web Service'")
    print("\nAfter deployment, your API will be available at:")
    print("https://YOUR_SERVICE_NAME.onrender.com")
    print("\nAPI endpoints:")
    print("- POST /solve")
    print("- GET /health")
    print("- GET /stats")

def setup_github_actions():
    """Instructions for setting up GitHub Actions"""
    print("Setting up GitHub Actions for self-hosted deployment...")
    print("Make sure you have:")
    print("1. A GitHub account")
    print("2. A machine to act as a self-hosted runner")
    print("\nSetup steps:")
    print("1. Go to your repository on GitHub")
    print("2. Navigate to Settings -> Actions -> Runners")
    print("3. Click 'New self-hosted runner'")
    print("4. Select your operating system")
    print("5. Follow the instructions to download and configure the runner")
    print("6. Start the runner with: ./run.sh")
    print("\nThe GitHub Actions workflow (.github/workflows/deploy-api.yml) will:")
    print("- Automatically deploy on push to main branch")
    print("- Install dependencies")
    print("- Start the API service")
    print("\nTo expose your local API publicly, use a tunneling service like ngrok:")
    print("1. Download ngrok from https://ngrok.com/download")
    print("2. Start tunnel: ngrok http 8000")
    print("3. Use the HTTPS URL provided by ngrok as your API endpoint")

def main():
    """Main function"""
    print("CAPTCHA Solver API Deployment Helper")
    print("=" * 40)
    
    print("\nAvailable deployment options:")
    print("1. Render.com (Cloud - Free Tier)")
    print("2. GitHub Actions with Self-Hosted Runner (Local - Free)")
    print("3. Both")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == "1":
        deploy_to_render()
    elif choice == "2":
        setup_github_actions()
    elif choice == "3":
        deploy_to_render()
        print("\n" + "=" * 40)
        setup_github_actions()
    else:
        print("Invalid choice. Please run the script again and select 1, 2, or 3.")
        sys.exit(1)
    
    print("\nDeployment documentation is available in:")
    print("- API_DEPLOYMENT.md")
    print("- API_DEPLOYMENT_METHOD.md")
    print("- DEPLOYMENT.md")
    print("- CLIENT_IMPLEMENTATION.md")

if __name__ == "__main__":
    main()