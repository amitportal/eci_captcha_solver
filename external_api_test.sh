#!/bin/bash
# External User Test Script for ECI CAPTCHA Solver API
# This script allows external users to test the CAPTCHA solver API using curl

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print usage
print_usage() {
    echo "ECI CAPTCHA Solver API Test Script (Bash Version)"
    echo "==============================================="
    echo "Usage:"
    echo "  ./test_external_api.sh <captcha_image_path> <api_url>"
    echo "  ./test_external_api.sh --health <api_url>"
    echo "  ./test_external_api.sh --stats <api_url>"
    echo ""
    echo "Examples:"
    echo "  ./test_external_api.sh captcha.png https://your-api.onrender.com"
    echo "  ./test_external_api.sh --health https://your-api.onrender.com"
    echo "  ./test_external_api.sh --stats https://your-api.onrender.com"
    echo ""
}

# Check if curl is installed
if ! command -v curl &> /dev/null; then
    echo -e "${RED}Error: curl is not installed. Please install curl to use this script.${NC}"
    exit 1
fi

# Check arguments
if [ $# -lt 1 ]; then
    print_usage
    exit 1
fi

# Parse arguments
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    print_usage
    exit 0
elif [ "$1" == "--health" ]; then
    # Health check
    if [ -z "$2" ]; then
        echo -e "${RED}Error: API URL is required for health check${NC}"
        print_usage
        exit 1
    fi
    
    echo -e "${YELLOW}Performing health check on: $2${NC}"
    response=$(curl -s -w "%{http_code}" "$2/health" -o /tmp/health_response.json)
    http_code=$(tail -c 3 <<< "$response")
    content=$(head -c -3 <<< "$response")
    
    if [ "$http_code" == "200" ]; then
        echo -e "${GREEN}Health check successful${NC}"
        cat /tmp/health_response.json | python -m json.tool
    else
        echo -e "${RED}Health check failed with HTTP code: $http_code${NC}"
        echo "Response: $content"
    fi
    rm -f /tmp/health_response.json
    exit 0
elif [ "$1" == "--stats" ]; then
    # Get statistics
    if [ -z "$2" ]; then
        echo -e "${RED}Error: API URL is required for statistics${NC}"
        print_usage
        exit 1
    fi
    
    echo -e "${YELLOW}Getting statistics from: $2${NC}"
    response=$(curl -s -w "%{http_code}" "$2/stats" -o /tmp/stats_response.json)
    http_code=$(tail -c 3 <<< "$response")
    content=$(head -c -3 <<< "$response")
    
    if [ "$http_code" == "200" ]; then
        echo -e "${GREEN}Statistics request successful${NC}"
        cat /tmp/stats_response.json | python -m json.tool
    else
        echo -e "${RED}Statistics request failed with HTTP code: $http_code${NC}"
        echo "Response: $content"
    fi
    rm -f /tmp/stats_response.json
    exit 0
else
    # Solve CAPTCHA
    IMAGE_PATH="$1"
    API_URL="$2"
    
    if [ -z "$API_URL" ]; then
        echo -e "${RED}Error: API URL is required${NC}"
        print_usage
        exit 1
    fi
    
    # Check if image file exists
    if [ ! -f "$IMAGE_PATH" ]; then
        echo -e "${RED}Error: CAPTCHA image file not found: $IMAGE_PATH${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Solving CAPTCHA using: $API_URL${NC}"
    response=$(curl -s -w "%{http_code}" -X POST -F "image=@$IMAGE_PATH" "$API_URL/solve" -o /tmp/solve_response.json)
    http_code=$(tail -c 3 <<< "$response")
    content=$(head -c -3 <<< "$response")
    
    if [ "$http_code" == "200" ]; then
        echo -e "${GREEN}CAPTCHA solving request successful${NC}"
        cat /tmp/solve_response.json | python -m json.tool
    else
        echo -e "${RED}CAPTCHA solving request failed with HTTP code: $http_code${NC}"
        echo "Response: $content"
    fi
    rm -f /tmp/solve_response.json
    exit 0
fi