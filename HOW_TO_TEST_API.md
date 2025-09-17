# How to Test the ECI CAPTCHA Solver API as an External User

This document provides instructions for external users to test the ECI CAPTCHA Solver API that has been deployed using the free options described in the project.

## Prerequisites

1. Python 3.6 or higher (for Python test script)
2. curl (for bash script or direct API calls)
3. Access to a deployed instance of the CAPTCHA solver API

## Deployment Options

The API can be deployed using these completely free methods:

1. **Render.com** - Cloud deployment with free tier
2. **GitHub Actions with Self-Hosted Runner** - Local deployment exposed via tunneling
3. **Other cloud platforms** (Railway.app, Fly.io, etc.)

## Testing Methods

### Method 1: Using the Python Test Script

1. Download the `external_api_test.py` script from the repository
2. Install required dependencies:
   ```bash
   pip install requests
   ```
3. Run the script with your CAPTCHA image and API URL:
   ```bash
   python external_api_test.py <captcha_image_path> <api_url>
   ```

Example:
```bash
python external_api_test.py test_captcha.png https://your-app.onrender.com
```

### Method 2: Using the Bash Script

1. Download the `external_api_test.sh` script from the repository
2. Make it executable:
   ```bash
   chmod +x external_api_test.sh
   ```
3. Run the script:
   ```bash
   ./external_api_test.sh <captcha_image_path> <api_url>
   ```

Example:
```bash
./external_api_test.sh test_captcha.png https://your-app.onrender.com
```

### Method 3: Using curl Directly

You can also test the API directly using curl:

1. **Solve a CAPTCHA:**
   ```bash
   curl -X POST -F "image=@<captcha_image_path>" <api_url>/solve
   ```

2. **Health Check:**
   ```bash
   curl <api_url>/health
   ```

3. **Get Statistics:**
   ```bash
   curl <api_url>/stats
   ```

## Example Usage

### Python Script Examples:
```bash
# Solve a CAPTCHA
python external_api_test.py captcha.png https://your-api.onrender.com

# Check API health
python external_api_test.py --health https://your-api.onrender.com

# Get API statistics
python external_api_test.py --stats https://your-api.onrender.com
```

### Bash Script Examples:
```bash
# Solve a CAPTCHA
./external_api_test.sh captcha.png https://your-api.onrender.com

# Check API health
./external_api_test.sh --health https://your-api.onrender.com

# Get API statistics
./external_api_test.sh --stats https://your-api.onrender.com
```

### Curl Examples:
```bash
# Solve a CAPTCHA
curl -X POST -F "image=@captcha.png" https://your-api.onrender.com/solve

# Check API health
curl https://your-api.onrender.com/health

# Get API statistics
curl https://your-api.onrender.com/stats
```

## Expected Responses

### Successful CAPTCHA Solution:
```json
{
  "solution": "abcdef",
  "success": true,
  "total_time": 0.753,
  "model_load_time": 0.210,
  "image_load_time": 0.020,
  "processing_time": 0.010,
  "prediction_time": 0.500,
  "result_processing_time": 0.013,
  "confidence": 0.998,
  "device": "CPU",
  "timestamp": "2025-09-16T22:30:00Z"
}
```

### Failed CAPTCHA Solution:
```json
{
  "solution": "no_solution_found",
  "success": false,
  "error": "Error description",
  "total_time": 0.123,
  "timestamp": "2025-09-16T22:30:00Z"
}
```

### Health Check:
```json
{
  "status": "healthy",
  "device_info": "CPU",
  "timestamp": "2025-09-16T22:30:00Z"
}
```

### Statistics:
```json
{
  "total_requests": 1250,
  "successful_solutions": 1245,
  "failed_solutions": 5,
  "average_solve_time": 0.725,
  "success_rate": 0.996,
  "timestamp": "2025-09-16T22:30:00Z"
}
```

## Troubleshooting

1. **Timeout Errors**: Free tier deployments (especially Render.com) may have cold starts that cause timeouts. Try increasing the timeout value in the scripts or retry the request.

2. **Connection Errors**: Ensure the API URL is correct and accessible from your network.

3. **File Not Found Errors**: Verify that the CAPTCHA image file exists and is accessible.

4. **API Errors**: Check the error message in the response for specific details about what went wrong.