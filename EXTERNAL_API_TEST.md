# External User Test Script for ECI CAPTCHA Solver API

This script allows external users to test the CAPTCHA solver API by sending CAPTCHA images for solving.

## Prerequisites

1. Python 3.6 or higher
2. Required Python packages (install with `pip install requests`):
   - requests

## Usage

### 1. Basic CAPTCHA Solving
```bash
python test_external_api.py <captcha_image_path> [api_url]
```

Example:
```bash
python test_external_api.py test_captcha.png
python test_external_api.py test_captcha.png https://your-api.onrender.com
```

### 2. Health Check
```bash
python test_external_api.py --health [api_url]
```

Example:
```bash
python test_external_api.py --health https://your-api.onrender.com
```

### 3. Get Statistics
```bash
python test_external_api.py --stats [api_url]
```

Example:
```bash
python test_external_api.py --stats https://your-api.onrender.com
```

## Configuration

To use specific API endpoints, you can either:

1. Pass the API URL as a command line argument
2. Modify the `endpoints` list in the script directly

## API Endpoints

The script supports multiple API endpoints for redundancy:
- Render.com deployment
- Self-hosted runner (via ngrok or direct access)
- Other cloud platforms

## Error Handling

The script includes comprehensive error handling for:
- Network timeouts
- Connection errors
- API errors
- File not found errors

## Response Format

### Successful CAPTCHA Solution
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

### Failed CAPTCHA Solution
```json
{
  "solution": "no_solution_found",
  "success": false,
  "error": "Error description",
  "total_time": 0.123,
  "timestamp": "2025-09-16T22:30:00Z"
}
```

### Health Check
```json
{
  "status": "healthy",
  "device_info": "CPU",
  "timestamp": "2025-09-16T22:30:00Z"
}
```

### Statistics
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