# ECI CAPTCHA Solver API (Completely Free Deployment)

This is a FastAPI-based web service that exposes the ECI CAPTCHA solving functionality via HTTP endpoints. It can be deployed completely free using multiple options.

## Features

- Fast CAPTCHA solving using trained deep learning models
- RESTful API with JSON responses
- Automatic documentation via Swagger UI
- Performance statistics tracking
- Health check endpoint
- Model caching for improved performance

## API Endpoints

### POST /solve
Solve a CAPTCHA image.

**Request:**
```
POST /solve
Content-Type: multipart/form-data

image: [image file data]
```

**Response:**
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

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "device_info": "CPU",
  "timestamp": "2025-09-16T22:30:00Z"
}
```

### GET /stats
Performance statistics.

**Response:**
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

## Local Development

1. Install dependencies:
   ```bash
   pip install -r api_requirements.txt
   ```

2. Run the API server:
   ```bash
   python captcha_api.py
   ```

3. Access the API at `http://localhost:8000`

4. View interactive API documentation at `http://localhost:8000/docs`

## Deployment (Completely Free Options)

### Option 1: Render.com (Easiest - Recommended for Quick Start)

1. Fork this repository to your GitHub account
2. Create a new **Web Service** on Render (NOT Private Service)
3. Connect it to your forked repository
4. Set the build command to: `pip install -r api_requirements.txt`
5. Set the start command to: `python captcha_api.py`
6. Add environment variable: `PYTHON_VERSION` = `3.11.0`

The `render.yaml` file in this repository provides the necessary configuration for deployment.

**Important Notes:**
- Render's free tier costs $0/month with no credit card required
- Free tier includes 750 GB bandwidth/month and 5 million requests/month
- Services sleep after 15 minutes of inactivity (cold start on first request)

### Option 2: GitHub Actions with Self-Hosted Runner (Best Performance)

1. Set up a GitHub Actions workflow that triggers on push to main branch
2. Configure a self-hosted runner on your development machine
3. Use a tunneling service (like ngrok) to expose the local API to the internet

See `API_DEPLOYMENT.md` and `API_DEPLOYMENT_METHOD.md` for detailed instructions.

## Testing

Run the test script to verify the API functionality:

```bash
python test_api.py [optional_image_path]
```

You can also test a deployed API:

```bash
python test_api.py https://your-api-url.onrender.com
```

## Performance

The API maintains the same performance characteristics as the command-line version:
- Model caching to avoid reloading within 10 minutes
- Fast solving time (~0.7 seconds total for first request, much faster for subsequent requests)
- Minimal preprocessing for maximum speed

## Requirements

- Python 3.11
- TensorFlow 2.20
- OpenCV
- FastAPI
- Uvicorn

See `api_requirements.txt` for the complete list of Python dependencies.

## Cost Summary

All deployment options are completely free:
- Render.com: $0/month (free tier with generous limits)
- GitHub Actions: $0/month (free for public repositories)
- Ngrok: $0/month (free tier sufficient for testing)
- No credit cards required for any service