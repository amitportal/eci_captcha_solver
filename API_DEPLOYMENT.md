# CAPTCHA Solver API Deployment (Completely Free)

This document provides instructions for deploying the CAPTCHA Solver API using various free methods. All options described here cost $0 to implement and run.

## Render.com Deployment (Easiest - Recommended for Quick Start)

### Prerequisites
1. Render.com account (completely free, no credit card required)
2. GitHub account

### Setup Instructions

1. Fork this repository to your GitHub account
2. Create a new **Web Service** on Render (NOT Private Service)
3. Connect it to your forked repository
4. Set the build command to: `pip install -r api_requirements.txt`
5. Set the start command to: `python captcha_api.py`
6. Add environment variable: `PYTHON_VERSION` = `3.11.0`

The `render.yaml` file in this repository provides the necessary configuration for deployment.

### Important Notes for Free Deployment

- Render's free tier is completely free with no credit card required
- Free tier includes:
  - 750 GB bandwidth/month
  - 5 million requests/month
  - 500 build minutes/month
- Services sleep after 15 minutes of inactivity (cold start on first request)
- No custom domains with SSL on free tier (you get a free subdomain with SSL)

## GitHub Actions Deployment (Self-Hosted Runner - Best Performance)

### Prerequisites
1. GitHub account
2. Local machine to act as self-hosted runner
3. Public IP or tunneling service (like ngrok - free tier available)

### Setup Instructions

#### 1. Configure Self-Hosted Runner

1. Go to your GitHub repository
2. Navigate to Settings > Actions > Runners
3. Click "New self-hosted runner"
4. Select your operating system
5. Follow the instructions to download and configure the runner

#### 2. Start the Runner

```bash
# Navigate to the runner directory
cd actions-runner

# Configure the runner (run once)
./config.sh --url https://github.com/your-username/your-repo --token YOUR_TOKEN

# Start the runner
./run.sh
```

#### 3. Set Up Tunneling

To expose your local API to the internet, use a tunneling service like ngrok:

```bash
# Install ngrok
# Download from https://ngrok.com/download

# Start tunnel to your API (default port 8000)
ngrok http 8000
```

Note the HTTPS URL provided by ngrok - this will be your public API endpoint.

### GitHub Actions Workflow

The workflow will automatically:
1. Deploy the latest code to your self-hosted runner
2. Install dependencies
3. Start the API service
4. (Optional) Restart ngrok tunnel

### Usage

Once deployed, you can access your API at the ngrok URL:
- POST `/solve` - Solve CAPTCHA images
- GET `/health` - Health check
- GET `/stats` - Performance statistics

## Fallback Strategy (Recommended for Production)

For production use, implement a client-side fallback strategy:

1. Primary endpoint: Render.com deployment (always available)
2. Secondary endpoint: Self-hosted runner (better performance when available)
3. Client automatically switches to secondary if primary is slow/unavailable

## Performance Considerations

### Self-Hosted Runner
- No cold start delays
- Full control over hardware
- Same performance as local execution
- Requires machine to be always on

### Cloud Platforms (Render, Railway, etc.)
- Cold start delays after inactivity (15+ seconds for first request)
- Resource limitations
- Network latency
- No machine maintenance required

## Security

When using self-hosted runners:
1. Limit runner access to specific repositories
2. Use secure tokens
3. Keep runner software updated
4. Monitor runner activity

When exposing APIs publicly:
1. Implement rate limiting
2. Validate all inputs
3. Use HTTPS (provided by ngrok and cloud platforms)
4. Monitor for abuse

## Model Requirements

For the API to function correctly, you must have the trained model files in the `models/` directory:
- `models/simple_crnn_200x80_raw.keras` (required for default operation)
- Other models as needed for different preprocessing methods

Ensure these files are committed to your repository and not excluded in `.gitignore`.

The model file should be under 500MB to work with Render's free tier.

## Testing Your Deployment

After deployment, you can test the API using curl:

```bash
# Health check
curl https://your-api-url.onrender.com/health

# Statistics
curl https://your-api-url.onrender.com/stats

# Solve a CAPTCHA (replace with your actual URL and image path)
curl -X POST -F "image=@path/to/captcha.png" https://your-api-url.onrender.com/solve
```

You can also use the provided `test_api.py` script:
```bash
python test_api.py https://your-api-url.onrender.com
```

## Cost Summary

All deployment options are completely free:
- Render.com: $0/month (free tier with generous limits)
- GitHub Actions: $0/month (free for public repositories)
- Ngrok: $0/month (free tier sufficient for testing)
- No credit cards required for any service

Choose Render.com for the easiest quick deployment, or self-hosted runner for the best performance.