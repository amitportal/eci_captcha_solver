# Deploying CAPTCHA Solver API to Render (Free Tier)

This document provides step-by-step instructions for deploying the CAPTCHA Solver API to Render.com using their free tier, which costs $0.

## Prerequisites

1. A GitHub account
2. A Render.com account (free tier is completely free)

## Deployment Steps

### 1. Fork the Repository

1. Go to your GitHub repository
2. Click the "Fork" button to create a copy of the repository under your account

### 2. Create a Render Account

1. Visit [https://render.com](https://render.com)
2. Sign up for a free account using your GitHub account
3. No credit card required for free tier

### 3. Create a New Web Service

1. After logging into Render, click "New" → "Web Service" (NOT Private Service)
2. Connect your GitHub account when prompted
3. Select the forked repository

### 4. Configure the Web Service

Fill in the following information:

- **Name**: eci-captcha-solver (or any name you prefer)
- **Branch**: main (or the branch you want to deploy)
- **Root Directory**: Leave empty (the root of the repository)
- **Environment**: Python 3
- **Build Command**: `pip install -r api_requirements.txt`
- **Start Command**: `python captcha_api.py`

### 5. Add Environment Variables

Add the following environment variable:

- **Key**: PYTHON_VERSION
- **Value**: 3.11.0

### 6. Advanced Settings

In the advanced settings, you may want to configure:

- **Auto-Deploy**: Yes (to automatically deploy new commits)
- **Health Check Path**: /health

### 7. Deploy

Click "Create Web Service" to begin the deployment process.

Render will:
1. Clone your repository
2. Run the build command to install dependencies
3. Run the start command to launch the API

### 8. Access Your API

Once deployment is complete, Render will provide a URL for your API, which will look something like:
`https://your-service-name.onrender.com`

You can access your API endpoints at:
- `https://your-service-name.onrender.com/solve` (POST)
- `https://your-service-name.onrender.com/health` (GET)
- `https://your-service-name.onrender.com/stats` (GET)

## Free Tier Limitations

1. **Cold Start**: Render's free tier puts services to sleep after 15 minutes of inactivity. The first request after sleep will take longer (15-30 seconds) as the service wakes up.

2. **No SSL Custom Domains**: Free tier doesn't include custom domains with SSL, but you get a free subdomain with SSL.

3. **Limited Resources**: Free tier has limited CPU and memory compared to paid plans.

## Model Requirements

For the API to function correctly, you must have the trained model files in the `models/` directory:
- `models/simple_crnn_200x80_raw.keras` (required for default operation)
- Other models as needed for different preprocessing methods

**IMPORTANT**: Ensure these files are committed to your repository and not excluded in `.gitignore`. The model file size should be under 500MB to work with Render's free tier.

## Testing the Deployed API

You can test the deployed API using curl:

```bash
# Health check
curl https://your-service-name.onrender.com/health

# Stats
curl https://your-service-name.onrender.com/stats

# Solve (replace with your actual URL and image path)
curl -X POST -F "image=@path/to/captcha.png" https://your-service-name.onrender.com/solve
```

You can also use the provided `test_api.py` script:
```bash
python test_api.py https://your-service-name.onrender.com
```

## Cost Information

This deployment is 100% free:
- Render's web service free tier includes:
  - Monthly bandwidth: 750 GB
  - Monthly requests: 5 million
  - Build minutes: 500 min/month
  - Sleep after 15 min inactivity
- No credit card required
- No time limit on free usage

## Important Notes

1. **Cold Start**: The first request after inactivity will be slow as the service wakes up. Subsequent requests will be fast.

2. **Model Files**: Make sure your trained model files are in the `models/` directory:
   - `models/simple_crnn_200x80_raw.keras` (required)
   - Other models as needed

3. **File Size Limits**: Render has file size limits for free accounts. Ensure your model files don't exceed these limits.

4. **Performance**: The API will maintain the same performance characteristics as the local version, with model caching for subsequent requests.

5. **Logging**: Logs are available through the Render dashboard for monitoring and debugging.

## Troubleshooting

1. **Deployment Failures**: Check the build logs in the Render dashboard for error messages.

2. **Runtime Errors**: Check the application logs in the Render dashboard.

3. **Slow Responses**: This might be due to cold start. Try making a second request after a few seconds.

4. **Missing Models**: Ensure all required model files are committed to the repository and not in `.gitignore`.

5. **Memory Issues**: If you see out-of-memory errors, the model might be too large for the free tier. Consider optimizing the model size.

## Alternative Free Deployment Options

If Render's free tier doesn't meet your needs, you can also deploy using:

1. **GitHub Actions with Self-Hosted Runner**: 
   - Run the API on your own machine
   - Use ngrok for public access
   - Completely free but requires keeping your machine on

2. **Railway.app Free Tier**:
   - Similar to Render but with different limitations
   - May have better performance for some use cases

Both options are also completely free and documented in other files in this repository.