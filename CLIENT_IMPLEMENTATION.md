# CAPTCHA Solver Client Implementation (Free Deployment Options)

This document provides examples of how to use the CAPTCHA Solver API from client-side code with fallback strategies. All deployment options are completely free.

## JavaScript Implementation

```javascript
class CaptchaSolver {
    constructor() {
        // API endpoints in order of preference
        this.endpoints = [
            'https://your-app.onrender.com',           // Primary: Render.com (free tier)
            'https://your-ngrok-url.ngrok.io',         // Secondary: Self-hosted via ngrok
            'https://your-app.railway.app'             // Tertiary: Railway.app (free tier)
        ];
        
        // Timeout for each request (ms)
        this.timeout = 30000; // Increased timeout to handle cold starts
    }
    
    // Solve a CAPTCHA with fallback
    async solveCaptcha(imageFile) {
        // Try each endpoint in order
        for (let i = 0; i < this.endpoints.length; i++) {
            try {
                const result = await this.solveWithEndpoint(this.endpoints[i], imageFile);
                console.log(`Successfully solved with endpoint ${i + 1}`);
                return result;
            } catch (error) {
                console.warn(`Endpoint ${i + 1} failed:`, error.message);
                
                // If this is the last endpoint, throw the error
                if (i === this.endpoints.length - 1) {
                    throw new Error('All CAPTCHA solver endpoints failed');
                }
                
                // Continue to next endpoint
                continue;
            }
        }
    }
    
    // Solve with a specific endpoint
    async solveWithEndpoint(endpoint, imageFile) {
        // Create FormData
        const formData = new FormData();
        formData.append('image', imageFile);
        
        // Create AbortController for timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);
        
        try {
            const response = await fetch(`${endpoint}/solve`, {
                method: 'POST',
                body: formData,
                signal: controller.signal
            });
            
            // Clear timeout
            clearTimeout(timeoutId);
            
            // Check if response is ok
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            // Parse JSON response
            const result = await response.json();
            return result;
        } catch (error) {
            // Clear timeout
            clearTimeout(timeoutId);
            
            // Re-throw error
            throw error;
        }
    }
    
    // Health check with fallback
    async healthCheck() {
        for (let i = 0; i < this.endpoints.length; i++) {
            try {
                const response = await fetch(`${this.endpoints[i]}/health`, {
                    method: 'GET',
                    headers: {
                        'Accept': 'application/json'
                    }
                });
                
                if (response.ok) {
                    const result = await response.json();
                    return {
                        endpoint: this.endpoints[i],
                        status: result
                    };
                }
            } catch (error) {
                console.warn(`Health check failed for endpoint ${i + 1}:`, error.message);
                continue;
            }
        }
        
        throw new Error('All health checks failed');
    }
    
    // Get statistics with fallback
    async getStats() {
        for (let i = 0; i < this.endpoints.length; i++) {
            try {
                const response = await fetch(`${this.endpoints[i]}/stats`, {
                    method: 'GET',
                    headers: {
                        'Accept': 'application/json'
                    }
                });
                
                if (response.ok) {
                    const result = await response.json();
                    return result;
                }
            } catch (error) {
                console.warn(`Stats request failed for endpoint ${i + 1}:`, error.message);
                continue;
            }
        }
        
        throw new Error('All stats requests failed');
    }
}

// Usage example
async function example() {
    const solver = new CaptchaSolver();
    
    // Health check
    try {
        const health = await solver.healthCheck();
        console.log('Health check:', health);
    } catch (error) {
        console.error('Health check failed:', error.message);
    }
    
    // Solve CAPTCHA
    try {
        // Assuming you have an image file input element with id 'captcha-input'
        const fileInput = document.getElementById('captcha-input');
        const file = fileInput.files[0];
        
        if (file) {
            const result = await solver.solveCaptcha(file);
            console.log('CAPTCHA solution:', result);
            
            if (result.success) {
                console.log('Solution:', result.solution);
                console.log('Confidence:', result.confidence);
                console.log('Time taken:', result.total_time);
            } else {
                console.log('Failed to solve CAPTCHA');
            }
        } else {
            console.log('No file selected');
        }
    } catch (error) {
        console.error('CAPTCHA solving failed:', error.message);
    }
}
```

## Python Implementation

```python
import requests
from typing import List, Dict, Any, Optional
import time

class CaptchaSolver:
    def __init__(self):
        # API endpoints in order of preference
        self.endpoints = [
            'https://your-app.onrender.com',           # Primary: Render.com (free tier)
            'https://your-ngrok-url.ngrok.io',         # Secondary: Self-hosted via ngrok
        ]
        
        # Timeout for each request (seconds)
        # Increased timeout to handle cold starts on free tiers
        self.timeout = 45
    def solve_captcha(self, image_path: str) -> Dict[str, Any]:
        """
        Solve a CAPTCHA with fallback to alternative endpoints.
        
        Args:
            image_path: Path to the CAPTCHA image file
            
        Returns:
            Dictionary with solution and metadata
            
        Raises:
            Exception: If all endpoints fail
        """
        # Try each endpoint in order
        for i, endpoint in enumerate(self.endpoints):
            try:
                result = self._solve_with_endpoint(endpoint, image_path)
                print(f"Successfully solved with endpoint {i + 1}")
                return result
            except Exception as error:
                print(f"Endpoint {i + 1} failed: {error}")
                
                # If this is the last endpoint, raise the error
                if i == len(self.endpoints) - 1:
                    raise Exception("All CAPTCHA solver endpoints failed")
                
                # Continue to next endpoint
                continue
    
    def _solve_with_endpoint(self, endpoint: str, image_path: str) -> Dict[str, Any]:
        """
        Solve a CAPTCHA with a specific endpoint.
        
        Args:
            endpoint: API endpoint URL
            image_path: Path to the CAPTCHA image file
            
        Returns:
            Dictionary with solution and metadata
        """
        with open(image_path, 'rb') as image_file:
            files = {'image': image_file}
            response = requests.post(
                f"{endpoint}/solve",
                files=files,
                timeout=self.timeout
            )
            
            # Raise an exception for bad status codes
            response.raise_for_status()
            
            # Return JSON response
            return response.json()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check with fallback.
        
        Returns:
            Dictionary with health status
            
        Raises:
            Exception: If all health checks fail
        """
        for i, endpoint in enumerate(self.endpoints):
            try:
                response = requests.get(
                    f"{endpoint}/health",
                    timeout=self.timeout
                )
                
                if response.ok:
                    return response.json()
            except Exception as error:
                print(f"Health check failed for endpoint {i + 1}: {error}")
                continue
        
        raise Exception("All health checks failed")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics with fallback.
        
        Returns:
            Dictionary with statistics
            
        Raises:
            Exception: If all stats requests fail
        """
        for i, endpoint in enumerate(self.endpoints):
            try:
                response = requests.get(
                    f"{endpoint}/stats",
                    timeout=self.timeout
                )
                
                if response.ok:
                    return response.json()
            except Exception as error:
                print(f"Stats request failed for endpoint {i + 1}: {error}")
                continue
        
        raise Exception("All stats requests failed")

# Usage example
if __name__ == "__main__":
    solver = CaptchaSolver()
    
    # Health check
    try:
        health = solver.health_check()
        print("Health check:", health)
    except Exception as error:
        print("Health check failed:", error)
    
    # Solve CAPTCHA
    try:
        result = solver.solve_captcha("path/to/captcha.png")
        print("CAPTCHA solution:", result)
        
        if result.get("success"):
            print("Solution:", result["solution"])
            print("Confidence:", result["confidence"])
            print("Time taken:", result["total_time"])
        else:
            print("Failed to solve CAPTCHA")
    except Exception as error:
        print("CAPTCHA solving failed:", error)
```

## Configuration

To use these implementations:

1. Replace the endpoint URLs with your actual deployment URLs:
   - Render.com deployment URL (recommended primary - completely free)
   - Self-hosted runner URL (via ngrok or similar)
   - Other cloud platform URLs (Railway.app, Fly.io, etc.)

2. Adjust timeout values based on your network conditions and requirements:
   - For free tier deployments, use 30-45 seconds to handle cold starts
   - For self-hosted runners, 15 seconds is usually sufficient

3. Add authentication headers if your API requires them

## Free Deployment Options

All client implementations work with these completely free deployment options:

1. **Render.com** ($0/month):
   - Free tier with 750GB bandwidth/month
   - Services sleep after 15 minutes of inactivity
   - Cold start delays of 15-30 seconds after sleep

2. **GitHub Actions with Self-Hosted Runner** ($0/month):
   - Requires keeping your machine on
   - No cold start delays
   - Best performance

3. **Ngrok Free Tier** ($0/month):
   - For exposing self-hosted runners
   - Sufficient for testing and low-volume usage

## Error Handling

The implementations include comprehensive error handling:

1. **Network errors**: Connection timeouts, DNS failures, etc.
2. **HTTP errors**: 4xx and 5xx status codes
3. **Parsing errors**: Invalid JSON responses
4. **Timeouts**: Requests taking too long (especially important for free tier cold starts)

## Performance Considerations

1. **Caching**: Consider caching health check results to avoid repeated requests
2. **Connection pooling**: Use session objects for multiple requests
3. **Async operations**: JavaScript implementation uses async/await for non-blocking operations
4. **Retry logic**: Implement exponential backoff for failed requests if needed
5. **Cold Start Handling**: Increased timeouts for free tier deployments

## Security

1. **HTTPS**: All endpoints should use HTTPS (provided by all free deployment options)
2. **Input validation**: Validate file types and sizes before uploading
3. **Rate limiting**: Implement client-side rate limiting to avoid overwhelming the API
4. **Authentication**: Add API keys or tokens if required by your deployment