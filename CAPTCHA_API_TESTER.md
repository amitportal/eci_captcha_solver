# CAPTCHA API Tester

This script tests the ECI CAPTCHA Solver API using the `test_captcha_2.png` file that is included in the repository.

## Usage

```bash
python captcha_api_tester.py <api_url>
```

Example:
```bash
python captcha_api_tester.py https://your-api.onrender.com
```

## What the script tests

1. **Health endpoint** - Verifies the API is running
2. **Stats endpoint** - Retrieves current statistics
3. **Solve endpoint** - Sends `test_captcha_2.png` to be solved
4. **Stats endpoint again** - Verifies statistics were updated

## Expected Output

When successful, the script will show:
- Health check status
- Current statistics
- CAPTCHA solution with confidence score
- Updated statistics

## Error Handling

The script handles common issues:
- Connection timeouts (especially important for free tier cold starts)
- Connection errors
- API errors with descriptive messages

## Requirements

- Python 3.6 or higher
- requests library (`pip install requests`)