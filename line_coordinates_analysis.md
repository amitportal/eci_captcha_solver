# CAPTCHA LINE COORDINATES ANALYSIS REPORT

## Image Dimensions
- Width: 200 pixels
- Height: 80 pixels

## HORIZONTAL LINES

### Yellow Line
- **Peak position**: Row 30
- **Buffer zone**: Rows 28-32 (y=30±2)
- **Pixel count in zone**: 47 pixels
- **Strongest concentration**: Row 30 (30 pixels)

### Cyan Line
- **Peak position**: Row 40
- **Buffer zone**: Rows 38-42 (y=40±2)
- **Pixel count in zone**: 296 pixels
- **Strongest concentration**: Row 40 (150 pixels)

### Pink Line
- **Peak position**: Row 50
- **Buffer zone**: Rows 48-52 (y=50±2)
- **Pixel count in zone**: 202 pixels
- **Strongest concentration**: Row 50 (169 pixels)

## DIAGONAL LINE

### Theoretical (Expected)
- **Equation**: y = -0.397x + 79
- **From**: (0,79) to (199,0)
- **Angle**: 21.7 degrees

### Actual (Measured)
- **Equation**: y = -0.136231x + 54.8866
- **Angle**: 7.7577 degrees
- **Pixel count**: 363 green pixels on this line

## RECOMMENDED PROCESSING COORDINATES

For optimal line removal, process these regions:

1. **Yellow Line Region**: Process rows 28-32 (y=30±2)
2. **Cyan Line Region**: Process rows 38-42 (y=40±2)
3. **Pink Line Region**: Process rows 48-52 (y=50±2)
4. **Diagonal Line**: Use equation y = -0.136231x + 54.8866 with ±2 pixel buffer

## COLOR THRESHOLDS (HSV)

### Yellow Line Detection
- Lower: [10, 30, 100]
- Upper: [50, 255, 255]

### Cyan Line Detection
- Lower: [80, 30, 100]
- Upper: [100, 255, 255]

### Pink Line Detection
- Lower: [140, 30, 100]
- Upper: [170, 255, 255]

### Green Text Detection
- Lower: [35, 20, 20]
- Upper: [90, 255, 255]

## PROCESSING STRATEGY

1. Detect all colored pixels using the above thresholds
2. Remove pixels in horizontal line regions (with 2-pixel buffer)
3. Remove pixels along the diagonal line (using actual equation with 2-pixel buffer)
4. For each removed pixel, replace with nearest green text pixel or white background