# ULTRA-PRECISE CAPTCHA LINE COORDINATES ANALYSIS REPORT

## Image Dimensions
- Width: 200 pixels
- Height: 80 pixels

## HORIZONTAL LINES

### Yellow Line
- **Position**: Row 30
- **Processing Region**: Rows 28-32 (y=30±2)
- **Pixels to Remove**: 47 pixels
- **Color Range (HSV)**: [10, 30, 100] to [50, 255, 255]

### Cyan Line
- **Position**: Row 40
- **Processing Region**: Rows 38-42 (y=40±2)
- **Pixels to Remove**: 296 pixels
- **Color Range (HSV)**: [80, 30, 100] to [100, 255, 255]

### Pink Line
- **Position**: Row 50
- **Processing Region**: Rows 48-52 (y=50±2)
- **Pixels to Remove**: 202 pixels
- **Color Range (HSV)**: [140, 30, 100] to [170, 255, 255]

## DIAGONAL LINE

### Specifications
- **Equation**: y = 79 - 0.393910x
- **Angle**: 21.5 degrees (within expected 20-23° range)
- **Thickness**: 5 pixels (±2 pixels from center line)
- **Pixels to Remove**: 550 green pixels

### Overlap Analysis
- **Pure Diagonal Pixels** (not in horizontal regions): 436 pixels
- **Diagonal Pixels in Horizontal Regions**: 114 pixels
  - Yellow region: ~50 pixels
  - Cyan region: ~54 pixels
  - Pink region: ~10 pixels

## PROCESSING SUMMARY

### Total Pixels Removed
- **Horizontal Lines**: 545 pixels
- **Diagonal Line**: 550 pixels
- **Total**: 1,072 pixels

### Processing Method
- **Pixels Filled** (replaced with nearby text color): 1,051 pixels
- **Pixels Erased** (set to white background): 21 pixels

## COLOR THRESHOLDS (HSV)

### Green Text Detection
- Lower: [35, 20, 20]
- Upper: [90, 255, 255]

### Yellow Line Detection
- Lower: [10, 30, 100]
- Upper: [50, 255, 255]

### Cyan Line Detection
- Lower: [80, 30, 100]
- Upper: [100, 255, 255]

### Pink Line Detection
- Lower: [140, 30, 100]
- Upper: [170, 255, 255]

## IMPLEMENTATION COORDINATES

For precise line removal, use these coordinates:

1. **Yellow Line Region**: Process rows 28-32
2. **Cyan Line Region**: Process rows 38-42
3. **Pink Line Region**: Process rows 48-52
4. **Diagonal Line**: Use equation y = 79 - 0.393910x with ±2 pixel buffer

This ultra-precise approach ensures complete removal of all lines while preserving the green text for optimal OCR processing.