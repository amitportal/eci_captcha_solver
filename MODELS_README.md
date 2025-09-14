# Multi-Model CAPTCHA Solver

This updated version of the CAPTCHA solver supports training and using separate models for each preprocessing method.

## Features

1. **Three Preprocessing Methods:**
   - `grayscale`: Basic preprocessing that converts images to grayscale
   - `light_smoothing`: Applies light smoothing to reduce noise while preserving edges
   - `precise_mathematical`: Uses precise mathematical approach for line removal

2. **Separate Model Training:**
   - Each preprocessing method can have its own trained model
   - Models are saved with method-specific suffixes

3. **Automatic Model Selection:**
   - During solving, the appropriate model is automatically loaded based on the selected preprocessing method
   - Fallback to base models if method-specific models are not found

## Usage

### Training All Models

To train models with all three preprocessing methods:

```bash
python train_all_models.py <dataset_dir> [csv_file]
```

This will create three sets of models in the `models/` directory:
- `optimized_crnn_200x80_grayscale.keras` and `optimized_segmentation_200x80_grayscale.keras`
- `optimized_crnn_200x80_light_smoothing.keras` and `optimized_segmentation_200x80_light_smoothing.keras`
- `optimized_crnn_200x80_precise_mathematical.keras` and `optimized_segmentation_200x80_precise_mathematical.keras`

### Training Individual Models

To train a model with a specific preprocessing method:

```bash
python main_solver.py --mode train --dataset-dir <dataset_dir> --preprocessing-method <method>
```

### Solving with Specific Preprocessing

To solve CAPTCHAs using a specific preprocessing method and its corresponding model:

```bash
python main_solver.py --mode solve --image-path <image_path> --preprocessing-method <method>
```

The solver will automatically load the model that was trained with the same preprocessing method.

## Default Behavior

- Default preprocessing method for solving: `light_smoothing`
- If a model specific to the preprocessing method is not found, the system will fallback to the base model (without suffix)

## Model Compatibility

The system maintains backward compatibility with existing models. If you have previously trained models without suffixes, they will be used as fallbacks when method-specific models are not found.


  ✅ Implementation Summary

   1. Model Saving with Preprocessing Suffixes:
      - Modified save_models() to include preprocessing method suffixes
      - Models are now saved as:
        - optimized_crnn_200x80_grayscale.keras
        - optimized_segmentation_200x80_grayscale.keras
        - optimized_crnn_200x80_light_smoothing.keras
        - optimized_segmentation_200x80_light_smoothing.keras
        - optimized_crnn_200x80_precise_mathematical.keras
        - optimized_segmentation_200x80_precise_mathematical.keras

   2. Intelligent Model Loading:
      - Updated load_models() to load method-specific models
      - Added fallback mechanism to base models if method-specific models don't exist
      - Models are automatically selected based on the preprocessing method

   3. Training Integration:
      - Modified train_models() to save models with the correct preprocessing method suffix        
      - Each training session now creates models specific to its preprocessing method

   4. Backward Compatibility:
      - Existing models without suffixes are still loaded as fallbacks
      - System gracefully handles cases where method-specific models don't exist

  📁 Added Files

   1. `train_all_models.py` - Script to train models with all three preprocessing methods
   2. `test_model_loading.py` - Script to verify model loading functionality
   3. `example_solver.py` - Example script showing usage with different preprocessing
      methods
   4. `MODELS_README.md` - Documentation explaining the new functionality

  🚀 Usage

  Train All Models:

   1 python train_all_models.py <dataset_dir> [csv_file]

  Solve with Specific Preprocessing:

   1 python main_solver.py --mode solve --image-path <image_path>
     --preprocessing-method <method>

  Default Behavior:
   - Default preprocessing method for solving: light_smoothing
   - Automatic model selection based on preprocessing method
   - Fallback to base models when needed

  The system now supports training and using separate models optimized for each
  preprocessing method, which should improve overall accuracy by matching the right model
  to the right preprocessing approach.
