# STATS-507---Coursework

# SCANNER: Skin Cancer Analysis Neural Network for Early Recognition

A deep learning system for multi-class skin lesion classification using Explainable AI (XAI) for clinical transparency.

## Project Overview

SCANNER is an advanced deep learning model that classifies dermatoscopic images into 7 types of skin lesions with integrated explainable AI components for clinical interpretability.

## Classification Categories

The model classifies skin lesions into 7 diagnostic categories:
- Actinic Keratosis (AKIEC)
- Basal Cell Carcinoma (BCC) 
- Benign Keratosis (BKL)
- Dermatofibroma (DF)
- Melanoma (MEL)
- Melanocytic Nevus (NV)
- Vascular Lesion (VASC)

## Data Processing
- Dataset: HAM10000 (5,000 images used)
- Image Size: 224×224 pixels
- Data Balancing: Strategic oversampling (rare classes) + undersampling (dominant classes)
- Augmentation: Rotation (40°), Shifting (25%), Zooming (30%), Flipping, Brightness adjustment

## Key Features:
- Multi-class skin lesion classification 
- ResNet50-based architecture with custom classification head
- Comprehensive data balancing strategies
- Explainable AI with saliency mapping and lesion boundary detection
- Clinical-grade visualization and interpretability

## Performance

- Test Accuracy: 69.60%
- Validation Accuracy: 78.96%
