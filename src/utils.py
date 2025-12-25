"""
Utility functions for dental X-ray detection
"""
import cv2
import numpy as np
from pathlib import Path

def load_image(image_path):
    """Load and preprocess image"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return img

def save_results(results, output_dir='results/predictions'):
    """Save detection results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, result in enumerate(results):
        result.save(filename=str(output_dir / f'prediction_{i}.jpg'))

def calculate_metrics(results):
    """Calculate detection metrics"""
    # Implement your metrics calculation
    pass