"""
Detection script for dental X-ray analysis
"""
from ultralytics import YOLO
import cv2
import argparse

def detect(image_path, model_path='models/best.pt', conf=0.5):
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(image_path, conf=conf)
    
    # Display results
    for result in results:
        result.show()
        result.save(filename='result.jpg')
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to X-ray image')
    parser.add_argument('--model', type=str, default='models/best.pt', help='Path to model')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    
    args = parser.parse_args()
    detect(args.image, args.model, args.conf)