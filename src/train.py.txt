"""
Training script for YOLOv8 dental X-ray detection
"""
from ultralytics import YOLO

def train_model():
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # Train the model
    results = model.train(
        data='data/dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='dental_xray_detection',
        patience=50,
        save=True,
        device=0  # GPU
    )
    
    return results

if __name__ == "__main__":
    train_model()