"""
Configuration file for dental X-ray detection
"""

# Model configuration
MODEL_CONFIG = {
    'model_name': 'yolov8n.pt',
    'imgsz': 640,
    'conf_threshold': 0.5,
}

# Training configuration
TRAIN_CONFIG = {
    'epochs': 100,
    'batch_size': 16,
    'learning_rate': 0.01,
    'patience': 50,
}

# Data paths
DATA_PATHS = {
    'train': 'data/train',
    'val': 'data/valid',
    'test': 'data/test',
}

# Class names
CLASSES = ['implant', 'filling', 'root_canal']
```

---

### 3️⃣ Notebook Dosyaları

Eğer Colab'da çalıştıysan, **Colab notebook'u indir** ve ekle:
```
notebooks/
├── training.ipynb           # Colab'daki eğitim notebook'u
├── evaluation.ipynb         # Test notebook'u (isteğe bağlı)
└── data_preprocessing.ipynb # Veri hazırlama (isteğe bağlı)