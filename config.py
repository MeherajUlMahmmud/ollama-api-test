import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
PROCESSED_DIR = BASE_DIR / "static" / "processed"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# LLM configuration
LLM_CONFIG = {
    "base_url": "http://localhost:11434/api/generate",
    "model": "qwen2.5vl:7b",
    "timeout": 300,
    "options": {
        "temperature": 0.1,
        "top_p": 0.9,
        "num_predict": 1500,
        "repeat_penalty": 1.1,
    },
}

# Image processing configuration
IMAGE_CONFIG = {
    "max_size": 1024,
    "min_size": 400,
    "contrast_factor": 1.2,
    "sharpness_factor": 1.1,
    "brightness_factor": 1.02,
    "jpeg_quality": 95,
    "face_padding_ratio": 0.3,
    "card_padding": 10,
}

# Decision metrics
DECISION_METRICS = {
    "liveness_confidence_threshold": 70.0,
    "nid_validation_fields": ["english_name", "nid_no", "date_of_birth"],
}