import os
from pathlib import Path

class Config:
    # Base directories
    BASE_DIR = Path(__file__).resolve().parent
    UPLOAD_DIR = BASE_DIR / "static" / "uploads"

    # Ensure directories exist
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    JWT_SECRET = os.environ.get('SECRET_KEY') or 'your-jwt-secret-key-here'
    JWT_ALGORITHM = 'HS256'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///ollama_api.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    DEBUG = os.environ.get('FLASK_ENV') == 'development'

# LLM configuration
LLM_CONFIG = {
    "base_url": os.environ.get('LLM_BASE_URL') or "http://localhost:11434/api/generate",
    "model": os.environ.get('LLM_MODEL') or "qwen2.5vl:7b",
    "timeout": int(os.environ.get('LLM_TIMEOUT', 300)),
    "options": {
        "temperature": float(os.environ.get('LLM_TEMPERATURE', 0.1)),
        "top_p": float(os.environ.get('LLM_TOP_P', 0.9)),
        "num_predict": int(os.environ.get('LLM_NUM_PREDICT', 5000)),
        "repeat_penalty": float(os.environ.get('LLM_REPEAT_PENALTY', 1.1)),
    },
}

# Image processing configuration
IMAGE_CONFIG = {
    "max_size": int(os.environ.get('IMAGE_MAX_SIZE', 1024)),
    "min_size": int(os.environ.get('IMAGE_MIN_SIZE', 400)),
    "contrast_factor": float(os.environ.get('IMAGE_CONTRAST_FACTOR', 1.2)),
    "sharpness_factor": float(os.environ.get('IMAGE_SHARPNESS_FACTOR', 1.1)),
    "brightness_factor": float(os.environ.get('IMAGE_BRIGHTNESS_FACTOR', 1.02)),
    "jpeg_quality": int(os.environ.get('IMAGE_JPEG_QUALITY', 95)),
    "face_padding_ratio": float(os.environ.get('IMAGE_FACE_PADDING_RATIO', 0.3)),
    "card_padding": int(os.environ.get('IMAGE_CARD_PADDING', 10)),
}

# Decision metrics
DECISION_METRICS = {
    "liveness_confidence_threshold": float(os.environ.get('LIVENESS_CONFIDENCE_THRESHOLD', 70.0)),
    "nid_validation_fields": os.environ.get('NID_VALIDATION_FIELDS', "english_name,nid_no,date_of_birth").split(','),
}
