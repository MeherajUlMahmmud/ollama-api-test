from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# LLM configuration
LLM_CONFIG = {
    "base_url": "http://localhost:11434/api/generate",
    "model": "qwen2.5vl:7b",
    "timeout": 300,
    "options": {
        "temperature": 0.1,     # Controls the randomness/creativity of the output. Lower values (e.g., 0.1) make the output more deterministic and focused, picking the most probable tokens. Higher values lead to more diverse and surprising output.
        "top_p": 0.9,           # Controls diversity by selecting tokens from a dynamic set whose cumulative probability exceeds this value (nucleus sampling). A top_p of 0.9 means the model considers tokens that make up the top 90% of the cumulative probability.
        "num_predict": 5000,    # Maximum number of tokens the model should generate in a single response. This limits the length of the output.
        "repeat_penalty": 1.1,  # Penalizes the model for repeating tokens or sequences of tokens that have already appeared. A value greater than 1.0 discourages repetition.
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
