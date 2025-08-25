import os

from configs.api_routes import Route


class Config:
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    DEBUG = os.environ.get('FLASK_ENV') == 'development'

    # Upload configuration
    UPLOAD_FOLDER: str = "static/uploads/"
    HOST: str = "0.0.0.0"
    PORT: int = 5000
    MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20MB
    ALLOWED_EXTENSIONS: set = {"png", "jpg", "jpeg"}
    FILE_CLEANING_INTERVAL: int = 5  # 5 days
    API_DICT = {
        Route.INDEX_ROUTE: "API Index",
        Route.NID_OCR_ROUTE: "OCR NID",
    }


class ImageConfig:
    MAX_SIZE: int = int(os.environ.get('IMAGE_MAX_SIZE', 1024))
    MIN_SIZE: int = int(os.environ.get('IMAGE_MIN_SIZE', 400))
    CONTRAST_FACTOR: int = float(os.environ.get('IMAGE_CONTRAST_FACTOR', 1.2))
    SHARPNESS_FACTOR: int = float(os.environ.get('IMAGE_SHARPNESS_FACTOR', 1.1))
    BRIGHTNESS_FACTOR: int = float(os.environ.get('IMAGE_BRIGHTNESS_FACTOR', 1.02))
    JPEG_QUALITY: int = int(os.environ.get('IMAGE_JPEG_QUALITY', 400))


class LLMConfig:
    BASE_URL: str = os.environ.get('LLM_BASE_URL') or "http://localhost:11434/api/generate"


# OCR
class OCRConfig:
    # Image processing configuration
    IMAGE_CONFIG = {
        "max_size": ImageConfig.MAX_SIZE,
        "min_size": ImageConfig.MIN_SIZE,
        "contrast_factor": ImageConfig.CONTRAST_FACTOR,
        "sharpness_factor": ImageConfig.SHARPNESS_FACTOR,
        "brightness_factor": ImageConfig.BRIGHTNESS_FACTOR,
        "jpeg_quality": ImageConfig.JPEG_QUALITY,
        "card_padding": int(os.environ.get('IMAGE_CARD_PADDING', 10)),
    }

    # LLM configuration
    LLM_CONFIG = {
        "base_url": LLMConfig.BASE_URL,
        "model": "qwen2.5vl:7b",
        "timeout": int(os.environ.get('LLM_TIMEOUT', 300)),
        "options": {
            "temperature": float(os.environ.get('LLM_TEMPERATURE', 0.1)),
            "top_p": float(os.environ.get('LLM_TOP_P', 0.9)),
            "num_predict": int(os.environ.get('LLM_NUM_PREDICT', 5000)),
            "repeat_penalty": float(os.environ.get('LLM_REPEAT_PENALTY', 1.1)),
        },
    }

    # Decision metrics
    DECISION_METRICS = {
        "nid_validation_fields": os.environ.get('NID_VALIDATION_FIELDS', "english_name,nid_no,date_of_birth").split(
            ','),
    }

    # Original OCR configuration
    DOB_FORMAT = "%d-%b-%Y"
    DOB_FORMAT_PATTERN = r'^\d{2}-[A-Za-z]{3}-\d{4}$'


class ErrorMessage:
    INVALID_FILE_ERROR_MESSAGE = "Invalid file. Please upload a valid image file."
    FILE_NOT_UPLOADED_ERROR_MESSAGE = "File not uploaded. Please upload a valid image file."
    GENERIC_ERROR_MESSAGE = "Unable to process your request at this moment. Please try again."
    GENERIC_SUCCESS_MESSAGE = "Operation successful"
    UNAUTHORIZED_ACCESS_ERROR_MESSAGE = "Unauthorized access."

    FLD_RESPONSE_SUCCESS_MESSAGE = "Real image detection successful."
    FLD_RESPONSE_ERROR_MESSAGE = "Please adjust your camera position and try again."
    FACE_IS_NOT_VISIBLE_MESSAGE = "Face is not clearly visible."
    FACE_IS_NOT_STRAIGHT = "Face is not straight. Please keep the face straight to the camera."
    FLD_RESPONSE_NO_FACE_MESSAGE = "No face detected. Make sure the face is clearly visible and facing the camera."
    FLD_RESPONSE_MULTI_FACE_MESSAGE = "Multiple faces detected. Make sure only one face is visible and facing the camera."

    OCR_RESPONSE_SUCCESS_MESSAGE = "OCR Performed Successfully."
    FETCH_FROM_EC_MESSAGE = "Fetch data from EC."
    OCR_RESPONSE_ERROR_MESSAGE = "Unable to perform OCR. Please try again."
    REF_NO_ERROR_MESSAGE = "Reference number not found. Please try again with a valid reference number."
    NID_NO_ERROR_MESSAGE = "NID number not found. Please try again with a valid NID number."
    DATE_OF_BIRTH_ERROR_MESSAGE = "Date of Birth not found. Please try again with a valid Date of Birth (e.g. 28-Jan-1983)."
    INVALID_BACK_IMAGE_ERROR_MESSAGE = "Invalid NID back image. Please upload a valid NID back image."
    ADDRESS_ERROR_MESSAGE = "Failed to extract address. Please try again."


class QueryConfig:
    # LLM configuration
    LLM_CONFIG = {
        "base_url": LLMConfig.BASE_URL,
        "model": "llama3.1:8b",
        "timeout": int(os.environ.get('LLM_TIMEOUT', 300)),
        "options": {
            "temperature": float(os.environ.get('LLM_TEMPERATURE', 0.1)),
            "top_p": float(os.environ.get('LLM_TOP_P', 0.9)),
            "num_predict": int(os.environ.get('LLM_NUM_PREDICT', 5000)),
            "repeat_penalty": float(os.environ.get('LLM_REPEAT_PENALTY', 1.1)),
        },
    }
