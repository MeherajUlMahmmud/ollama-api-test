import json
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify

from config import UPLOAD_DIR
from core.liveness import FaceLivenessDetector
from core.nid import NIDDataExtractor
from logger import Logger, trace_id_context
from utils import Helper

logger = Logger.get_logger()

app = Flask(__name__)

# Load models at startup
nid_extractor = NIDDataExtractor()
liveness_detector = FaceLivenessDetector()


def get_date_trace_path(trace_id: str, date_str: str, time_str: str, subfolder: str) -> Path:
    """Generate path with date and trace_id structure."""
    return UPLOAD_DIR / date_str / f"{time_str}_{trace_id}" / subfolder


@app.before_request
def before_request():
    """Set a unique trace ID for each request."""
    trace_id = str(uuid.uuid4())
    trace_id_context.set(trace_id)
    logger.info(f"Starting request with trace_id: {trace_id}")


@app.after_request
def after_request(response):
    """Clear trace ID after request completion."""
    logger.info(f"Request completed with trace_id: {trace_id_context.get()}")
    trace_id_context.set(None)
    return response


@app.route('/api/nid-ocr', methods=['POST'])
def nid_ocr():
    """Extract data from NID card images."""
    start_time = datetime.now()
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H_%M_%S_%f")
    trace_id = trace_id_context.get()
    logger.info(f"Processing NID OCR request")

    # Validate input files
    front_image = request.files.get('front_image')
    back_image = request.files.get('back_image')

    try:
        # Create directories for uploaded and processed images
        uploaded_dir = get_date_trace_path(trace_id, date_str, time_str, "uploaded")
        processed_dir = get_date_trace_path(trace_id, date_str, time_str, "processed")
        uploaded_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filenames
        front_filename = f"{front_image.filename.rsplit('.', 1)[0]}_{uuid.uuid4().hex}.{front_image.filename.rsplit('.', 1)[-1]}"
        back_filename = f"{back_image.filename.rsplit('.', 1)[0]}_{uuid.uuid4().hex}.{back_image.filename.rsplit('.', 1)[-1]}"
        front_image_path = uploaded_dir / front_filename
        back_image_path = uploaded_dir / back_filename

        # Save uploaded images
        front_image.save(front_image_path)
        back_image.save(back_image_path)
        logger.info(f"Saved images: front={front_image_path}, back={back_image_path}")

        # Extract NID data
        logger.info(f"Starting NID data extraction")
        nid_data = nid_extractor.extract_complete_nid_data(
            str(front_image_path),
            str(back_image_path),
            processed_dir,
        )
        logger.info(f"NID data extraction completed")

        # Prepare response
        response_data = {
            "message": "NID data extracted successfully.",
            "is_success": True,
            "time_taken": Helper.format_time_taken(start_time),
            "data": json.loads(nid_data.to_json()),
            "is_valid": nid_data.is_valid(),
        }

        logger.info(f"NID OCR request completed successfully")
        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"NID OCR failed: {str(e)}", exc_info=True)
        return Helper.error_response(
            message=f"NID data extraction failed: {str(e)}",
            start_time=start_time,
            status_code=500
        )


@app.route('/api/liveness-check', methods=['POST'])
def liveness_check():
    """Perform face liveness detection on an image."""
    start_time = datetime.now()
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H_%M_%S_%f")
    trace_id = trace_id_context.get()
    logger.info(f"Processing liveness check request")

    # Validate input file
    image = request.files.get('image')

    try:
        # Create directory for uploaded image
        uploaded_dir = get_date_trace_path(trace_id, date_str, time_str, "uploaded")
        processed_dir = get_date_trace_path(trace_id, date_str, time_str, "processed")
        uploaded_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        filename = f"{image.filename.rsplit('.', 1)[0]}_{uuid.uuid4().hex}.{image.filename.rsplit('.', 1)[-1]}"
        image_path = uploaded_dir / filename

        # Save uploaded image
        image.save(image_path)
        logger.info(f"Saved image: {image_path}")

        # Perform liveness detection
        logger.info(f"Starting liveness detection")
        liveness_result = liveness_detector.detect_liveness(str(image_path), processed_dir)
        logger.info(
            f"Liveness detection completed: result={liveness_result.result.value}, confidence={liveness_result.confidence}")

        # Prepare response
        response_data = {
            "message": "Liveness check completed successfully.",
            "is_success": True,
            "time_taken": Helper.format_time_taken(start_time),
            "data": json.loads(liveness_result.to_json()),
            "is_reliable": liveness_result.is_reliable(),
        }

        logger.info(f"Liveness check request completed successfully")
        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Liveness check failed: {str(e)}", exc_info=True)
        return Helper.error_response(
            message=f"Liveness detection failed: {str(e)}",
            start_time=start_time,
            status_code=500
        )


if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True)
