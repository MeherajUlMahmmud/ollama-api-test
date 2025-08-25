import uuid

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from flask_httpauth import HTTPBasicAuth

from configs.api_routes import Route
from configs.basic_auth import verify_user
from configs.config import Config
from logger import Logger, trace_id_context

# Initialize Flask application
app = Flask(__name__)

# Configure CORS
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

# Initialize HTTP basic authentication
auth = HTTPBasicAuth()
cors = CORS(app)

logger = Logger.get_logger()


@app.before_request
def before_request():
    """
    Set up request tracing before each request.

    This function generates a unique trace ID for each incoming request
    and sets it in the context for logging purposes.
    """
    trace_id = str(uuid.uuid4())
    trace_id_context.set(trace_id)


@app.after_request
def after_request(response):
    """
    Clean up request context after each request.

    This function clears the trace ID context after the request
    has been processed to prevent memory leaks.

    Args:
        response: The Flask response object.

    Returns:
        The Flask response object.
    """
    trace_id_context.set(None)
    return response


@app.route(Route.INDEX_ROUTE, methods=["POST"])
@cross_origin()
@verify_user
def index():
    return jsonify({'message': 'Welcome to the API', 'status': 'running'})


@app.route(Route.NID_OCR_ROUTE, methods=["POST"])
@cross_origin()
@verify_user
def nid_data_extraction():
    from pathlib import Path
    import tempfile
    import datetime
    import random
    import string
    from core.services.nid import NIDDataExtractor, NIDSide

    # Check what files are present in the request
    front_file = request.files.get('front_img')
    back_file = request.files.get('back_img')

    # Get side parameter for single side extraction
    side = request.form.get('side', '').lower()

    # Validate side parameter
    if side not in ['front', 'back', 'both']:
        response = {'error': 'Side must be either "front", "back", or "both"'}
        return jsonify(response), 400

    def generate_unique_filename(prefix="img"):
        """Generate a unique filename with datetime and random string"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        return f"{prefix}_{timestamp}_{random_str}.jpg"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create date-wise folder structure
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        date_folder = tmpdir_path / current_date
        date_folder.mkdir(exist_ok=True)

        extractor = NIDDataExtractor()

        # If both images are provided and side is 'both', do complete extraction
        if front_file and back_file and side == 'both':
            front_filename = generate_unique_filename("front")
            back_filename = generate_unique_filename("back")
            front_path = date_folder / front_filename
            back_path = date_folder / back_filename
            front_file.save(str(front_path))
            back_file.save(str(back_path))

            # Extract complete NID data
            nid_data = extractor.extract_complete_nid_data(
                str(front_path),
                str(back_path),
                date_folder
            )

            response = {
                "success": True,
                "extraction_type": "complete",
                "message": "Complete NID data extracted from both sides",
                "data": nid_data.__dict__
            }

        # If side is 'front' and front image is provided
        elif side == 'front' and front_file:
            front_filename = generate_unique_filename("front")
            image_path = date_folder / front_filename
            front_file.save(str(image_path))

            # Extract data from front side
            extracted_data = extractor.extract_from_single_side(
                str(image_path),
                NIDSide.FRONT,
                date_folder
            )

            response = {
                "success": True,
                "extraction_type": "single",
                "side": "front",
                "message": "Data extracted from front side",
                "data": extracted_data
            }

        # If side is 'back' and back image is provided
        elif side == 'back' and back_file:
            back_filename = generate_unique_filename("back")
            image_path = date_folder / back_filename
            back_file.save(str(image_path))

            # Extract data from back side
            extracted_data = extractor.extract_from_single_side(
                str(image_path),
                NIDSide.BACK,
                date_folder
            )

            response = {
                "success": True,
                "extraction_type": "single",
                "side": "back",
                "message": "Data extracted from back side",
                "data": extracted_data
            }

        # If side is 'both' but only one image is provided
        elif side == 'both' and (front_file or back_file):
            response = {
                'error': 'Both front_img and back_img are required when side is "both"',
                'usage': {
                    'complete_extraction': 'Send both front_img and back_img with side="both"',
                    'single_extraction': 'Send either front_img or back_img with side="front" or side="back"'
                }
            }
            return jsonify(response), 400

        else:
            # Handle cases where the requested side doesn't match the provided image
            if side == 'front' and not front_file:
                response = {
                    'error': 'Front image (front_img) is required when side is "front"',
                    'usage': {
                        'complete_extraction': 'Send both front_img and back_img with side="both"',
                        'single_extraction': 'Send front_img with side="front" or back_img with side="back"'
                    }
                }
            elif side == 'back' and not back_file:
                response = {
                    'error': 'Back image (back_img) is required when side is "back"',
                    'usage': {
                        'complete_extraction': 'Send both front_img and back_img with side="both"',
                        'single_extraction': 'Send front_img with side="front" or back_img with side="back"'
                    }
                }
            else:
                response = {
                    'error': 'At least one image file (front_img or back_img) is required.',
                    'usage': {
                        'complete_extraction': 'Send both front_img and back_img with side="both"',
                        'single_extraction': 'Send front_img with side="front" or back_img with side="back"'
                    }
                }
            return jsonify(response), 400

    return jsonify(response)


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'message': 'Service is running'})


if __name__ == '__main__':
    print("Starting Flask application...")
    app.run(host='0.0.0.0', port=8000, debug=True)
