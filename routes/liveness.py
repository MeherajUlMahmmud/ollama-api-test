import json
import uuid
from datetime import datetime, timezone

from flask import Blueprint, request

from core.services.liveness import FaceLivenessDetector
from logger import Logger
from token_auth import token_required
from utils import Helper
from validators import validate_liveness_upload

liveness_bp = Blueprint('liveness', __name__)
logger = Logger.get_logger()

liveness_detector = FaceLivenessDetector()


@liveness_bp.route('/liveness-check', methods=['POST'])
@token_required
def liveness_check():
    start_time = datetime.now(timezone.utc)
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    time_str = datetime.now(timezone.utc).strftime("%H_%M_%S_%f")
    trace_id = request.environ.get('trace_id')

    try:
        errors = validate_liveness_upload(request.files)
        if errors:
            return Helper.api_response(
                message='Validation failed',
                start_time=start_time,
                is_success=False,
                data={'errors': errors},
                status_code=400
            )

        image = request.files['image']
        uploaded_dir = Helper.get_date_trace_path(trace_id, date_str, time_str, "uploaded")
        processed_dir = Helper.get_date_trace_path(trace_id, date_str, time_str, "processed")
        uploaded_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{image.filename.rsplit('.', 1)[0]}_{uuid.uuid4().hex}.{image.filename.rsplit('.', 1)[-1]}"
        image_path = uploaded_dir / filename

        try:
            image.save(image_path)
            logger.info(f"Saved image: {image_path}")

            liveness_result = liveness_detector.detect_liveness(str(image_path), processed_dir)
            response_data = {
                "data": json.loads(liveness_result.to_json()),
                "is_reliable": liveness_result.is_reliable()
            }

            return Helper.api_response(
                message="Liveness check completed successfully",
                start_time=start_time,
                is_success=True,
                data=response_data
            )

        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}", exc_info=True)
            return Helper.api_response(
                message=f"Image processing failed: {str(e)}",
                start_time=start_time,
                is_success=False,
                status_code=500
            )

    except Exception as e:
        logger.error(f"Liveness check failed: {str(e)}", exc_info=True)
        return Helper.api_response(
            message=f"Liveness detection failed: {str(e)}",
            start_time=start_time,
            is_success=False,
            status_code=500
        )
