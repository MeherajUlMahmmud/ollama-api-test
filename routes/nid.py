import json
import uuid
from datetime import datetime, timezone

from flask import Blueprint, request

from core.services.nid import NIDDataExtractor
from logger import Logger
from token_auth import token_required
from utils import Helper
from validators import validate_nid_upload

nid_bp = Blueprint('nid', __name__)
logger = Logger.get_logger()

nid_extractor = NIDDataExtractor()


@nid_bp.route('/nid-ocr', methods=['POST'])
@token_required
def nid_ocr():
    start_time = datetime.now(timezone.utc)
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    time_str = datetime.now(timezone.utc).strftime("%H_%M_%S_%f")
    trace_id = request.environ.get('trace_id')

    try:
        multipart_data = request.files  # Get the files from the request
        # errors = validate_nid_upload(multipart_data)
        # if errors:
        #     return Helper.api_response(
        #         message='Validation failed',
        #         start_time=start_time,
        #         is_success=False,
        #         data={'errors': errors},
        #         status_code=400
        #     )

        front_image = multipart_data['front_image']
        back_image = multipart_data['back_image']

        uploaded_dir = Helper.get_date_trace_path(trace_id, date_str, time_str, "uploaded")
        processed_dir = Helper.get_date_trace_path(trace_id, date_str, time_str, "processed")
        uploaded_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        front_filename = f"{front_image.filename.rsplit('.', 1)[0]}_{uuid.uuid4().hex}.{front_image.filename.rsplit('.', 1)[-1]}"
        back_filename = f"{back_image.filename.rsplit('.', 1)[0]}_{uuid.uuid4().hex}.{back_image.filename.rsplit('.', 1)[-1]}"
        front_image_path = uploaded_dir / front_filename
        back_image_path = uploaded_dir / back_filename

        try:
            front_image.save(front_image_path)
            back_image.save(back_image_path)
            logger.info(f"Saved images: front={front_image_path}, back={back_image_path}")

            nid_data = nid_extractor.extract_complete_nid_data(
                str(front_image_path),
                str(back_image_path),
                processed_dir
            )

            response_data = {
                "data": json.loads(nid_data.to_json()),
                "is_valid": nid_data.is_valid()
            }

            return Helper.api_response(
                message="NID data extraction completed successfully",
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
        logger.error(f"NID OCR failed: {str(e)}", exc_info=True)
        return Helper.api_response(
            message=f"NID data extraction failed: {str(e)}",
            start_time=start_time,
            is_success=False,
            status_code=500
        )
