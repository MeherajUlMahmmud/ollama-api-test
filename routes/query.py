from datetime import datetime, timezone

from flask import Blueprint, request

from core.common.query_analyzer import QueryAnalyzer
from core.common.query_executor import QueryExecutor
from core.query_handler import QueryHandler
from logger import Logger
from token_auth import token_required
from utils import Helper
from validators import validate_query

query_bp = Blueprint('query', __name__)
logger = Logger.get_logger()

query_analyzer = QueryAnalyzer()
query_executor = QueryExecutor()
query_handler = QueryHandler(query_analyzer=query_analyzer, query_executor=query_executor)


@query_bp.route('/query', methods=['POST'])
@token_required
def query():
    """
    Handle incoming query requests via POST method.

    This endpoint receives a JSON payload containing a query string, validates it,
    processes the query using the QueryHandler, and returns the response. It requires
    a valid JWT access token in the Authorization header.

    Returns:
        JSON response containing:
        - message: A descriptive message about the request outcome
        - is_success: Boolean indicating success or failure
        - data: Query response data or validation errors
        - status_code: HTTP status code (200, 400, or 500)
        - execution_time: Time taken to process the request in milliseconds

    Raises:
        Exception: If query processing fails, returns a 500 error with the exception message.
    """
    start_time = datetime.now(timezone.utc)
    logger.info(f"Received POST request to /query endpoint from user_id: {request.user_id}")

    try:
        logger.debug("Parsing JSON payload from request")
        data = request.get_json()
        if not data:
            logger.warning("No JSON payload provided in request")
            return Helper.api_response(
                message='No JSON payload provided',
                start_time=start_time,
                is_success=False,
                data={'errors': ['No JSON payload provided']},
                status_code=400
            )

        logger.debug(f"Validating query data: {data}")
        errors = validate_query(data)
        if errors:
            logger.warning(f"Query validation failed: {errors}")
            return Helper.api_response(
                message='Validation failed',
                start_time=start_time,
                is_success=False,
                data={'errors': errors},
                status_code=400
            )

        query_text = data['query']
        logger.info(f"Processing query: {query_text}")

        logger.debug("Initiating query handling via QueryHandler")
        response = query_handler.handle_query(query_text)
        logger.info(f"Query processed successfully: {query_text}")

        response_data = {
            "query": query_text,
            "response": response
        }

        logger.debug("Preparing successful API response")
        return Helper.api_response(
            message="Query processed successfully",
            start_time=start_time,
            is_success=True,
            data=response_data
        )

    except Exception as e:
        logger.error(f"Query processing failed for query: {data.get('query', 'unknown')} - Error: {str(e)}",
                     exc_info=True)
        return Helper.api_response(
            message=f"Query processing failed: {str(e)}",
            start_time=start_time,
            is_success=False,
            status_code=500
        )
