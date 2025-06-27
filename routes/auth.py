from datetime import datetime, timezone

from flask import Blueprint, request

from logger import Logger
from models.user import db, User
from token_auth import TokenAuth, token_required, blacklisted_tokens
from utils import Helper
from validators import validate_registration, validate_login

auth_bp = Blueprint('auth', __name__)
logger = Logger.get_logger()


@auth_bp.route('/register', methods=['POST'])
def register():
    """
    Handle user registration via POST request.

    This endpoint accepts a JSON payload with username and password, validates the input,
    checks for existing users, and creates a new user in the database if valid.

    Returns:
        JSON response containing:
        - message: A descriptive message about the request outcome
        - is_success: Boolean indicating success or failure
        - data: Validation errors (if any)
        - status_code: HTTP status code (201, 400, 409, or 500)
        - execution_time: Time taken to process the request in milliseconds

    Raises:
        Exception: If registration fails (e.g., database error), rolls back the transaction
                   and returns a 500 error with the exception message.
    """
    start_time = datetime.now(timezone.utc)
    logger.info("Received POST request to /register endpoint")

    data = request.get_json()

    try:
        logger.debug("Parsing JSON payload for registration")
        if not data:
            logger.warning("No JSON payload provided in registration request")
            return Helper.api_response(
                message='No JSON payload provided',
                start_time=start_time,
                is_success=False,
                data={'errors': ['No JSON payload provided']},
                status_code=400
            )

        logger.debug(f"Validating registration data: {data}")
        errors = validate_registration(data)
        if errors:
            logger.warning(f"Registration validation failed: {errors}")
            return Helper.api_response(
                message='Validation failed',
                start_time=start_time,
                is_success=False,
                data={'errors': errors},
                status_code=400
            )

        username = data['username']
        logger.debug(f"Checking for existing user with username: {username}")
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            logger.warning(f"Registration failed: Username {username} already exists")
            return Helper.api_response(
                message='Username already exists',
                start_time=start_time,
                is_success=False,
                status_code=409
            )

        logger.debug(f"Creating new user with username: {username}")
        new_user = User(username=username, password=data['password'])
        db.session.add(new_user)
        db.session.commit()
        logger.info(f"User {username} registered successfully")
        return Helper.api_response(
            message='User registered successfully',
            start_time=start_time,
            is_success=True,
            status_code=201
        )

    except Exception as e:
        db.session.rollback()
        logger.error(f"Registration failed for username {data.get('username', 'unknown')}: {str(e)}", exc_info=True)
        return Helper.api_response(
            message=f'Registration failed: {str(e)}',
            start_time=start_time,
            is_success=False,
            status_code=500
        )


@auth_bp.route('/login', methods=['POST'])
def login():
    """
    Handle user login via POST request.

    This endpoint accepts a JSON payload with username and password, validates the input,
    verifies credentials, and issues access and refresh tokens upon successful authentication.

    Returns:
        JSON response containing:
        - message: A descriptive message about the request outcome
        - is_success: Boolean indicating success or failure
        - data: Access and refresh tokens with token type (if successful), or errors (if failed)
        - status_code: HTTP status code (200, 400, 401, or 500)
        - execution_time: Time taken to process the request in milliseconds

    Raises:
        Exception: If login fails (e.g., database error), rolls back the transaction
                   and returns a 500 error with the exception message.
    """
    start_time = datetime.now(timezone.utc)
    logger.info("Received POST request to /login endpoint")

    data = request.get_json()

    try:
        logger.debug("Parsing JSON payload for login")
        if not data:
            logger.warning("No JSON payload provided in login request")
            return Helper.api_response(
                message='No JSON payload provided',
                start_time=start_time,
                is_success=False,
                data={'errors': ['No JSON payload provided']},
                status_code=400
            )

        logger.debug(f"Validating login data: {data}")
        errors = validate_login(data)
        if errors:
            logger.warning(f"Login validation failed: {errors}")
            return Helper.api_response(
                message='Validation failed',
                start_time=start_time,
                is_success=False,
                data={'errors': errors},
                status_code=400
            )

        username = data['username']
        logger.debug(f"Querying user with username: {username}")
        user = User.query.filter_by(username=username).first()
        if not user or not user.check_password(data['password']):
            logger.warning(f"Login failed for username {username}: Invalid credentials")
            return Helper.api_response(
                message='Invalid credentials',
                start_time=start_time,
                is_success=False,
                status_code=401
            )

        logger.debug(f"Updating last login for user {username}")
        user.update_last_login()
        db.session.commit()

        logger.debug(f"Generating tokens for user {username}")
        access_token = TokenAuth.generate_access_token(username)
        refresh_token = TokenAuth.generate_refresh_token(username)
        logger.info(f"User {username} logged in successfully")
        return Helper.api_response(
            message='Login successful',
            start_time=start_time,
            is_success=True,
            data={
                'access_token': access_token,
                'refresh_token': refresh_token,
                'token_type': 'bearer'
            },
            status_code=200
        )

    except Exception as e:
        db.session.rollback()
        logger.error(f"Login failed for username {data.get('username', 'unknown')}: {str(e)}", exc_info=True)
        return Helper.api_response(
            message=f'Login failed: {str(e)}',
            start_time=start_time,
            is_success=False,
            status_code=500
        )


@auth_bp.route('/refresh', methods=['POST'])
def refresh_token():
    """
    Handle token refresh via POST request.

    This endpoint accepts a refresh token in the Authorization header, validates it,
    and issues a new access token if the refresh token is valid and not blacklisted.

    Returns:
        JSON response containing:
        - message: A descriptive message about the request outcome
        - is_success: Boolean indicating success or failure
        - data: New access token and token type (if successful)
        - status_code: HTTP status code (200, 401)
        - execution_time: Time taken to process the request in milliseconds
    """
    start_time = datetime.now(timezone.utc)
    logger.info("Received POST request to /refresh endpoint")

    logger.debug("Checking Authorization header")
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        logger.warning("Refresh token missing in Authorization header")
        return Helper.api_response(
            message='Refresh token is missing',
            start_time=start_time,
            is_success=False,
            status_code=401
        )

    refresh_token = auth_header.split(' ')[1]
    logger.debug(f"Checking if refresh token is blacklisted")
    if refresh_token in blacklisted_tokens:
        logger.warning("Refresh token is blacklisted")
        return Helper.api_response(
            message='Refresh token has been invalidated',
            start_time=start_time,
            is_success=False,
            status_code=401
        )

    logger.debug(f"Decoding refresh token")
    payload = TokenAuth.decode_token(refresh_token)
    if not payload or payload.get('type') != 'refresh':
        logger.warning(f"Invalid or expired refresh token")
        return Helper.api_response(
            message='Invalid or expired refresh token',
            start_time=start_time,
            is_success=False,
            status_code=401
        )

    user_id = payload['user_id']
    logger.debug(f"Generating new access token for user_id: {user_id}")
    new_access_token = TokenAuth.generate_access_token(user_id)
    logger.info(f"Token refreshed successfully for user_id: {user_id}")
    return Helper.api_response(
        message='Token refreshed successfully',
        start_time=start_time,
        is_success=True,
        data={
            'access_token': new_access_token,
            'token_type': 'bearer'
        },
        status_code=200
    )


@auth_bp.route('/logout', methods=['POST'])
@token_required
def logout():
    """
    Handle user logout via POST request.

    This endpoint requires a valid access token, blacklists it to prevent further use,
    and confirms successful logout. Requires the @token_required decorator.

    Returns:
        JSON response containing:
        - message: A descriptive message about the request outcome
        - is_success: Boolean indicating success or failure
        - status_code: HTTP status code (200)
        - execution_time: Time taken to process the request in milliseconds
    """
    start_time = datetime.now(timezone.utc)
    logger.info(f"Received POST request to /logout endpoint for user_id: {request.user_id}")

    logger.debug("Extracting token from Authorization header")
    auth_header = request.headers.get('Authorization')
    token = auth_header.split(' ')[1]
    logger.debug(f"Blacklisting token for user_id: {request.user_id}")
    blacklisted_tokens.add(token)
    logger.info(f"User {request.user_id} logged out successfully")
    return Helper.api_response(
        message='Logged out successfully',
        start_time=start_time,
        is_success=True,
        status_code=200
    )
