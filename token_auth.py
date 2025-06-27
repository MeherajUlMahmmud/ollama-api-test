from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Optional

import jwt
from flask import Blueprint, request, jsonify

from config import Config
from logger import Logger

logger = Logger.get_logger()
auth_bp = Blueprint('auth', __name__)
blacklisted_tokens = set()


class TokenAuth:
    """
    A utility class for handling JSON Web Token (JWT) generation and decoding.

    This class provides static methods to generate access and refresh tokens, and to decode
    tokens using the configured JWT secret and algorithm. It is used to secure API endpoints
    by creating and validating tokens for user authentication.

    Attributes:
        None (all methods are static).
    """

    @staticmethod
    def generate_access_token(user_id: str) -> str:
        """
        Generate a JWT access token for a user.

        Creates a short-lived access token with a user ID, token type, expiration time,
        and issued-at time, signed with the configured JWT secret and algorithm.

        Args:
            user_id (str): The identifier of the user for whom the token is generated.

        Returns:
            str: The encoded JWT access token.
        """
        logger.debug(f"Generating access token for user_id: {user_id}")
        payload = {
            'user_id': user_id,
            'type': 'access',
            'exp': datetime.now(timezone.utc) + timedelta(hours=10),
            'iat': datetime.now(timezone.utc)
        }
        token = jwt.encode(payload, Config.JWT_SECRET, algorithm=Config.JWT_ALGORITHM)
        logger.info(f"Access token generated successfully for user_id: {user_id}")
        return token

    @staticmethod
    def generate_refresh_token(user_id: str) -> str:
        """
        Generate a JWT refresh token for a user.

        Creates a long-lived refresh token with a user ID, token type, expiration time,
        and issued-at time, signed with the configured JWT secret and algorithm.

        Args:
            user_id (str): The identifier of the user for whom the token is generated.

        Returns:
            str: The encoded JWT refresh token.
        """
        logger.debug(f"Generating refresh token for user_id: {user_id}")
        payload = {
            'user_id': user_id,
            'type': 'refresh',
            'exp': datetime.now(timezone.utc) + timedelta(days=7),
            'iat': datetime.now(timezone.utc)
        }
        token = jwt.encode(payload, Config.JWT_SECRET, algorithm=Config.JWT_ALGORITHM)
        logger.info(f"Refresh token generated successfully for user_id: {user_id}")
        return token

    @staticmethod
    def decode_token(token: str) -> tuple[Optional[dict], Optional[str]]:
        """
        Decode and validate a JWT token.

        Verifies the token's signature, expiration, and algorithm using the configured JWT
        secret. Returns the decoded payload or an error message if validation fails.

        Args:
            token (str): The JWT token to decode.

        Returns:
            tuple[dict | None, str | None]: A tuple containing:
                - The decoded payload (dict) or None if decoding fails.
                - An error message (str) or None if decoding succeeds.
        """
        logger.debug("Decoding JWT token")
        try:
            payload = jwt.decode(
                token,
                Config.JWT_SECRET,
                algorithms=[Config.JWT_ALGORITHM],
                options={"verify_exp": True},
            )
            logger.debug(f"Token decoded successfully: user_id={payload.get('user_id')}")
            return payload, None
        except jwt.ExpiredSignatureError:
            logger.warning("Token decoding failed: Token has expired")
            return None, "Token has expired"
        except jwt.InvalidTokenError as e:
            logger.warning(f"Token decoding failed: Invalid token - {str(e)}")
            return None, f"Invalid token: {str(e)}"


def token_required(f):
    """
    Decorator to require a valid JWT access token for a route.

    Checks for a valid Bearer token in the Authorization header, verifies it is not
    blacklisted, and decodes it to extract the user ID. If valid, sets `request.user_id`
    and calls the decorated function. Returns an error response for invalid or missing tokens.

    Args:
        f (callable): The function to decorate.

    Returns:
        callable: The decorated function.
    """

    @wraps(f)
    def decorated(*args, **kwargs):
        logger.info("Processing token_required decorator for route")
        token = None
        auth_header = request.headers.get('Authorization')
        logger.debug(f"Checking Authorization header: {auth_header}")

        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            logger.debug("Bearer token extracted from header")
        else:
            logger.warning("No Bearer token found in Authorization header")
            return jsonify({'message': 'Token is missing'}), 401

        logger.debug("Checking if token is blacklisted")
        if token in blacklisted_tokens:
            logger.warning("Token is blacklisted")
            return jsonify({'message': 'Token has been invalidated'}), 401

        logger.debug("Decoding token")
        payload, error = TokenAuth.decode_token(token)
        if not payload:
            logger.warning(f"Token validation failed: {error}")
            return jsonify({'message': error or 'Invalid or expired token'}), 401

        if payload.get('type') != 'access':
            logger.warning("Token is not an access token")
            return jsonify({'message': 'Invalid or expired token'}), 401

        logger.debug(f"Token validated, setting request.user_id: {payload['user_id']}")
        request.user_id = payload['user_id']
        logger.info(f"Token validation successful for user_id: {payload['user_id']}")
        return f(*args, **kwargs)

    return decorated


def refresh_token_required(f):
    """
    Decorator to require a valid JWT refresh token for a route.

    Checks for a valid Bearer refresh token in the Authorization header, verifies it is not
    blacklisted, and decodes it to extract the user ID. If valid, sets `request.user_id`
    and calls the decorated function. Returns an error response for invalid or missing tokens.

    Args:
        f (callable): The function to decorate.

    Returns:
        callable: The decorated function.
    """

    @wraps(f)
    def decorated(*args, **kwargs):
        logger.info("Processing refresh_token_required decorator for route")
        logger.debug("Checking Authorization header for refresh token")
        if request.headers.get('Authorization', '').startswith('Bearer '):
            token = request.headers.get('Authorization', '').split(' ')[1]
        else:
            token = None
        if not token:
            logger.warning("No refresh token found in Authorization header")
            return jsonify({'message': 'Refresh token is missing'}), 401

        logger.debug("Checking if refresh token is blacklisted")
        if token in blacklisted_tokens:
            logger.warning("Refresh token is blacklisted")
            return jsonify({'message': 'Refresh token has been invalidated'}), 401

        logger.debug("Decoding refresh token")
        payload, error = TokenAuth.decode_token(token)
        if not payload or payload.get('type') != 'refresh':
            logger.warning(f"Refresh token validation failed: {error or 'Not a refresh token'}")
            return jsonify({'message': 'Invalid or expired refresh token'}), 401

        logger.debug(f"Refresh token validated, setting request.user_id: {payload['user_id']}")
        request.user_id = payload['user_id']
        logger.info(f"Refresh token validation successful for user_id: {payload['user_id']}")
        return f(*args, **kwargs)

    return decorated
