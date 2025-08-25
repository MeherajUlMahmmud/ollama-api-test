import time
import traceback
from functools import wraps

import bcrypt
from flask import request, jsonify

from configs.config import ErrorMessage
from logger import Logger

logger = Logger.get_logger()


# User Management
class UserAccount:
    def __init__(self, user_id, password, allowed_ips):
        self.user_id = user_id
        self.password = password
        self.allowed_ips = allowed_ips


# Initialize user accounts with logging
USER_ACCOUNTS = [
    UserAccount(
        "RocketApp",
        bcrypt.hashpw(
            "RocketApp@Dbbl".encode('utf-8'),
            bcrypt.gensalt(),
        ).decode('utf-8'),
        ["*"],  # Allowed IPs
    ),
    UserAccount(
        "WebEkycPortal",
        bcrypt.hashpw(
            "WebEkycPortal@Dbbl".encode('utf-8'),
            bcrypt.gensalt(),
        ).decode('utf-8'),
        ["*"],  # Allowed IPs
    ),
    UserAccount(
        "konasl",
        bcrypt.hashpw(
            "konasl@Dbbl".encode('utf-8'),
            bcrypt.gensalt(),
        ).decode('utf-8'),
        ["*"],  # Allowed IPs
    ),
    UserAccount(
        "apigw",
        bcrypt.hashpw(
            "apigw@Dbbl".encode('utf-8'),
            bcrypt.gensalt(),
        ).decode('utf-8'),
        ["*"],  # Allowed IPs
    ),
]


def check_password(username: str, password: str) -> bool:
    """
    Verify user password against stored hash

    Args:
        username: The username to check
        password: The plain text password to verify

    Returns:
        bool: True if password matches, False otherwise
    """
    logger.info(f"Password check initiated for user: {username}")

    if not username or not password:
        logger.warning(
            f"Empty username or password provided. Username: '{username}', Password length: {len(password) if password else 0}")
        return False

    try:
        for user_account in USER_ACCOUNTS:
            if user_account.user_id == username:
                logger.info(f"User account found for: {username}")
                stored_password_hash = user_account.password

                # Perform password verification
                start_time = time.time()
                is_valid = bcrypt.checkpw(password.encode("utf-8"), stored_password_hash.encode("utf-8"))
                verification_time = time.time() - start_time

                if is_valid:
                    logger.info(
                        f"Password verification successful for user: {username} (took {verification_time:.3f}s)")
                    return True
                else:
                    logger.warning(f"Password verification failed for user: {username} (took {verification_time:.3f}s)")
                    return False

        logger.warning(f"User account not found: {username}")
        return False

    except Exception as e:
        logger.error(f"Error during password check for user {username}: {str(e)}")
        logger.error(f"Password check error traceback: {traceback.format_exc()}")
        return False


def check_ip(username: str, ip: str) -> bool:
    """
    Verify if the client IP is allowed for the given user

    Args:
        username: The username to check IP restrictions for
        ip: The client IP address

    Returns:
        bool: True if IP is allowed, False otherwise
    """
    logger.info(f"IP check initiated for user: {username}, IP: {ip}")

    if not username or not ip:
        logger.warning(f"Empty username or IP provided. Username: '{username}', IP: '{ip}'")
        return False

    try:
        for user_account in USER_ACCOUNTS:
            if user_account.user_id == username:
                logger.info(f"Checking IP restrictions for user: {username}")

                # Check for wildcard access
                if "*" in user_account.allowed_ips:
                    logger.info(f"Wildcard IP access granted for user: {username}, IP: {ip}")
                    return True

                # Check for specific IP match
                if ip in user_account.allowed_ips:
                    logger.info(f"IP access granted for user: {username}, IP: {ip}")
                    return True

                logger.warning(
                    f"IP access denied for user: {username}, IP: {ip}, Allowed IPs: {user_account.allowed_ips}")
                return False

        logger.warning(f"User account not found during IP check: {username}")
        return False

    except Exception as e:
        logger.error(f"Error during IP check for user {username}, IP {ip}: {str(e)}")
        logger.error(f"IP check error traceback: {traceback.format_exc()}")
        return False


def verify_user(func: callable) -> callable:
    """
    Decorator to verify user authentication and authorization

    Args:
        func: The function to be decorated

    Returns:
        callable: The decorated function with authentication checks
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        client_ip = request.remote_addr
        user_agent = request.headers.get('User-Agent', 'Unknown')
        request_path = request.path
        request_method = request.method

        logger.info(
            f"Authentication attempt started - Path: {request_method} {request_path}, IP: {client_ip}, User-Agent: {user_agent}")

        try:
            # Check for basic authentication
            basic_auth = request.authorization

            if not basic_auth:
                logger.warning(f"No basic authentication provided - IP: {client_ip}, Path: {request_path}")
                return jsonify({
                    "message": ErrorMessage.UNAUTHORIZED_ACCESS_ERROR_MESSAGE,
                    "is_success": False,
                }), 401

            username = basic_auth.username
            password = basic_auth.password

            if not username or not password:
                logger.warning(f"Empty username or password in basic auth - IP: {client_ip}, Username: '{username}'")
                return jsonify({
                    "message": ErrorMessage.UNAUTHORIZED_ACCESS_ERROR_MESSAGE,
                    "is_success": False,
                }), 401

            logger.info(f"Verifying credentials for user: {username}, IP: {client_ip}")

            # Check password
            password_valid = check_password(username, password)
            if not password_valid:
                logger.warning(f"Authentication failed - Invalid password for user: {username}, IP: {client_ip}")
                return jsonify({
                    "message": ErrorMessage.UNAUTHORIZED_ACCESS_ERROR_MESSAGE,
                    "is_success": False,
                }), 401

            # Check IP restrictions
            ip_valid = check_ip(username, client_ip)
            if not ip_valid:
                logger.warning(f"Authentication failed - IP not allowed for user: {username}, IP: {client_ip}")
                return jsonify({
                    "message": ErrorMessage.UNAUTHORIZED_ACCESS_ERROR_MESSAGE,
                    "is_success": False,
                }), 401

            # Authentication successful
            auth_time = time.time() - start_time
            logger.info(
                f"Authentication successful for user: {username}, IP: {client_ip}, Path: {request_method} {request_path} (took {auth_time:.3f}s)")

            # Call the original function
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(
                    f"Request completed successfully for user: {username}, Path: {request_method} {request_path} (total time: {execution_time:.3f}s)")
                return result

            except Exception as e:
                logger.error(f"Error executing function {func.__name__} for user {username}: {str(e)}")
                logger.error(f"Function execution error traceback: {traceback.format_exc()}")
                raise

        except Exception as e:
            auth_time = time.time() - start_time
            logger.error(
                f"Unexpected error during authentication - IP: {client_ip}, Path: {request_path}: {str(e)} (took {auth_time:.3f}s)")
            logger.error(f"Authentication error traceback: {traceback.format_exc()}")

            return jsonify({
                "message": ErrorMessage.UNAUTHORIZED_ACCESS_ERROR_MESSAGE,
                "is_success": False,
            }), 401

    return wrapper
