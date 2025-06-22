from datetime import datetime

from flask import jsonify
from werkzeug.datastructures import FileStorage

from logger import Logger

logger = Logger.get_logger()


class Helper:

    @staticmethod
    def format_time_taken(start_time):
        """
        Formats the time taken to process a request.

        Args:
            start_time (datetime): The start time of the request processing.

        Returns:
            str: The formatted time taken.
        """

        time_taken = datetime.now() - start_time
        if time_taken.total_seconds() < 1:
            return f"{time_taken.total_seconds() * 1000:.2f} milliseconds"
        elif time_taken.total_seconds() < 60:
            return f"{time_taken.total_seconds():.2f} seconds"
        elif time_taken.total_seconds() < 3600:
            return f"{time_taken.total_seconds() / 60:.2f} minutes"
        else:
            return f"{time_taken.total_seconds() / 3600:.2f} hours"

    @staticmethod
    def validate_image_file(image_file) -> bool:
        """
        Validates the uploaded Image file.

        Args:
            image_file (FileStorage): The uploaded Image file.

        Returns:
            bool: True if the Image file is valid, False otherwise.
        """

        return image_file and image_file.filename and Helper.allowed_file(image_file.filename)

    @staticmethod
    def allowed_file(filename: str) -> bool:
        """
        Checks if the file has an allowed extension.

        Args:
            filename (str): The name of the file to check.

        Returns:
            bool: True if the file has an allowed extension, False otherwise.
        """

        ext = filename.split(".")[-1] if '.' in filename else ''
        return ext.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

    @staticmethod
    def error_response(message, start_time, status_code=400):
        return jsonify({
            "message": message,
            "is_success": False,
            "time_taken": Helper.format_time_taken(start_time),
        }), status_code
