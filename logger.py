import logging
import os
import uuid
from contextvars import ContextVar
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

# Context variables
trace_id_context = ContextVar('trace_id', default=None)

# Determine the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class CustomFormatter(logging.Formatter):
    def format(self, record):
        # Convert the pathname to a relative path
        if record.pathname:
            record.relativepath = str(Path(record.pathname).relative_to(PROJECT_ROOT))
        else:
            record.relativepath = record.pathname

        # Add context information
        record.trace_id = trace_id_context.get()

        return super().format(record)


class Logger:
    _instance = None
    _logger = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._setup_logger()
        return cls._instance

    @classmethod
    def _setup_logger(cls):
        # Create logger
        cls._logger = logging.getLogger('OllamaTestLogger')
        cls._logger.setLevel(logging.INFO)

        # Create logs directory if it doesn't exist
        logs_dir = 'logs'
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        # Clear existing handlers
        if cls._logger.hasHandlers():
            cls._logger.handlers.clear()

        # Create formatters
        log_formatter = CustomFormatter(
            '%(asctime)s %(levelname)s [%(relativepath)s:%(lineno)d] [%(trace_id)s] - %(message)s'
        )

        # Create handlers
        # 1. Time-based rotating handler (daily at midnight UTC)
        daily_handler = TimedRotatingFileHandler(
            filename='logs/record.log',
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8',
            delay=False,
            utc=True
        )
        daily_handler.setFormatter(log_formatter)

        # # 2. Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)

        # Add handlers to logger
        cls._logger.addHandler(daily_handler)
        cls._logger.addHandler(console_handler)

    @classmethod
    def get_logger(cls):
        if cls._logger is None:
            cls._setup_logger()
        return cls._logger

    @staticmethod
    def set_context(trace_id=None):
        """Set context for the current request"""
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        trace_id_context.set(trace_id)

    @staticmethod
    def clear_context():
        """Clear the context after request is processed"""
        trace_id_context.set(None)
