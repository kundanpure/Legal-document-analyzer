"""
Advanced logging configuration for LegalMind AI
Enhanced with multiple handlers, structured logging, and environment-specific settings
"""

import logging
import logging.handlers
import sys
import os
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from pathlib import Path


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        # Create structured log entry
        log_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process_id': os.getpid(),
            'thread_name': record.threadName
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id
        if hasattr(record, 'document_id'):
            log_entry['document_id'] = record.document_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        
        return json.dumps(log_entry, default=str)


class ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter for better readability"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Add color to level name
        level_name = record.levelname
        if level_name in self.COLORS:
            colored_level = f"{self.COLORS[level_name]}{level_name}{self.COLORS['RESET']}"
            record.levelname = colored_level
        
        # Format the message
        formatted = super().format(record)
        
        # Reset level name for other formatters
        record.levelname = level_name
        
        return formatted


def setup_logging(
    environment: str = "development",
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True,
    enable_structured: bool = False
) -> None:
    """
    Setup comprehensive logging configuration
    
    Args:
        environment: Environment (development/staging/production)
        log_level: Logging level (DEBUG/INFO/WARNING/ERROR)
        log_dir: Directory for log files
        enable_console: Enable console logging
        enable_file: Enable file logging
        enable_structured: Use structured JSON logging
    """
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.setLevel(numeric_level)
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        
        if environment == "production" or enable_structured:
            # Use structured logging in production
            console_formatter = StructuredFormatter()
        else:
            # Use colored formatter for development
            console_formatter = ColoredConsoleFormatter(
                '%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(numeric_level)
        root_logger.addHandler(console_handler)
    
    # File handlers
    if enable_file:
        if log_dir is None:
            log_dir = os.getenv("LOG_DIR", "logs")
        
        # Create log directory if it doesn't exist
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Main application log (rotating)
        app_log_file = os.path.join(log_dir, "legalmind.log")
        file_handler = logging.handlers.RotatingFileHandler(
            app_log_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10,
            encoding='utf-8'
        )
        
        if enable_structured or environment == "production":
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(numeric_level)
        root_logger.addHandler(file_handler)
        
        # Error log (separate file for errors only)
        error_log_file = os.path.join(log_dir, "errors.log")
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)
        
        # Access log for API requests (if needed)
        if environment in ["staging", "production"]:
            access_log_file = os.path.join(log_dir, "access.log")
            access_handler = logging.handlers.TimedRotatingFileHandler(
                access_log_file,
                when='midnight',
                interval=1,
                backupCount=30,
                encoding='utf-8'
            )
            access_handler.setLevel(logging.INFO)
            access_handler.setFormatter(StructuredFormatter())
            
            # Create separate logger for access logs
            access_logger = logging.getLogger("access")
            access_logger.addHandler(access_handler)
            access_logger.setLevel(logging.INFO)
            access_logger.propagate = False  # Don't propagate to root logger
    
    # Set specific logger levels
    logger_configs = {
        "uvicorn": logging.WARNING,
        "uvicorn.access": logging.INFO,
        "fastapi": logging.INFO,
        "httpx": logging.WARNING,
        "google": logging.WARNING,
        "urllib3": logging.WARNING,
    }
    
    for logger_name, level in logger_configs.items():
        logging.getLogger(logger_name).setLevel(level)
    
    # Log the configuration
    main_logger = logging.getLogger(__name__)
    main_logger.info(f"Logging configured for {environment} environment")
    main_logger.info(f"Log level: {log_level}")
    main_logger.info(f"Console logging: {enable_console}")
    main_logger.info(f"File logging: {enable_file}")
    main_logger.info(f"Structured logging: {enable_structured}")


def get_logger(
    name: str,
    level: Optional[int] = None,
    extra_fields: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """
    Get configured logger instance with optional context
    
    Args:
        name: Logger name (usually __name__)
        level: Optional specific level for this logger
        extra_fields: Optional extra fields to include in logs
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if level is not None:
        logger.setLevel(level)
    
    # If no handlers are configured, set up basic logging
    if not logging.getLogger().handlers:
        setup_logging()
    
    # Create a logger adapter if extra fields are provided
    if extra_fields:
        logger = logging.LoggerAdapter(logger, extra_fields)
    
    return logger


def get_access_logger() -> logging.Logger:
    """Get logger specifically for API access logging"""
    return logging.getLogger("access")


def log_api_request(
    method: str,
    path: str,
    status_code: int,
    processing_time: float,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log API request with structured data
    
    Args:
        method: HTTP method
        path: Request path
        status_code: Response status code
        processing_time: Processing time in seconds
        user_id: User identifier
        session_id: Session identifier
        request_id: Request identifier
        extra: Additional fields to log
    """
    access_logger = get_access_logger()
    
    log_data = {
        'event_type': 'api_request',
        'method': method,
        'path': path,
        'status_code': status_code,
        'processing_time': processing_time,
        'user_id': user_id,
        'session_id': session_id,
        'request_id': request_id
    }
    
    if extra:
        log_data.update(extra)
    
    # Filter out None values
    log_data = {k: v for k, v in log_data.items() if v is not None}
    
    access_logger.info("API Request", extra=log_data)


def log_document_processing(
    document_id: str,
    operation: str,
    status: str,
    processing_time: float,
    file_size: Optional[int] = None,
    document_type: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    error: Optional[str] = None
) -> None:
    """
    Log document processing events
    
    Args:
        document_id: Document identifier
        operation: Operation performed (upload, analyze, etc.)
        status: Operation status (success, error, etc.)
        processing_time: Processing time in seconds
        file_size: File size in bytes
        document_type: Type of document
        user_id: User identifier
        session_id: Session identifier
        error: Error message if operation failed
    """
    logger = get_logger("document_processing")
    
    log_data = {
        'event_type': 'document_processing',
        'document_id': document_id,
        'operation': operation,
        'status': status,
        'processing_time': processing_time,
        'file_size': file_size,
        'document_type': document_type,
        'user_id': user_id,
        'session_id': session_id,
        'error': error
    }
    
    # Filter out None values
    log_data = {k: v for k, v in log_data.items() if v is not None}
    
    if status == "error":
        logger.error(f"Document processing failed: {operation}", extra=log_data)
    else:
        logger.info(f"Document processing completed: {operation}", extra=log_data)


# Initialize logging if imported directly
if __name__ != "__main__":
    # Auto-setup logging based on environment
    environment = os.getenv("ENVIRONMENT", "development")
    log_level = os.getenv("LOG_LEVEL", "INFO")
    enable_structured = os.getenv("STRUCTURED_LOGGING", "false").lower() == "true"
    
    setup_logging(
        environment=environment,
        log_level=log_level,
        enable_structured=enable_structured
    )
