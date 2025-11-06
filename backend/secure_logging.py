"""
Enhanced Secure Logging System for AI Data Cleaning System
Features: Structured logging, log rotation, security filtering, audit trails
"""

import os
import json
import logging
import logging.handlers
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import hashlib
import re
from functools import wraps

# Configure log directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Security patterns to filter from logs
SENSITIVE_PATTERNS = [
    r'password["\s]*[:=]["\s]*[^"\s,}]+',
    r'token["\s]*[:=]["\s]*[^"\s,}]+',
    r'key["\s]*[:=]["\s]*[^"\s,}]+',
    r'secret["\s]*[:=]["\s]*[^"\s,}]+',
    r'api_key["\s]*[:=]["\s]*[^"\s,}]+',
    r'authorization["\s]*[:=]["\s]*[^"\s,}]+',
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
    r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card numbers
    r'\b\d{3}-\d{2}-\d{4}\b',  # SSN format
]


class SecurityFilter(logging.Filter):
    """Filter to remove sensitive information from logs"""

    def __init__(self):
        super().__init__()
        self.compiled_patterns = [re.compile(
            pattern, re.IGNORECASE) for pattern in SENSITIVE_PATTERNS]

    def filter(self, record):
        """Filter sensitive information from log records"""
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = self.sanitize_message(record.msg)

        if hasattr(record, 'args') and record.args:
            record.args = tuple(
                self.sanitize_message(
                    str(arg)) if isinstance(arg, str) else arg
                for arg in record.args
            )

        return True

    def sanitize_message(self, message: str) -> str:
        """Remove sensitive information from message"""
        for pattern in self.compiled_patterns:
            message = pattern.sub('[REDACTED]', message)
        return message


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'process_id': os.getpid(),
        }

        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id
        if hasattr(record, 'ip_address'):
            log_entry['ip_address'] = record.ip_address
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id

        return json.dumps(log_entry, ensure_ascii=False)


class SecureLogger:
    """Enhanced secure logger with audit capabilities"""

    def __init__(self, name: str = "DataCleaningSystem"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        self.logger.handlers.clear()

        self._setup_handlers()
        self._setup_database_logging()

    def _setup_handlers(self):
        """Setup logging handlers with rotation and filtering"""

        # Console handler with security filter
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(SecurityFilter())
        self.logger.addHandler(console_handler)

        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_DIR / "app.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(StructuredFormatter())
        file_handler.addFilter(SecurityFilter())
        self.logger.addHandler(file_handler)

        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            LOG_DIR / "error.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        error_handler.addFilter(SecurityFilter())
        self.logger.addHandler(error_handler)

        # Security audit handler
        security_handler = logging.handlers.RotatingFileHandler(
            LOG_DIR / "security.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=10
        )
        security_handler.setLevel(logging.WARNING)
        security_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(security_handler)

    def _setup_database_logging(self):
        """Setup database logging if available"""
        try:
            from database import DatabaseManager, get_db, SQLALCHEMY_AVAILABLE
            if SQLALCHEMY_AVAILABLE:
                self.db_logging_enabled = True
                self.db_manager = DatabaseManager()
            else:
                self.db_logging_enabled = False
        except ImportError:
            self.db_logging_enabled = False

    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message"""
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log(logging.CRITICAL, message, **kwargs)

    def security_event(self, event_type: str, message: str, **kwargs):
        """Log security event"""
        security_message = f"SECURITY [{event_type}]: {message}"
        self._log(logging.WARNING, security_message, **kwargs)

    def audit_trail(self, action: str, resource: str, user_id: str = None,
                    details: Dict[str, Any] = None, **kwargs):
        """Log audit trail event"""
        audit_data = {
            'action': action,
            'resource': resource,
            'user_id': user_id,
            'details': details or {}
        }
        audit_message = f"AUDIT: {action} on {resource}"
        self._log(logging.INFO, audit_message, audit_data=audit_data, **kwargs)

    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method"""
        # Create log record with extra data, avoiding reserved keywords
        extra = {}
        for k, v in kwargs.items():
            if k not in ['exc_info', 'module', 'function', 'line_number']:
                extra[k] = v

        # Use custom field names to avoid conflicts
        if 'module' in kwargs:
            extra['module_name'] = kwargs['module']
        if 'function' in kwargs:
            extra['function_name'] = kwargs['function']
        if 'line_number' in kwargs:
            extra['line_no'] = kwargs['line_number']

        # Log to standard logger
        self.logger.log(level, message, extra=extra,
                        exc_info=kwargs.get('exc_info'))

        # Log to database if enabled
        if self.db_logging_enabled and level >= logging.WARNING:
            self._log_to_database(level, message, **kwargs)

    def _log_to_database(self, level: int, message: str, **kwargs):
        """Log to database"""
        # Database logging disabled - would require PostgreSQL running
        return
        # try:
        #     from database import SessionLocal
        #
        #     level_name = logging.getLevelName(level)
        #
        #     with SessionLocal() as db:
        #         self.db_manager.log_system_event(
        #             db=db,
        #             level=level_name,
        #             message=message,
        #             module=kwargs.get('module'),
        #             function=kwargs.get('function'),
        #             line_number=kwargs.get('line_number'),
        #             exception_info=str(kwargs.get('exc_info')) if kwargs.get('exc_info') else None
        #         )
        # except Exception as e:
        #     # Don't let database logging errors break the application
        #     self.logger.error(f"Failed to log to database: {e}")


class RequestLogger:
    """Logger for HTTP requests with correlation IDs"""

    def __init__(self, logger: SecureLogger):
        self.logger = logger

    def log_request(self, method: str, path: str, status_code: int,
                    response_time: float, user_agent: str = None,
                    ip_address: str = None, request_id: str = None):
        """Log HTTP request"""
        message = f"{method} {path} {status_code} {response_time:.3f}s"
        self.logger.info(
            message,
            method=method,
            path=path,
            status_code=status_code,
            response_time=response_time,
            user_agent=user_agent,
            ip_address=ip_address,
            request_id=request_id
        )

    def log_file_upload(self, filename: str, file_size: int, file_hash: str,
                        ip_address: str = None, session_id: str = None):
        """Log file upload event"""
        self.logger.audit_trail(
            action="FILE_UPLOAD",
            resource=f"file:{filename}",
            details={
                'filename': filename,
                'file_size': file_size,
                'file_hash': file_hash
            },
            ip_address=ip_address,
            session_id=session_id
        )

    def log_data_processing(self, session_id: str, file_id: str,
                            processing_time: float, issues_found: int):
        """Log data processing event"""
        self.logger.audit_trail(
            action="DATA_PROCESSING",
            resource=f"session:{session_id}",
            details={
                'file_id': file_id,
                'processing_time': processing_time,
                'issues_found': issues_found
            },
            session_id=session_id
        )

    def log_file_download(self, filename: str, file_type: str,
                          ip_address: str = None, session_id: str = None):
        """Log file download event"""
        self.logger.audit_trail(
            action="FILE_DOWNLOAD",
            resource=f"file:{filename}",
            details={
                'filename': filename,
                'file_type': file_type
            },
            ip_address=ip_address,
            session_id=session_id
        )


def log_function_call(logger: SecureLogger):
    """Decorator to log function calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.utcnow()

            try:
                result = func(*args, **kwargs)
                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()

                logger.info(
                    f"Function {func.__name__} completed successfully",
                    function_name=func.__name__,
                    duration=duration,
                    module_name=func.__module__
                )

                return result
            except Exception as e:
                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()

                logger.error(
                    f"Function {func.__name__} failed: {str(e)}",
                    function_name=func.__name__,
                    duration=duration,
                    module_name=func.__module__,
                    exc_info=True
                )
                raise

        return wrapper
    return decorator


# Global logger instances
secure_logger = SecureLogger("DataCleaningSystem")
request_logger = RequestLogger(secure_logger)

# Convenience functions


def log_info(message: str, **kwargs):
    """Log info message"""
    secure_logger.info(message, **kwargs)


def log_warning(message: str, **kwargs):
    """Log warning message"""
    secure_logger.warning(message, **kwargs)


def log_error(message: str, **kwargs):
    """Log error message"""
    secure_logger.error(message, **kwargs)


def log_security_event(event_type: str, message: str, **kwargs):
    """Log security event"""
    secure_logger.security_event(event_type, message, **kwargs)


def log_audit_trail(action: str, resource: str, **kwargs):
    """Log audit trail event"""
    secure_logger.audit_trail(action, resource, **kwargs)
