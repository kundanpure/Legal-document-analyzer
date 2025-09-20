"""
Enhanced custom exception classes for LegalMind AI
Production-ready with comprehensive error handling, logging, and recovery strategies
"""

import traceback
import uuid
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timezone
from enum import Enum

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from config.logging import get_logger
from config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better classification"""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    PROCESSING = "processing"
    INTEGRATION = "integration"
    SYSTEM = "system"
    BUSINESS_LOGIC = "business_logic"
    RATE_LIMITING = "rate_limiting"


class LegalMindException(Exception):
    """Enhanced base exception for all application-specific errors"""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        user_message: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        recoverable: bool = True,
        context: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code or "GENERAL_ERROR"
        self.details = details or {}
        self.severity = severity
        self.category = category
        self.user_message = user_message or message
        self.suggestions = suggestions or []
        self.recoverable = recoverable
        self.context = context or {}
        self.error_id = str(uuid.uuid4())
        self.timestamp = datetime.now(timezone.utc)
        
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses"""
        return {
            "error_id": self.error_id,
            "error_code": self.error_code,
            "message": self.message,
            "user_message": self.user_message,
            "severity": self.severity.value,
            "category": self.category.value,
            "details": self.details,
            "suggestions": self.suggestions,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context
        }
    
    def log_error(self):
        """Log the error with appropriate level based on severity"""
        log_data = {
            "error_id": self.error_id,
            "error_code": self.error_code,
            "category": self.category.value,
            "severity": self.severity.value,
            "context": self.context
        }
        
        if self.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"Critical error: {self.message}", extra=log_data)
        elif self.severity == ErrorSeverity.HIGH:
            logger.error(f"High severity error: {self.message}", extra=log_data)
        elif self.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"Medium severity error: {self.message}", extra=log_data)
        else:
            logger.info(f"Low severity error: {self.message}", extra=log_data)


class ValidationError(LegalMindException):
    """Enhanced validation error with field-specific details"""
    
    def __init__(
        self,
        message: str = "Validation failed",
        field_errors: Optional[Dict[str, List[str]]] = None,
        error_code: str = "VALIDATION_ERROR",
        **kwargs
    ):
        details = kwargs.pop('details', {})
        if field_errors:
            details['field_errors'] = field_errors
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.VALIDATION,
            user_message="Please check your input and try again",
            suggestions=["Verify all required fields are filled", "Check data formats"],
            **kwargs
        )


class DocumentProcessingError(LegalMindException):
    """Enhanced document processing error with stage tracking"""
    
    def __init__(
        self,
        message: str = "Document processing failed",
        processing_stage: Optional[str] = None,
        document_id: Optional[str] = None,
        retry_possible: bool = True,
        error_code: str = "DOCUMENT_PROCESSING_ERROR",
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if processing_stage:
            context['processing_stage'] = processing_stage
        if document_id:
            context['document_id'] = document_id
        
        suggestions = kwargs.pop('suggestions', [])
        if retry_possible:
            suggestions.append("Try uploading the document again")
        if processing_stage == "text_extraction":
            suggestions.extend([
                "Ensure the document is not password protected",
                "Check if the document contains readable text"
            ])
        elif processing_stage == "ai_analysis":
            suggestions.extend([
                "The document may be too complex for analysis",
                "Try with a simpler document first"
            ])
        
        super().__init__(
            message=message,
            error_code=error_code,
            context=context,
            suggestions=suggestions,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.PROCESSING,
            recoverable=retry_possible,
            **kwargs
        )


class GeminiAnalysisError(LegalMindException):
    """Enhanced Gemini AI analysis error with model details"""
    
    def __init__(
        self,
        message: str = "AI analysis failed",
        model_name: Optional[str] = None,
        request_id: Optional[str] = None,
        quota_exceeded: bool = False,
        error_code: str = "GEMINI_ANALYSIS_ERROR",
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if model_name:
            context['model_name'] = model_name
        if request_id:
            context['request_id'] = request_id
        
        suggestions = kwargs.pop('suggestions', [])
        if quota_exceeded:
            suggestions.extend([
                "API quota may be exceeded",
                "Try again in a few minutes",
                "Contact support if the issue persists"
            ])
            severity = ErrorSeverity.HIGH
        else:
            suggestions.extend([
                "The document may contain unsupported content",
                "Try with a different document",
                "Check your internet connection"
            ])
            severity = ErrorSeverity.MEDIUM
        
        super().__init__(
            message=message,
            error_code=error_code,
            context=context,
            suggestions=suggestions,
            severity=severity,
            category=ErrorCategory.INTEGRATION,
            **kwargs
        )


class TranslationError(LegalMindException):
    """Enhanced translation error with language details"""
    
    def __init__(
        self,
        message: str = "Translation failed",
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
        text_length: Optional[int] = None,
        error_code: str = "TRANSLATION_ERROR",
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if source_language:
            context['source_language'] = source_language
        if target_language:
            context['target_language'] = target_language
        if text_length:
            context['text_length'] = text_length
        
        suggestions = kwargs.pop('suggestions', [])
        if text_length and text_length > 5000:
            suggestions.append("Text may be too long for translation")
        suggestions.extend([
            "Check if the target language is supported",
            "Try with shorter text",
            "Verify your internet connection"
        ])
        
        super().__init__(
            message=message,
            error_code=error_code,
            context=context,
            suggestions=suggestions,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.INTEGRATION,
            **kwargs
        )


class VoiceGenerationError(LegalMindException):
    """Enhanced voice generation error with audio details"""
    
    def __init__(
        self,
        message: str = "Voice generation failed",
        language: Optional[str] = None,
        voice_type: Optional[str] = None,
        text_length: Optional[int] = None,
        error_code: str = "VOICE_GENERATION_ERROR",
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if language:
            context['language'] = language
        if voice_type:
            context['voice_type'] = voice_type
        if text_length:
            context['text_length'] = text_length
        
        suggestions = kwargs.pop('suggestions', [])
        if text_length and text_length > 5000:
            suggestions.append("Text may be too long for voice generation")
        suggestions.extend([
            "Check if the language supports voice generation",
            "Try with shorter text",
            "Verify your internet connection"
        ])
        
        super().__init__(
            message=message,
            error_code=error_code,
            context=context,
            suggestions=suggestions,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.INTEGRATION,
            **kwargs
        )


class ReportGenerationError(LegalMindException):
    """Enhanced report generation error with template details"""
    
    def __init__(
        self,
        message: str = "Report generation failed",
        template_name: Optional[str] = None,
        document_id: Optional[str] = None,
        export_format: Optional[str] = None,
        error_code: str = "REPORT_GENERATION_ERROR",
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if template_name:
            context['template_name'] = template_name
        if document_id:
            context['document_id'] = document_id
        if export_format:
            context['export_format'] = export_format
        
        suggestions = kwargs.pop('suggestions', [])
        suggestions.extend([
            "Try with a different template",
            "Check if the document analysis is complete",
            "Verify the export format is supported"
        ])
        
        super().__init__(
            message=message,
            error_code=error_code,
            context=context,
            suggestions=suggestions,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.PROCESSING,
            **kwargs
        )


class StorageError(LegalMindException):
    """Enhanced storage error with operation details"""
    
    def __init__(
        self,
        message: str = "Storage operation failed",
        operation: Optional[str] = None,
        file_path: Optional[str] = None,
        storage_type: Optional[str] = None,
        error_code: str = "STORAGE_ERROR",
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if operation:
            context['operation'] = operation
        if file_path:
            context['file_path'] = file_path
        if storage_type:
            context['storage_type'] = storage_type
        
        suggestions = kwargs.pop('suggestions', [])
        if operation == "upload":
            suggestions.extend([
                "Check your internet connection",
                "Verify file size is within limits",
                "Try uploading again"
            ])
        elif operation == "download":
            suggestions.extend([
                "The file may have been moved or deleted",
                "Try refreshing and downloading again"
            ])
        else:
            suggestions.append("Contact support if the issue persists")
        
        super().__init__(
            message=message,
            error_code=error_code,
            context=context,
            suggestions=suggestions,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.SYSTEM,
            **kwargs
        )


class ChatError(LegalMindException):
    """Enhanced chat error with session details"""
    
    def __init__(
        self,
        message: str = "Chat operation failed",
        session_id: Optional[str] = None,
        message_type: Optional[str] = None,
        error_code: str = "CHAT_ERROR",
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if session_id:
            context['session_id'] = session_id
        if message_type:
            context['message_type'] = message_type
        
        suggestions = kwargs.pop('suggestions', [])
        suggestions.extend([
            "Try refreshing the chat session",
            "Check your internet connection",
            "Start a new chat session if the problem persists"
        ])
        
        super().__init__(
            message=message,
            error_code=error_code,
            context=context,
            suggestions=suggestions,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.PROCESSING,
            **kwargs
        )


class AuthenticationError(LegalMindException):
    """Enhanced authentication error"""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        auth_method: Optional[str] = None,
        user_identifier: Optional[str] = None,
        error_code: str = "AUTHENTICATION_ERROR",
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if auth_method:
            context['auth_method'] = auth_method
        if user_identifier:
            context['user_identifier'] = user_identifier
        
        super().__init__(
            message=message,
            error_code=error_code,
            context=context,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AUTHENTICATION,
            user_message="Please check your credentials and try again",
            suggestions=[
                "Verify your email and password",
                "Check if your account is active",
                "Try resetting your password"
            ],
            recoverable=True,
            **kwargs
        )


class AuthorizationError(LegalMindException):
    """Enhanced authorization error"""
    
    def __init__(
        self,
        message: str = "Access denied",
        required_permission: Optional[str] = None,
        user_role: Optional[str] = None,
        error_code: str = "AUTHORIZATION_ERROR",
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if required_permission:
            context['required_permission'] = required_permission
        if user_role:
            context['user_role'] = user_role
        
        super().__init__(
            message=message,
            error_code=error_code,
            context=context,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.AUTHORIZATION,
            user_message="You don't have permission to perform this action",
            suggestions=[
                "Contact your administrator for access",
                "Check if you're logged in with the correct account"
            ],
            recoverable=False,
            **kwargs
        )


class RateLimitError(LegalMindException):
    """Enhanced rate limiting error with timing details"""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        limit: Optional[int] = None,
        window_seconds: Optional[int] = None,
        retry_after: Optional[int] = None,
        endpoint: Optional[str] = None,
        error_code: str = "RATE_LIMIT_ERROR",
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if limit:
            context['limit'] = limit
        if window_seconds:
            context['window_seconds'] = window_seconds
        if retry_after:
            context['retry_after'] = retry_after
        if endpoint:
            context['endpoint'] = endpoint
        
        suggestions = kwargs.pop('suggestions', [])
        if retry_after:
            suggestions.append(f"Wait {retry_after} seconds before trying again")
        suggestions.extend([
            "Reduce the frequency of your requests",
            "Consider upgrading your plan for higher limits"
        ])
        
        super().__init__(
            message=message,
            error_code=error_code,
            context=context,
            suggestions=suggestions,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.RATE_LIMITING,
            user_message="You're sending requests too quickly. Please slow down.",
            **kwargs
        )


class ConfigurationError(LegalMindException):
    """Enhanced configuration error"""
    
    def __init__(
        self,
        message: str = "Configuration error",
        config_key: Optional[str] = None,
        config_value: Optional[str] = None,
        error_code: str = "CONFIGURATION_ERROR",
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if config_key:
            context['config_key'] = config_key
        if config_value:
            context['config_value'] = config_value
        
        super().__init__(
            message=message,
            error_code=error_code,
            context=context,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.SYSTEM,
            user_message="System configuration error. Please contact support.",
            suggestions=["Contact system administrator"],
            recoverable=False,
            **kwargs
        )


class SystemError(LegalMindException):
    """Enhanced system error for infrastructure issues"""
    
    def __init__(
        self,
        message: str = "System error occurred",
        service_name: Optional[str] = None,
        error_code: str = "SYSTEM_ERROR",
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if service_name:
            context['service_name'] = service_name
        
        super().__init__(
            message=message,
            error_code=error_code,
            context=context,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.SYSTEM,
            user_message="A system error occurred. Please try again later.",
            suggestions=[
                "Try again in a few minutes",
                "Contact support if the issue persists"
            ],
            **kwargs
        )


class TemplateError(LegalMindException):
    """Template processing error"""
    
    def __init__(
        self,
        message: str = "Template processing failed",
        template_name: Optional[str] = None,
        error_code: str = "TEMPLATE_ERROR",
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if template_name:
            context['template_name'] = template_name
        
        super().__init__(
            message=message,
            error_code=error_code,
            context=context,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.PROCESSING,
            **kwargs
        )


# Enhanced Exception Handlers

async def legalmind_exception_handler(request: Request, exc: LegalMindException) -> JSONResponse:
    """Handle custom LegalMind exceptions with enhanced logging and response"""
    
    # Log the error
    exc.log_error()
    
    # Determine HTTP status code
    status_code = _get_status_code_for_error(exc.error_code, exc.category, exc.severity)
    
    # Create response data
    response_data = {
        "success": False,
        "error": exc.to_dict(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    # Add debug information in development
    if settings.DEBUG and settings.ENVIRONMENT == "development":
        response_data["debug"] = {
            "traceback": traceback.format_exc(),
            "request_url": str(request.url),
            "request_method": request.method
        }
    
    return JSONResponse(
        status_code=status_code,
        content=response_data,
        headers={"X-Error-ID": exc.error_id}
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle FastAPI validation errors"""
    
    error_id = str(uuid.uuid4())
    
    # Extract field errors
    field_errors = {}
    for error in exc.errors():
        field_name = ".".join(str(loc) for loc in error["loc"][1:])  # Skip 'body' prefix
        if field_name not in field_errors:
            field_errors[field_name] = []
        field_errors[field_name].append(error["msg"])
    
    # Create ValidationError
    validation_error = ValidationError(
        message="Request validation failed",
        field_errors=field_errors,
        context={"request_method": request.method, "request_url": str(request.url)}
    )
    validation_error.error_id = error_id
    
    validation_error.log_error()
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": validation_error.to_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        headers={"X-Error-ID": error_id}
    )


async def http_exception_handler(request: Request, exc: Union[HTTPException, StarletteHTTPException]) -> JSONResponse:
    """Handle HTTP exceptions with consistent format"""
    
    error_id = str(uuid.uuid4())
    
    # Log non-client errors
    if exc.status_code >= 500:
        logger.error(f"HTTP {exc.status_code} error: {exc.detail}", extra={
            "error_id": error_id,
            "request_url": str(request.url),
            "request_method": request.method
        })
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "error_id": error_id,
                "error_code": f"HTTP_{exc.status_code}",
                "message": str(exc.detail),
                "user_message": _get_user_friendly_message(exc.status_code),
                "severity": "high" if exc.status_code >= 500 else "medium",
                "category": "system" if exc.status_code >= 500 else "client",
                "recoverable": exc.status_code < 500,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        },
        headers={"X-Error-ID": error_id}
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions with full error tracking"""
    
    error_id = str(uuid.uuid4())
    
    # Create system error
    system_error = SystemError(
        message=f"Unexpected error: {type(exc).__name__}",
        context={
            "exception_type": type(exc).__name__,
            "request_method": request.method,
            "request_url": str(request.url),
            "traceback": traceback.format_exc()
        }
    )
    system_error.error_id = error_id
    
    system_error.log_error()
    
    # Additional critical logging
    logger.critical(f"Unhandled exception {error_id}: {str(exc)}", extra={
        "error_id": error_id,
        "exception_type": type(exc).__name__,
        "traceback": traceback.format_exc(),
        "request_method": request.method,
        "request_url": str(request.url)
    })
    
    response_data = {
        "success": False,
        "error": system_error.to_dict(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    # Add debug information in development
    if settings.DEBUG and settings.ENVIRONMENT == "development":
        response_data["debug"] = {
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc()
        }
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response_data,
        headers={"X-Error-ID": error_id}
    )


# Helper Functions

def _get_status_code_for_error(error_code: str, category: ErrorCategory, severity: ErrorSeverity) -> int:
    """Map error codes to HTTP status codes with category and severity consideration"""
    
    # Specific error code mappings
    error_code_mapping = {
        "VALIDATION_ERROR": status.HTTP_400_BAD_REQUEST,
        "AUTHENTICATION_ERROR": status.HTTP_401_UNAUTHORIZED,
        "AUTHORIZATION_ERROR": status.HTTP_403_FORBIDDEN,
        "RATE_LIMIT_ERROR": status.HTTP_429_TOO_MANY_REQUESTS,
        "DOCUMENT_PROCESSING_ERROR": status.HTTP_422_UNPROCESSABLE_ENTITY,
        "GEMINI_ANALYSIS_ERROR": status.HTTP_503_SERVICE_UNAVAILABLE,
        "TRANSLATION_ERROR": status.HTTP_503_SERVICE_UNAVAILABLE,
        "VOICE_GENERATION_ERROR": status.HTTP_503_SERVICE_UNAVAILABLE,
        "REPORT_GENERATION_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "STORAGE_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "CHAT_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "CONFIGURATION_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "SYSTEM_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "TEMPLATE_ERROR": status.HTTP_422_UNPROCESSABLE_ENTITY
    }
    
    # Try specific error code first
    if error_code in error_code_mapping:
        return error_code_mapping[error_code]
    
    # Fall back to category-based mapping
    category_mapping = {
        ErrorCategory.VALIDATION: status.HTTP_400_BAD_REQUEST,
        ErrorCategory.AUTHENTICATION: status.HTTP_401_UNAUTHORIZED,
        ErrorCategory.AUTHORIZATION: status.HTTP_403_FORBIDDEN,
        ErrorCategory.PROCESSING: status.HTTP_422_UNPROCESSABLE_ENTITY,
        ErrorCategory.INTEGRATION: status.HTTP_503_SERVICE_UNAVAILABLE,
        ErrorCategory.SYSTEM: status.HTTP_500_INTERNAL_SERVER_ERROR,
        ErrorCategory.BUSINESS_LOGIC: status.HTTP_422_UNPROCESSABLE_ENTITY,
        ErrorCategory.RATE_LIMITING: status.HTTP_429_TOO_MANY_REQUESTS
    }
    
    return category_mapping.get(category, status.HTTP_500_INTERNAL_SERVER_ERROR)


def _get_user_friendly_message(status_code: int) -> str:
    """Get user-friendly message for HTTP status codes"""
    
    messages = {
        400: "The request was invalid. Please check your input.",
        401: "Authentication required. Please log in.",
        403: "You don't have permission to access this resource.",
        404: "The requested resource was not found.",
        422: "The request data could not be processed.",
        429: "Too many requests. Please slow down.",
        500: "An internal server error occurred. Please try again later.",
        502: "Service temporarily unavailable. Please try again later.",
        503: "Service temporarily unavailable. Please try again later.",
        504: "Request timed out. Please try again."
    }
    
    return messages.get(status_code, "An error occurred. Please try again.")


def setup_exception_handlers(app):
    """Setup comprehensive exception handlers for FastAPI app"""
    
    # Custom exception handlers
    app.add_exception_handler(LegalMindException, legalmind_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
    
    logger.info("Exception handlers configured successfully")


# Error Reporting and Monitoring

class ErrorReporter:
    """Error reporting for external monitoring systems"""
    
    def __init__(self):
        self.logger = logger
    
    async def report_error(self, error: LegalMindException, request: Optional[Request] = None):
        """Report error to external monitoring systems"""
        
        error_data = error.to_dict()
        
        if request:
            error_data["request_info"] = {
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "client": request.client.host if request.client else None
            }
        
        # In production, you would send this to error monitoring services
        # like Sentry, Rollbar, or custom monitoring systems
        
        if error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.error(f"Error reported to monitoring: {error.error_id}", extra=error_data)


# Global error reporter
error_reporter = ErrorReporter()
