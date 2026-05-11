"""
Helper utility functions for legal document analysis
General purpose utilities for IDs, data processing, and business logic
"""

import re
import hashlib
import uuid
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
from pathlib import Path

from config.logging import get_logger

logger = get_logger(__name__)

# ===============================
# ID GENERATION FUNCTIONS
# ===============================

def generate_document_id(prefix: str = "doc") -> str:
    """
    Generate unique document identifier with timestamp
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        Unique document ID
    """
    timestamp = int(time.time())
    unique_part = str(uuid.uuid4()).replace('-', '')[:12]
    return f"{prefix}_{timestamp}_{unique_part}"

def generate_session_id() -> str:
    """Generate unique session identifier"""
    timestamp = int(time.time())
    unique_part = str(uuid.uuid4()).replace('-', '')[:16]
    return f"session_{timestamp}_{unique_part}"

def generate_request_id() -> str:
    """Generate unique request identifier"""
    return str(uuid.uuid4())

def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """
    Generate unique chunk ID for document sections
    
    Args:
        document_id: Parent document ID
        chunk_index: Index of the chunk
        
    Returns:
        Unique chunk ID
    """
    return f"{document_id}_chunk_{chunk_index:04d}"

# ===============================
# FILE AND DATA UTILITIES
# ===============================

def sanitize_filename(filename: str) -> str:
    """
    Enhanced filename sanitization for safe storage
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    try:
        if not filename:
            return "untitled"
        
        # Remove or replace unsafe characters
        sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
        
        # Remove control characters
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32)
        
        # Handle dots and spaces
        sanitized = sanitized.strip(' .')
        
        # Reserved names in Windows
        reserved_names = {
            'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 'COM5',
            'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 'LPT4',
            'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        }
        
        name_part = Path(sanitized).stem.upper()
        if name_part in reserved_names:
            sanitized = f"file_{sanitized}"
        
        # Limit length (most filesystems support 255 chars)
        max_length = 255
        if len(sanitized) > max_length:
            name, ext = Path(sanitized).stem, Path(sanitized).suffix
            max_name_length = max_length - len(ext)
            sanitized = name[:max_name_length] + ext
        
        return sanitized if sanitized else "untitled"
        
    except Exception as e:
        logger.error(f"Error sanitizing filename {filename}: {str(e)}")
        return f"file_{int(time.time())}"

def calculate_file_hash(file_content: bytes, algorithm: str = 'sha256') -> str:
    """
    Calculate hash of file content with selectable algorithm
    
    Args:
        file_content: File content as bytes
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256', 'sha512')
        
    Returns:
        Hexadecimal hash string
    """
    try:
        if algorithm == 'md5':
            hash_obj = hashlib.md5()
        elif algorithm == 'sha1':
            hash_obj = hashlib.sha1()
        elif algorithm == 'sha256':
            hash_obj = hashlib.sha256()
        elif algorithm == 'sha512':
            hash_obj = hashlib.sha512()
        else:
            logger.warning(f"Unknown algorithm {algorithm}, using SHA-256")
            hash_obj = hashlib.sha256()
        
        hash_obj.update(file_content)
        return hash_obj.hexdigest()
        
    except Exception as e:
        logger.error(f"Error calculating file hash: {str(e)}")
        return ""

def calculate_content_hash(content: str, algorithm: str = 'md5') -> str:
    """
    Calculate hash of text content
    
    Args:
        content: Text content to hash
        algorithm: Hash algorithm to use
        
    Returns:
        Hexadecimal hash string
    """
    return calculate_file_hash(content.encode('utf-8'), algorithm)

def mask_sensitive_data(data: Dict[str, Any], 
                       sensitive_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Enhanced sensitive data masking with customizable keys
    
    Args:
        data: Dictionary potentially containing sensitive data
        sensitive_keys: Custom list of sensitive keys
        
    Returns:
        Dictionary with masked sensitive fields
    """
    try:
        if sensitive_keys is None:
            sensitive_keys = [
                'password', 'token', 'key', 'secret', 'api_key', 'auth',
                'credentials', 'private', 'session', 'cookie', 'authorization'
            ]
        
        def mask_value(value: Any, key: str) -> Any:
            if isinstance(value, dict):
                return mask_sensitive_data(value, sensitive_keys)
            elif isinstance(value, list):
                return [mask_value(item, f"{key}[{i}]") for i, item in enumerate(value)]
            elif any(sensitive in key.lower() for sensitive in sensitive_keys):
                if isinstance(value, str) and len(value) > 4:
                    return f"{value[:2]}***{value[-2:]}"
                else:
                    return "***MASKED***"
            else:
                return value
        
        masked_data = {}
        for key, value in data.items():
            masked_data[key] = mask_value(value, key)
        
        return masked_data
        
    except Exception as e:
        logger.error(f"Error masking sensitive data: {str(e)}")
        return data

# ===============================
# TEXT PROCESSING UTILITIES
# ===============================

def extract_key_terms(text: str, max_terms: int = 20, 
                     prioritize_legal: bool = True) -> List[Dict[str, Any]]:
    """
    Enhanced key term extraction with scoring and categorization
    
    Args:
        text: Text to analyze
        max_terms: Maximum number of terms to return
        prioritize_legal: Whether to prioritize legal terms
        
    Returns:
        List of key terms with metadata
    """
    try:
        if not text:
            return []
        
        # Legal terms with categories
        legal_terms_by_category = {
            'contract': {'contract', 'agreement', 'clause', 'term', 'condition', 'obligation'},
            'liability': {'liability', 'penalty', 'breach', 'default', 'damages', 'indemnity'},
            'payment': {'payment', 'fee', 'cost', 'interest', 'rent', 'salary', 'compensation'},
            'parties': {'party', 'parties', 'landlord', 'tenant', 'employer', 'employee', 'contractor'},
            'process': {'notice', 'termination', 'renewal', 'dispute', 'arbitration', 'mediation'},
            'property': {'property', 'premises', 'title', 'ownership', 'lease', 'warranty'}
        }
        
        # Flatten legal terms with categories
        legal_terms = {}
        for category, terms in legal_terms_by_category.items():
            for term in terms:
                legal_terms[term] = category
        
        # Extract and count words (3+ characters)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_freq = {}
        
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Score and categorize terms
        scored_terms = []
        
        for word, freq in word_freq.items():
            if freq < 2:  # Skip terms that appear less than twice
                continue
            
            score = freq
            category = 'general'
            importance = 'low'
            
            # Check if it's a legal term
            if word in legal_terms:
                category = legal_terms[word]
                if prioritize_legal:
                    score *= 2  # Boost legal terms
                importance = 'high' if freq >= 5 else 'medium'
            else:
                importance = 'medium' if freq >= 5 else 'low'
            
            scored_terms.append({
                'term': word,
                'frequency': freq,
                'score': score,
                'category': category,
                'importance': importance
            })
        
        # Sort by score and return top terms
        scored_terms.sort(key=lambda x: x['score'], reverse=True)
        return scored_terms[:max_terms]
        
    except Exception as e:
        logger.error(f"Error extracting key terms: {str(e)}")
        return []

def parse_risk_score(score: Any) -> float:
    """
    Enhanced risk score parsing with validation
    
    Args:
        score: Score value to parse
        
    Returns:
        Valid risk score between 0 and 10
    """
    try:
        # Handle different input types
        if isinstance(score, str):
            # Remove any non-numeric characters except decimal point
            clean_score = re.sub(r'[^\d.-]', '', score)
            if not clean_score:
                return 5.0
            parsed_score = float(clean_score)
        elif isinstance(score, (int, float)):
            parsed_score = float(score)
        else:
            return 5.0  # Default for unknown types
        
        # Clamp to valid range
        return max(0.0, min(10.0, parsed_score))
        
    except (TypeError, ValueError) as e:
        logger.warning(f"Could not parse risk score '{score}': {str(e)}")
        return 5.0  # Default middle score

def clean_and_normalize_text(text: str) -> str:
    """
    Basic text cleaning for processing
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    cleaned = re.sub(r'\s+', ' ', text.strip())
    
    # Remove non-printable characters except newlines and tabs
    cleaned = re.sub(r'[^\x20-\x7E\n\t]', '', cleaned)
    
    # Normalize line breaks
    cleaned = re.sub(r'\r\n|\r', '\n', cleaned)
    
    return cleaned

# ===============================
# RESPONSE GENERATION UTILITIES
# ===============================

def generate_response(success: bool, data: Any = None, error: str = None, 
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Enhanced standardized API response generation
    
    Args:
        success: Success status
        data: Response data
        error: Error message
        metadata: Additional metadata
        
    Returns:
        Formatted response with comprehensive information
    """
    try:
        response = {
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": str(uuid.uuid4())
        }
        
        if success:
            if data is not None:
                response["data"] = data
            
            # Add success metadata
            response["metadata"] = {
                "processing_time": metadata.get("processing_time", 0) if metadata else 0,
                "version": "2.0",
                **(metadata or {})
            }
        else:
            response["error"] = {
                "message": error or "Unknown error occurred",
                "code": metadata.get("error_code", "GENERIC_ERROR") if metadata else "GENERIC_ERROR",
                "details": metadata.get("error_details") if metadata else None
            }
            
            # Add error context
            if metadata:
                response["error_context"] = {
                    key: value for key, value in metadata.items() 
                    if key.startswith("error_") and key != "error_code" and key != "error_details"
                }
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return {
            "success": False,
            "error": {"message": "Response generation failed", "code": "RESPONSE_ERROR"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": str(uuid.uuid4())
        }

def handle_error(error_message: str, error_code: str = "GENERIC_ERROR", 
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Enhanced error handling with structured error information
    
    Args:
        error_message: Error message
        error_code: Specific error code
        context: Additional error context
        
    Returns:
        Formatted error response
    """
    try:
        error_response = {
            "success": False,
            "error": {
                "message": error_message,
                "code": error_code,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": str(uuid.uuid4())
            }
        }
        
        if context:
            # Mask sensitive data in context
            safe_context = mask_sensitive_data(context)
            error_response["error"]["context"] = safe_context
        
        # Add suggestions based on error code
        suggestions = get_error_suggestions(error_code)
        if suggestions:
            error_response["suggestions"] = suggestions
        
        return error_response
        
    except Exception as e:
        logger.error(f"Error in error handling: {str(e)}")
        return {
            "success": False,
            "error": {
                "message": "Error handling failed",
                "code": "ERROR_HANDLER_ERROR",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

def get_error_suggestions(error_code: str) -> List[str]:
    """
    Get helpful suggestions based on error code
    
    Args:
        error_code: Error code
        
    Returns:
        List of suggestions
    """
    suggestions_map = {
        'FILE_TOO_LARGE': [
            "Try compressing the file or splitting it into smaller parts",
            "Ensure the file size is under 50MB"
        ],
        'INVALID_FILE_TYPE': [
            "Supported file types are PDF, DOCX, DOC, TXT, and RTF",
            "Check that the file extension matches the actual file type"
        ],
        'VALIDATION_ERROR': [
            "Verify all required fields are provided and properly formatted",
            "Check the API documentation for correct request format"
        ],
        'PROCESSING_TIMEOUT': [
            "Try uploading a smaller document",
            "Retry the request after a few minutes"
        ],
        'RATE_LIMIT_EXCEEDED': [
            "Wait before making additional requests",
            "Consider upgrading your plan for higher limits"
        ]
    }
    
    return suggestions_map.get(error_code, ["Contact support if the issue persists"])

def create_error_context(error: Exception, 
                        context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Enhanced error context creation for logging and debugging
    
    Args:
        error: Exception object
        context: Additional context information
        
    Returns:
        Comprehensive error context dictionary
    """
    try:
        error_context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_id": str(uuid.uuid4())
        }
        
        # Add traceback information if available
        import traceback
        error_context["traceback"] = traceback.format_exc()
        
        # Add context information
        if context:
            error_context["context"] = mask_sensitive_data(context)
        
        # Add system information
        error_context["system_info"] = {
            "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
            "platform": __import__('platform').platform()
        }
        
        return error_context
        
    except Exception as e:
        logger.error(f"Error creating error context: {str(e)}")
        return {
            "error_type": "CONTEXT_CREATION_ERROR",
            "error_message": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# ===============================
# BUSINESS LOGIC UTILITIES
# ===============================

def is_business_hours(timezone_name: str = 'UTC') -> bool:
    """
    Enhanced business hours check with timezone support
    
    Args:
        timezone_name: Timezone name (e.g., 'UTC', 'US/Eastern')
        
    Returns:
        True if within business hours
    """
    try:
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(timezone_name)
            current_time = datetime.now(tz)
        except ImportError:
            # Fallback for systems without zoneinfo
            current_time = datetime.now(timezone.utc)
    except Exception:
        current_time = datetime.now(timezone.utc)
    
    # Business hours: 8 AM to 6 PM, Monday to Friday
    if current_time.weekday() >= 5:  # Weekend
        return False
    
    return 8 <= current_time.hour < 18

def calculate_processing_priority(file_size: int, document_type: str, 
                                user_tier: str = 'standard') -> int:
    """
    Calculate processing priority based on various factors
    
    Args:
        file_size: Size of file in bytes
        document_type: Type of document
        user_tier: User subscription tier
        
    Returns:
        Priority score (1-10, higher is more urgent)
    """
    try:
        priority = 5  # Base priority
        
        # File size factor (smaller files get higher priority)
        if file_size < 1024 * 1024:  # < 1MB
            priority += 1
        elif file_size > 10 * 1024 * 1024:  # > 10MB
            priority -= 1
        
        # Document type factor
        high_priority_types = {'employment', 'lease', 'loan'}
        if document_type in high_priority_types:
            priority += 1
        
        # User tier factor
        tier_bonuses = {'premium': 3, 'pro': 2, 'standard': 0}
        priority += tier_bonuses.get(user_tier, 0)
        
        # Business hours factor
        if is_business_hours():
            priority += 1
        
        return max(1, min(10, priority))
        
    except Exception as e:
        logger.error(f"Error calculating processing priority: {str(e)}")
        return 5

def estimate_processing_time(file_size: int, document_complexity: str = 'medium') -> float:
    """
    Estimate processing time based on file characteristics
    
    Args:
        file_size: File size in bytes
        document_complexity: Complexity level ('low', 'medium', 'high')
        
    Returns:
        Estimated processing time in seconds
    """
    try:
        # Base time per MB
        base_time_per_mb = 2.0  # seconds
        
        # Complexity multipliers
        complexity_multipliers = {
            'low': 0.8,
            'medium': 1.0,
            'high': 1.5
        }
        
        file_size_mb = file_size / (1024 * 1024)
        multiplier = complexity_multipliers.get(document_complexity, 1.0)
        
        estimated_time = file_size_mb * base_time_per_mb * multiplier
        
        # Add base overhead
        estimated_time += 10  # 10 second base processing time
        
        return max(5.0, estimated_time)  # Minimum 5 seconds
        
    except Exception as e:
        logger.error(f"Error estimating processing time: {str(e)}")
        return 30.0  # Default estimate

# Export helper functions
__all__ = [
    # ID generation
    'generate_document_id', 'generate_session_id', 'generate_request_id', 'generate_chunk_id',
    
    # File and data utilities
    'sanitize_filename', 'calculate_file_hash', 'calculate_content_hash', 'mask_sensitive_data',
    
    # Text processing
    'extract_key_terms', 'parse_risk_score', 'clean_and_normalize_text',
    
    # Response utilities
    'generate_response', 'handle_error', 'create_error_context', 'get_error_suggestions',
    
    # Business logic
    'is_business_hours', 'calculate_processing_priority', 'estimate_processing_time'
]
