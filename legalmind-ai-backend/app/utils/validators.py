"""
Validation utilities for legal document analysis
Comprehensive validation functions for files, data, and business logic
"""

import re
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path

from config.logging import get_logger

logger = get_logger(__name__)

# Try to import magic with fallback
try:
    import magic
    MAGIC_AVAILABLE = True
    logger.info("✅ python-magic available - enhanced file validation enabled")
except ImportError:
    MAGIC_AVAILABLE = False
    logger.warning("⚠️ python-magic not available - using basic file validation")

# Validation constants
SUPPORTED_FILE_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt', '.rtf', '.odt'}
SUPPORTED_MIME_TYPES = {
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/msword',
    'text/plain',
    'text/rtf',
    'application/vnd.oasis.opendocument.text'
}

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MIN_FILE_SIZE = 100  # 100 bytes

FILE_SIGNATURES = {
    b'%PDF': 'PDF',
    b'PK\x03\x04': 'ZIP/DOCX/ODT',
    b'\xd0\xcf\x11\xe0': 'DOC',
    b'{\rtf': 'RTF'
}

SUPPORTED_LANGUAGES = {
    'en', 'hi', 'ta', 'te', 'bn', 'gu', 'kn', 'ml', 'mr', 'or', 'pa', 'ur', 'as'
}

# ===============================
# FILE VALIDATION FUNCTIONS
# ===============================

async def validate_file_upload(file) -> Dict[str, Any]:
    """
    Comprehensive file upload validation with detailed feedback
    
    Args:
        file: FastAPI UploadFile object
        
    Returns:
        Detailed validation result with errors, warnings, and file info
    """
    try:
        validation_result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "file_info": {},
            "recommendations": []
        }
        
        # Basic file info
        validation_result["file_info"] = {
            "filename": file.filename if file else None,
            "content_type": getattr(file, 'content_type', None),
            "size": getattr(file, 'size', 0)
        }
        
        # Check if file exists
        if not file or not file.filename:
            validation_result["errors"].append("No file provided")
            validation_result["recommendations"].append("Please select a file to upload")
            return validation_result
        
        # Validate filename
        filename_validation = validate_filename(file.filename)
        if not filename_validation["valid"]:
            validation_result["errors"].extend(filename_validation["errors"])
        
        # Check file size
        file_size = getattr(file, 'size', 0)
        size_validation = validate_file_size_limits(file_size)
        if not size_validation["valid"]:
            validation_result["errors"].extend(size_validation["errors"])
            validation_result["recommendations"].extend(size_validation["recommendations"])
        
        # Check file extension
        extension_validation = validate_file_extension(file.filename)
        if not extension_validation["valid"]:
            validation_result["errors"].extend(extension_validation["errors"])
            validation_result["recommendations"].extend(extension_validation["recommendations"])
        
        # Check MIME type
        if hasattr(file, 'content_type') and file.content_type:
            mime_validation = validate_mime_type(file.content_type)
            if not mime_validation["valid"]:
                validation_result["warnings"].extend(mime_validation["warnings"])
        
        # Validate file content (read first few bytes)
        try:
            file_start = await file.read(16)
            await file.seek(0)  # Reset file pointer
            
            content_validation = validate_file_signature(file_start, file.filename)
            if not content_validation["valid"]:
                validation_result["warnings"].extend(content_validation["warnings"])
            
            validation_result["file_info"].update(content_validation["info"])
            
        except Exception as e:
            validation_result["warnings"].append(f"Could not read file header: {str(e)}")
        
        # Overall validation result
        validation_result["valid"] = len(validation_result["errors"]) == 0
        
        # Add summary
        if validation_result["valid"]:
            validation_result["summary"] = "File passed all validation checks"
        else:
            error_count = len(validation_result["errors"])
            warning_count = len(validation_result["warnings"])
            validation_result["summary"] = f"File validation failed: {error_count} errors, {warning_count} warnings"
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Error validating file upload: {str(e)}")
        return {
            "valid": False,
            "errors": [f"File validation failed: {str(e)}"],
            "warnings": [],
            "file_info": {},
            "recommendations": ["Please try uploading the file again"]
        }

def validate_pdf_file(file_path: str) -> Dict[str, Any]:
    """
    Enhanced PDF validation with detailed analysis
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Comprehensive validation result
    """
    try:
        validation_result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "pdf_info": {},
            "recommendations": []
        }
        
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            validation_result["errors"].append(f"File not found: {file_path}")
            validation_result["recommendations"].append("Verify the file path is correct")
            return validation_result
        
        # Check file size
        try:
            file_size = file_path.stat().st_size
            validation_result["pdf_info"]["file_size"] = file_size
            validation_result["pdf_info"]["file_size_mb"] = round(file_size / (1024 * 1024), 2)
            
            size_validation = validate_file_size_limits(file_size)
            if not size_validation["valid"]:
                validation_result["errors"].extend(size_validation["errors"])
                validation_result["recommendations"].extend(size_validation["recommendations"])
        except OSError as e:
            validation_result["errors"].append(f"Could not access file: {str(e)}")
            return validation_result
        
        # Check file signature and content
        try:
            with open(file_path, 'rb') as f:
                header = f.read(32)  # Read more bytes for better validation
                
            if not header.startswith(b'%PDF'):
                validation_result["errors"].append("File is not a valid PDF (missing PDF header)")
                validation_result["recommendations"].append("Ensure the file is a genuine PDF document")
                return validation_result
            
            # Extract PDF version
            version_match = re.search(rb'%PDF-(\d\.\d)', header)
            if version_match:
                pdf_version = version_match.group(1).decode('ascii')
                validation_result["pdf_info"]["pdf_version"] = pdf_version
                
                # Check for very old PDF versions
                if pdf_version < '1.4':
                    validation_result["warnings"].append(f"PDF version {pdf_version} is quite old")
            
            # Check for PDF structure indicators
            if b'xref' in header or b'trailer' in header:
                validation_result["pdf_info"]["has_xref_table"] = True
            
        except Exception as e:
            validation_result["errors"].append(f"Could not read PDF content: {str(e)}")
            return validation_result
        
        # Use magic library for additional validation if available
        if MAGIC_AVAILABLE:
            try:
                mime_type = magic.from_file(str(file_path), mime=True)
                validation_result["pdf_info"]["detected_mime_type"] = mime_type
                
                if mime_type != 'application/pdf':
                    validation_result["warnings"].append(
                        f"File MIME type is '{mime_type}', expected 'application/pdf'"
                    )
                    validation_result["recommendations"].append("Verify this is a genuine PDF file")
                    
            except Exception as e:
                validation_result["warnings"].append(f"MIME type detection failed: {str(e)}")
        
        # Overall validation result
        validation_result["valid"] = len(validation_result["errors"]) == 0
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Error validating PDF {file_path}: {str(e)}")
        return {
            "valid": False,
            "errors": [f"PDF validation failed: {str(e)}"],
            "warnings": [],
            "pdf_info": {},
            "recommendations": ["Please try with a different PDF file"]
        }

def validate_filename(filename: str) -> Dict[str, Any]:
    """
    Validate filename for security and compatibility
    
    Args:
        filename: Filename to validate
        
    Returns:
        Validation result
    """
    try:
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "sanitized_filename": filename
        }
        
        if not filename or not filename.strip():
            result["valid"] = False
            result["errors"].append("Filename cannot be empty")
            return result
        
        # Check for dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\x00']
        if any(char in filename for char in dangerous_chars):
            result["warnings"].append("Filename contains potentially unsafe characters")
            # Sanitize filename
            sanitized = filename
            for char in dangerous_chars:
                sanitized = sanitized.replace(char, '_')
            result["sanitized_filename"] = sanitized
        
        # Check length
        if len(filename) > 255:
            result["warnings"].append("Filename is very long and may cause issues")
        
        # Check for reserved names (Windows)
        reserved_names = {'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 'COM5',
                         'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 'LPT4',
                         'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'}
        
        base_name = Path(filename).stem.upper()
        if base_name in reserved_names:
            result["warnings"].append(f"Filename '{base_name}' is reserved on Windows systems")
        
        return result
        
    except Exception as e:
        logger.error(f"Error validating filename {filename}: {str(e)}")
        return {
            "valid": False,
            "errors": [f"Filename validation failed: {str(e)}"],
            "warnings": [],
            "sanitized_filename": filename
        }

def validate_file_extension(filename: str) -> Dict[str, Any]:
    """
    Validate file extension against supported types
    
    Args:
        filename: Filename with extension
        
    Returns:
        Validation result with recommendations
    """
    try:
        result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "detected_extension": None
        }
        
        file_extension = Path(filename).suffix.lower()
        result["detected_extension"] = file_extension
        
        if not file_extension:
            result["errors"].append("File has no extension")
            result["recommendations"].append("Add a proper file extension (e.g., .pdf, .docx)")
            return result
        
        if file_extension not in SUPPORTED_FILE_EXTENSIONS:
            result["errors"].append(f"File extension '{file_extension}' is not supported")
            result["recommendations"].extend([
                f"Supported extensions: {', '.join(sorted(SUPPORTED_FILE_EXTENSIONS))}",
                "Convert your file to a supported format"
            ])
            return result
        
        result["valid"] = True
        return result
        
    except Exception as e:
        logger.error(f"Error validating file extension for {filename}: {str(e)}")
        return {
            "valid": False,
            "errors": [f"Extension validation failed: {str(e)}"],
            "warnings": [],
            "recommendations": []
        }

def validate_file_size_limits(file_size: int) -> Dict[str, Any]:
    """
    Validate file size against limits
    
    Args:
        file_size: File size in bytes
        
    Returns:
        Validation result with size information
    """
    try:
        from .formatters import format_file_size
        
        result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "size_info": {
                "size_bytes": file_size,
                "size_formatted": format_file_size(file_size),
                "max_size_formatted": format_file_size(MAX_FILE_SIZE)
            }
        }
        
        if file_size == 0:
            result["errors"].append("File is empty (0 bytes)")
            result["recommendations"].append("Ensure the file contains content")
            return result
        
        if file_size < MIN_FILE_SIZE:
            result["warnings"].append(f"File is very small ({format_file_size(file_size)})")
            result["recommendations"].append("Verify the file contains meaningful content")
        
        if file_size > MAX_FILE_SIZE:
            result["errors"].append(
                f"File size {format_file_size(file_size)} exceeds maximum {format_file_size(MAX_FILE_SIZE)}"
            )
            result["recommendations"].extend([
                "Compress the file to reduce its size",
                "Split large documents into smaller parts",
                "Remove unnecessary images or content"
            ])
            return result
        
        # Size warnings
        if file_size > MAX_FILE_SIZE * 0.8:  # 80% of max size
            result["warnings"].append("File is quite large and may take longer to process")
        
        result["valid"] = True
        return result
        
    except Exception as e:
        logger.error(f"Error validating file size {file_size}: {str(e)}")
        return {
            "valid": False,
            "errors": [f"File size validation failed: {str(e)}"],
            "warnings": [],
            "recommendations": []
        }

def validate_mime_type(content_type: str) -> Dict[str, Any]:
    """
    Validate MIME type against supported types
    
    Args:
        content_type: MIME type to validate
        
    Returns:
        Validation result
    """
    try:
        result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "mime_info": {
                "provided_type": content_type,
                "is_supported": content_type in SUPPORTED_MIME_TYPES
            }
        }
        
        if not content_type:
            result["warnings"].append("No MIME type provided")
            return result
        
        if content_type not in SUPPORTED_MIME_TYPES:
            result["warnings"].append(f"MIME type '{content_type}' may not be fully supported")
            result["recommendations"].extend([
                f"Supported MIME types: {', '.join(sorted(SUPPORTED_MIME_TYPES))}",
                "File may still be processed if the content is valid"
            ])
        else:
            result["valid"] = True
        
        return result
        
    except Exception as e:
        logger.error(f"Error validating MIME type {content_type}: {str(e)}")
        return {
            "valid": False,
            "errors": [f"MIME type validation failed: {str(e)}"],
            "warnings": [],
            "recommendations": []
        }

def validate_file_signature(file_header: bytes, filename: str) -> Dict[str, Any]:
    """
    Validate file signature (magic bytes) against filename extension
    
    Args:
        file_header: First bytes of the file
        filename: Original filename
        
    Returns:
        Validation result with detected type information
    """
    try:
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": {
                "detected_type": None,
                "expected_type": None,
                "signature_match": False
            }
        }
        
        if not file_header:
            result["warnings"].append("Could not read file signature")
            return result
        
        # Detect file type from signature
        detected_type = None
        for signature, file_type in FILE_SIGNATURES.items():
            if file_header.startswith(signature):
                detected_type = file_type
                break
        
        result["info"]["detected_type"] = detected_type
        
        # Determine expected type from extension
        file_extension = Path(filename).suffix.lower()
        expected_type_map = {
            '.pdf': 'PDF',
            '.docx': 'ZIP/DOCX/ODT',
            '.doc': 'DOC',
            '.rtf': 'RTF',
            '.odt': 'ZIP/DOCX/ODT'
        }
        
        expected_type = expected_type_map.get(file_extension)
        result["info"]["expected_type"] = expected_type
        
        # Check if signature matches extension
        if detected_type and expected_type:
            if detected_type == expected_type or (
                detected_type == 'ZIP/DOCX/ODT' and expected_type in ['ZIP/DOCX/ODT']
            ):
                result["info"]["signature_match"] = True
            else:
                result["warnings"].append(
                    f"File signature indicates {detected_type} but extension suggests {expected_type}"
                )
        elif not detected_type and file_extension != '.txt':
            result["warnings"].append("Could not detect file type from signature")
        
        return result
        
    except Exception as e:
        logger.error(f"Error validating file signature: {str(e)}")
        return {
            "valid": True,  # Don't fail validation for signature issues
            "errors": [],
            "warnings": [f"File signature validation failed: {str(e)}"],
            "info": {}
        }

# ===============================
# DATA VALIDATION FUNCTIONS
# ===============================

def validate_language_code(lang_code: str) -> Tuple[bool, str]:
    """
    Enhanced language code validation
    
    Args:
        lang_code: Language code to validate
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        if not lang_code or not isinstance(lang_code, str):
            return False, "Language code must be a non-empty string"
        
        lang_code = lang_code.lower().strip()
        
        # Check format (2 letters)
        if not re.match(r'^[a-z]{2}$', lang_code):
            return False, "Language code must be exactly 2 lowercase letters"
        
        # Check if supported
        if lang_code not in SUPPORTED_LANGUAGES:
            supported_list = ', '.join(sorted(SUPPORTED_LANGUAGES))
            return False, f"Language '{lang_code}' not supported. Supported: {supported_list}"
        
        return True, f"Language code '{lang_code}' is valid"
        
    except Exception as e:
        logger.error(f"Error validating language code {lang_code}: {str(e)}")
        return False, f"Validation error: {str(e)}"

def validate_document_content(content: str, min_length: int = 100, 
                            max_length: int = 1000000) -> Dict[str, Any]:
    """
    Validate document content for processing
    
    Args:
        content: Document text content
        min_length: Minimum content length
        max_length: Maximum content length
        
    Returns:
        Comprehensive validation result
    """
    try:
        result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "content_info": {
                "length": len(content) if content else 0,
                "word_count": len(content.split()) if content else 0,
                "line_count": len(content.split('\n')) if content else 0
            }
        }
        
        if not content or not content.strip():
            result["errors"].append("Document content is empty")
            result["recommendations"].append("Ensure the document contains readable text")
            return result
        
        content_length = len(content)
        word_count = len(content.split())
        
        # Length validation
        if content_length < min_length:
            result["errors"].append(f"Content too short: {content_length} characters (minimum {min_length})")
            result["recommendations"].append("Document should contain more substantial content")
            return result
        
        if content_length > max_length:
            result["errors"].append(f"Content too long: {content_length} characters (maximum {max_length})")
            result["recommendations"].append("Consider splitting the document into smaller parts")
            return result
        
        # Word count validation
        if word_count < 20:
            result["warnings"].append(f"Very few words detected: {word_count}")
            result["recommendations"].append("Ensure the document contains meaningful text")
        
        # Character composition analysis
        alpha_chars = sum(1 for c in content if c.isalpha())
        alpha_ratio = alpha_chars / content_length if content_length > 0 else 0
        
        if alpha_ratio < 0.3:
            result["warnings"].append(f"Low alphabetic character ratio: {alpha_ratio:.1%}")
            result["recommendations"].append("Content may be corrupted or contain mostly non-text data")
        
        # Check for excessive repetition
        lines = content.split('\n')
        unique_lines = set(line.strip() for line in lines if line.strip())
        
        if len(lines) > 10 and len(unique_lines) / len(lines) < 0.1:
            result["warnings"].append("Excessive repetition detected in content")
            result["recommendations"].append("Check for duplicate content or extraction errors")
        
        # Sentence structure check
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) > 0:
            avg_sentence_length = word_count / len(sentences)
            result["content_info"]["sentence_count"] = len(sentences)
            result["content_info"]["avg_sentence_length"] = round(avg_sentence_length, 1)
            
            if avg_sentence_length > 50:
                result["warnings"].append("Very long average sentence length detected")
                result["recommendations"].append("Content may have formatting issues")
        
        result["valid"] = True
        return result
        
    except Exception as e:
        logger.error(f"Error validating document content: {str(e)}")
        return {
            "valid": False,
            "errors": [f"Content validation failed: {str(e)}"],
            "warnings": [],
            "recommendations": ["Please try processing the document again"],
            "content_info": {}
        }

def validate_session_request(request_data: Dict[str, Any]) -> Tuple[bool, str, List[str]]:
    """
    Enhanced session request validation with detailed feedback
    
    Args:
        request_data: Request data dictionary
        
    Returns:
        Tuple of (is_valid, error_message, recommendations)
    """
    try:
        recommendations = []
        
        # Check required fields
        required_fields = ['session_id']
        for field in required_fields:
            if field not in request_data or not request_data[field]:
                return False, f"Missing required field: {field}", ["Ensure all required fields are provided"]
        
        # Validate session ID format
        session_id = request_data['session_id']
        if not re.match(r'^session_\d+_[a-f0-9]{12,16}$', session_id):
            return False, "Invalid session ID format", [
                "Session ID should be in format: session_timestamp_randomhex",
                "Generate a new session ID using the proper format"
            ]
        
        # Optional field validation
        if 'language' in request_data:
            is_valid, message = validate_language_code(request_data['language'])
            if not is_valid:
                return False, f"Invalid language: {message}", [
                    "Use a supported language code",
                    "Check the API documentation for supported languages"
                ]
        
        # Validate document_id if present
        if 'document_id' in request_data:
            doc_id = request_data['document_id']
            if not re.match(r'^doc_\d+_[a-f0-9]{8,12}$', doc_id):
                recommendations.append("Document ID format appears non-standard but will be accepted")
        
        return True, "Validation successful", recommendations
        
    except Exception as e:
        logger.error(f"Error validating session request: {str(e)}")
        return False, f"Validation error: {str(e)}", ["Check request format and try again"]

def validate_risk_score_input(score: Any) -> Dict[str, Any]:
    """
    Validate risk score input with detailed feedback
    
    Args:
        score: Risk score to validate
        
    Returns:
        Validation result with parsed score
    """
    try:
        result = {
            "valid": False,
            "parsed_score": None,
            "original_input": score,
            "errors": [],
            "warnings": []
        }
        
        if score is None:
            result["errors"].append("Risk score cannot be null")
            return result
        
        # Try to parse score
        try:
            if isinstance(score, str):
                # Remove non-numeric characters except decimal point and minus
                clean_score = re.sub(r'[^\d.-]', '', score)
                if not clean_score:
                    result["errors"].append("Risk score contains no numeric data")
                    return result
                parsed_score = float(clean_score)
            elif isinstance(score, (int, float)):
                parsed_score = float(score)
            else:
                result["errors"].append(f"Risk score must be numeric, got {type(score).__name__}")
                return result
            
            # Validate range
            if parsed_score < 0 or parsed_score > 10:
                result["warnings"].append(f"Risk score {parsed_score} is outside normal range (0-10)")
                # Clamp to valid range
                parsed_score = max(0.0, min(10.0, parsed_score))
                result["warnings"].append(f"Score clamped to {parsed_score}")
            
            result["valid"] = True
            result["parsed_score"] = parsed_score
            
        except (ValueError, TypeError) as e:
            result["errors"].append(f"Could not parse risk score: {str(e)}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error validating risk score: {str(e)}")
        return {
            "valid": False,
            "parsed_score": None,
            "original_input": score,
            "errors": [f"Risk score validation failed: {str(e)}"],
            "warnings": []
        }

# Export validation functions
__all__ = [
    # File validation
    'validate_file_upload', 'validate_pdf_file', 'validate_filename',
    'validate_file_extension', 'validate_file_size_limits', 'validate_mime_type',
    'validate_file_signature',
    
    # Data validation
    'validate_language_code', 'validate_document_content', 'validate_session_request',
    'validate_risk_score_input',
    
    # Constants
    'SUPPORTED_FILE_EXTENSIONS', 'SUPPORTED_MIME_TYPES', 'SUPPORTED_LANGUAGES'
]
