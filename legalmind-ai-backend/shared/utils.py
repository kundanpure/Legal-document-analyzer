"""
Utility functions for document processing
"""
import os
import uuid
import magic
from typing import Tuple, Optional
from shared.constants import SUPPORTED_FILE_TYPES, MAX_FILE_SIZE_MB


def generate_task_id() -> str:
    """Generate unique task ID"""
    return str(uuid.uuid4())


def validate_file(file_content: bytes, filename: str) -> Tuple[bool, Optional[str]]:
    """
    Validate uploaded file
    Returns: (is_valid, error_message)
    """
    # Check file size
    file_size_mb = len(file_content) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed size ({MAX_FILE_SIZE_MB}MB)"
    
    # Check file extension
    file_ext = os.path.splitext(filename.lower())[1]
    if file_ext not in SUPPORTED_FILE_TYPES:
        supported_formats = ', '.join(SUPPORTED_FILE_TYPES.keys())
        return False, f"Unsupported file format. Supported formats: {supported_formats}"
    
    # Check MIME type (if python-magic is available)
    try:
        mime_type = magic.from_buffer(file_content, mime=True)
        expected_mime = SUPPORTED_FILE_TYPES[file_ext]
        
        # Some flexibility for Office documents
        if file_ext in ['.doc', '.docx'] and 'application' in mime_type:
            return True, None
        elif mime_type != expected_mime and file_ext not in ['.doc', '.docx']:
            return False, f"File content doesn't match extension. Expected: {expected_mime}, Got: {mime_type}"
            
    except ImportError:
        # python-magic not available, skip MIME check
        pass
    except Exception as e:
        print(f"Warning: MIME type check failed: {e}")
    
    return True, None


def get_file_info(filename: str) -> dict:
    """Get file information"""
    file_ext = os.path.splitext(filename.lower())[1]
    return {
        'extension': file_ext,
        'mime_type': SUPPORTED_FILE_TYPES.get(file_ext, 'application/octet-stream'),
        'is_supported': file_ext in SUPPORTED_FILE_TYPES
    }
