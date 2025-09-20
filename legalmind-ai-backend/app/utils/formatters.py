"""
Data formatting utilities for legal document analysis
Specialized functions for formatting text, numbers, dates, and export data
"""

import re
import json
import csv
import io
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone, timedelta
from pathlib import Path
from decimal import Decimal, InvalidOperation

from config.logging import get_logger

logger = get_logger(__name__)

# Constants for formatting
CURRENCY_SYMBOLS = {
    'USD': '$',
    'EUR': '€',
    'GBP': '£',
    'INR': '₹',
    'JPY': '¥',
    'CNY': '¥'
}

DATE_FORMATS = {
    'US': '%m/%d/%Y',
    'EU': '%d/%m/%Y',
    'ISO': '%Y-%m-%d',
    'LONG': '%B %d, %Y',
    'SHORT': '%b %d, %Y'
}

# Legal document formatting patterns
LEGAL_SECTION_PATTERNS = [
    r'^(\d+\.?\s+[A-Z][^.\n]*)',           # Numbered sections (1. Introduction)
    r'^([A-Z][A-Z\s]{2,}):?\s*$',          # ALL CAPS headers (DEFINITIONS:)
    r'^([A-Z][a-z\s]{3,}):?\s*$',          # Title case headers (Terms and Conditions:)
    r'^\*\*([^*]+)\*\*',                   # Bold headers (**Section Title**)
    r'^([IVX]+\.?\s+[A-Z][^.\n]*)',        # Roman numeral sections
    r'^(Article\s+[IVX0-9]+[:\.\s].*)',    # Article sections
    r'^(Section\s+\d+[\.\:]?\s*.*)',       # Section headers
    r'^(Clause\s+\d+[\.\:]?\s*.*)',        # Clause headers
    r'^([A-Z][a-z]+\s+[A-Z][a-z]+):?\s*$', # Two-word titles
]

# ===============================
# TEXT FORMATTING FUNCTIONS
# ===============================

def clean_text(text: str, preserve_structure: bool = True) -> str:
    """
    Advanced text cleaning and normalization
    
    Args:
        text: Raw text to clean
        preserve_structure: Whether to preserve paragraph structure
        
    Returns:
        Cleaned and normalized text
    """
    if not text:
        return ""
    
    try:
        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        
        # Normalize Unicode characters
        import unicodedata
        text = unicodedata.normalize('NFKC', text)
        
        # Fix common OCR errors
        ocr_corrections = {
            r'\b0(?=[A-Za-z])': 'O',  # 0 to O at word start
            r'(?<=[a-z])1(?=[a-z])': 'l',  # 1 to l in middle of words
            r'\bl(?=[A-Z])': 'I',  # l to I before capitals
            r'rn(?=\s|$)': 'm',  # rn to m at word end
        }
        
        for pattern, replacement in ocr_corrections.items():
            text = re.sub(pattern, replacement, text)
        
        # Normalize whitespace
        if preserve_structure:
            # Keep paragraph breaks but normalize other whitespace
            text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
            text = re.sub(r'\n[ \t]*\n', '\n\n', text)  # Clean paragraph breaks
            text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
        else:
            # Normalize all whitespace to single spaces
            text = re.sub(r'\s+', ' ', text)
        
        # Normalize line endings
        text = re.sub(r'\r\n|\r', '\n', text)
        
        # Clean up common formatting artifacts
        text = re.sub(r'_{3,}', '___', text)  # Long underscores
        text = re.sub(r'-{3,}', '---', text)  # Long dashes
        text = re.sub(r'={3,}', '===', text)  # Long equals signs
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.;:!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)  # Ensure space after sentence endings
        
        return text.strip()
        
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}")
        return text.strip() if text else ""

def format_legal_text(text: str) -> str:
    """
    Format text for legal document display with proper styling
    
    Args:
        text: Legal document text
        
    Returns:
        Formatted legal text
    """
    try:
        if not text:
            return ""
        
        # Clean the text first
        formatted_text = clean_text(text, preserve_structure=True)
        
        # Add proper spacing around legal references
        legal_ref_patterns = [
            (r'(Section\s+\d+)', r'\n\n\1'),
            (r'(Article\s+[IVX0-9]+)', r'\n\n\1'),
            (r'(Clause\s+\d+)', r'\n\n\1'),
            (r'(Paragraph\s+\d+)', r'\n\n\1'),
        ]
        
        for pattern, replacement in legal_ref_patterns:
            formatted_text = re.sub(pattern, replacement, formatted_text, flags=re.IGNORECASE)
        
        # Format definitions (TERM means...)
        formatted_text = re.sub(
            r'([A-Z]{2,})\s+means', 
            r'\n\n**\1** means',
            formatted_text
        )
        
        # Clean up excessive newlines created by formatting
        formatted_text = re.sub(r'\n{3,}', '\n\n', formatted_text)
        
        return formatted_text.strip()
        
    except Exception as e:
        logger.error(f"Error formatting legal text: {str(e)}")
        return text

def extract_sections(text: str) -> List[Dict[str, str]]:
    """
    Enhanced section extraction with multiple pattern matching
    
    Args:
        text: Document text
        
    Returns:
        List of sections with titles, content, and metadata
    """
    try:
        sections = []
        lines = text.split('\n')
        current_section = {
            "title": "Preamble",
            "content": "",
            "section_number": 0,
            "section_type": "introduction"
        }
        section_counter = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check against all section patterns
            section_match = None
            section_type = "content"
            
            for pattern in LEGAL_SECTION_PATTERNS:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    section_match = match
                    
                    # Determine section type
                    if 'article' in line.lower():
                        section_type = "article"
                    elif 'section' in line.lower():
                        section_type = "section"
                    elif 'clause' in line.lower():
                        section_type = "clause"
                    elif re.match(r'^\d+', line):
                        section_type = "numbered"
                    elif re.match(r'^[IVX]+', line):
                        section_type = "roman"
                    else:
                        section_type = "header"
                    break
            
            if section_match:
                # Save previous section if it has content
                if current_section["content"].strip():
                    sections.append(current_section)
                
                # Start new section
                section_counter += 1
                current_section = {
                    "title": section_match.group(1).strip(),
                    "content": "",
                    "section_number": section_counter,
                    "section_type": section_type,
                    "line_number": i + 1
                }
            else:
                # Add line to current section
                current_section["content"] += line + " "
        
        # Add final section
        if current_section["content"].strip():
            sections.append(current_section)
        
        # Post-process sections
        for section in sections:
            section["content"] = section["content"].strip()
            section["word_count"] = len(section["content"].split())
            section["char_count"] = len(section["content"])
        
        return sections
        
    except Exception as e:
        logger.error(f"Error extracting sections: {str(e)}")
        return [{"title": "Full Document", "content": text, "section_number": 1, "section_type": "content"}]

def format_document_structure(sections: List[Dict[str, str]]) -> str:
    """
    Format document sections into a structured outline
    
    Args:
        sections: List of document sections
        
    Returns:
        Formatted document outline
    """
    try:
        if not sections:
            return "No document structure found"
        
        outline = []
        outline.append("# Document Structure\n")
        
        for i, section in enumerate(sections, 1):
            section_type = section.get('section_type', 'content')
            word_count = section.get('word_count', 0)
            
            # Format section entry
            if section_type == 'article':
                outline.append(f"## {section['title']}")
            elif section_type == 'section':
                outline.append(f"### {section['title']}")
            elif section_type == 'clause':
                outline.append(f"#### {section['title']}")
            else:
                outline.append(f"{i}. {section['title']}")
            
            outline.append(f"   *({word_count} words)*\n")
        
        return '\n'.join(outline)
        
    except Exception as e:
        logger.error(f"Error formatting document structure: {str(e)}")
        return "Document structure formatting failed"

def truncate_text(text: str, max_length: int, suffix: str = "...", 
                 smart_truncate: bool = True) -> str:
    """
    Enhanced text truncation with smart word boundaries
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        smart_truncate: Whether to truncate at word boundaries
        
    Returns:
        Truncated text
    """
    try:
        if not text or len(text) <= max_length:
            return text
        
        if smart_truncate:
            # Find the last space within the limit
            truncate_length = max_length - len(suffix)
            if truncate_length <= 0:
                return suffix
            
            # Look for word boundary
            last_space = text.rfind(' ', 0, truncate_length)
            if last_space > truncate_length * 0.7:  # Don't truncate too early
                return text[:last_space] + suffix
        
        # Simple truncation
        return text[:max_length - len(suffix)] + suffix
        
    except Exception as e:
        logger.error(f"Error truncating text: {str(e)}")
        return text[:max_length] if text else ""

# ===============================
# NUMERIC FORMATTING FUNCTIONS
# ===============================

def format_currency(amount: Union[float, int, str, Decimal], 
                   currency: str = 'USD', 
                   locale: str = 'US') -> str:
    """
    Format currency amounts with proper symbols and localization
    
    Args:
        amount: Amount to format
        currency: Currency code (USD, EUR, etc.)
        locale: Locale for formatting (US, EU, etc.)
        
    Returns:
        Formatted currency string
    """
    try:
        # Convert to Decimal for precision
        if isinstance(amount, str):
            # Remove any existing currency symbols and commas
            clean_amount = re.sub(r'[^\d.-]', '', amount)
            amount = Decimal(clean_amount)
        else:
            amount = Decimal(str(amount))
        
        # Get currency symbol
        symbol = CURRENCY_SYMBOLS.get(currency, currency)
        
        # Format based on locale
        if locale == 'EU':
            # European format: 1.234.567,89 €
            formatted = f"{amount:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
            return f"{formatted} {symbol}"
        else:
            # US format: $1,234,567.89
            formatted = f"{amount:,.2f}"
            return f"{symbol}{formatted}"
            
    except (InvalidOperation, ValueError) as e:
        logger.error(f"Error formatting currency {amount}: {str(e)}")
        return str(amount)

def format_percentage(value: Union[float, int, str], 
                     decimal_places: int = 1) -> str:
    """
    Format percentage values
    
    Args:
        value: Value to format as percentage
        decimal_places: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    try:
        if isinstance(value, str):
            # Remove % sign if present
            clean_value = value.replace('%', '')
            value = float(clean_value)
        else:
            value = float(value)
        
        # If value is already a percentage (>1), use as-is
        # If value is a ratio (<=1), multiply by 100
        if value <= 1:
            value *= 100
        
        return f"{value:.{decimal_places}f}%"
        
    except (ValueError, TypeError) as e:
        logger.error(f"Error formatting percentage {value}: {str(e)}")
        return str(value)

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format with enhanced precision
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    try:
        if size_bytes == 0:
            return "0 B"
        
        # Use binary (1024) units for file sizes
        size_units = ["B", "KB", "MB", "GB", "TB", "PB"]
        unit_index = 0
        size = float(size_bytes)
        
        while size >= 1024.0 and unit_index < len(size_units) - 1:
            size /= 1024.0
            unit_index += 1
        
        # Format with appropriate decimal places
        if size >= 100:
            return f"{size:.0f} {size_units[unit_index]}"
        elif size >= 10:
            return f"{size:.1f} {size_units[unit_index]}"
        else:
            return f"{size:.2f} {size_units[unit_index]}"
            
    except Exception as e:
        logger.error(f"Error formatting file size {size_bytes}: {str(e)}")
        return f"{size_bytes} B"

def format_duration(seconds: Union[float, int], 
                   style: str = 'short') -> str:
    """
    Format duration with multiple style options
    
    Args:
        seconds: Duration in seconds
        style: Format style ('short', 'long', 'precise')
        
    Returns:
        Formatted duration string
    """
    try:
        seconds = float(seconds)
        
        if seconds < 0:
            return "0s"
        
        if style == 'precise':
            # High precision for short durations
            if seconds < 1:
                return f"{seconds*1000:.0f}ms"
            elif seconds < 60:
                return f"{seconds:.1f}s"
        
        # Calculate time components
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if style == 'long':
            # Long format: "2 hours, 30 minutes, 15 seconds"
            parts = []
            if hours > 0:
                parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
            if minutes > 0:
                parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
            if secs > 0 or not parts:
                parts.append(f"{secs} second{'s' if secs != 1 else ''}")
            
            if len(parts) == 1:
                return parts[0]
            elif len(parts) == 2:
                return f"{parts[0]} and {parts[1]}"
            else:
                return f"{', '.join(parts[:-1])}, and {parts[-1]}"
        
        else:
            # Short format: "2h 30m 15s"
            if hours > 0:
                return f"{hours}h {minutes}m {secs}s"
            elif minutes > 0:
                return f"{minutes}m {secs}s"
            else:
                return f"{secs}s"
                
    except Exception as e:
        logger.error(f"Error formatting duration {seconds}: {str(e)}")
        return f"{seconds}s"

def format_number_with_ordinal(number: int) -> str:
    """
    Format number with ordinal suffix (1st, 2nd, 3rd, etc.)
    
    Args:
        number: Integer to format
        
    Returns:
        Number with ordinal suffix
    """
    try:
        if 10 <= number % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th')
        
        return f"{number}{suffix}"
        
    except Exception as e:
        logger.error(f"Error formatting ordinal {number}: {str(e)}")
        return str(number)

# ===============================
# DATE AND TIME FORMATTING
# ===============================

def format_datetime(dt: Union[datetime, str], 
                   format_style: str = 'ISO',
                   timezone_aware: bool = True) -> str:
    """
    Format datetime with various style options
    
    Args:
        dt: Datetime object or ISO string
        format_style: Style ('ISO', 'US', 'EU', 'LONG', 'SHORT', 'RELATIVE')
        timezone_aware: Whether to include timezone info
        
    Returns:
        Formatted datetime string
    """
    try:
        # Convert string to datetime if needed
        if isinstance(dt, str):
            # Handle various input formats
            for fmt in ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                try:
                    dt = datetime.strptime(dt, fmt)
                    if timezone_aware and dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    break
                except ValueError:
                    continue
            else:
                return str(dt)  # Return original if parsing fails
        
        if format_style == 'RELATIVE':
            return format_relative_time(dt)
        
        # Make timezone aware if not already
        if timezone_aware and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        # Format based on style
        if format_style in DATE_FORMATS:
            if format_style == 'ISO' and timezone_aware:
                return dt.isoformat()
            else:
                return dt.strftime(DATE_FORMATS[format_style])
        else:
            return dt.isoformat()
            
    except Exception as e:
        logger.error(f"Error formatting datetime {dt}: {str(e)}")
        return str(dt)

def format_relative_time(dt: datetime) -> str:
    """
    Format datetime as relative time (e.g., "2 hours ago", "in 5 minutes")
    
    Args:
        dt: Datetime to format
        
    Returns:
        Relative time string
    """
    try:
        now = datetime.now(timezone.utc)
        
        # Make both timezone aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        diff = now - dt
        
        # Future times
        if diff.total_seconds() < 0:
            diff = dt - now
            future = True
        else:
            future = False
        
        seconds = abs(diff.total_seconds())
        
        # Define time periods
        periods = [
            ('year', 31536000),
            ('month', 2592000),
            ('week', 604800),
            ('day', 86400),
            ('hour', 3600),
            ('minute', 60),
            ('second', 1)
        ]
        
        for period_name, period_seconds in periods:
            if seconds >= period_seconds:
                period_count = int(seconds // period_seconds)
                plural_suffix = 's' if period_count != 1 else ''
                period_text = f"{period_count} {period_name}{plural_suffix}"
                
                return f"in {period_text}" if future else f"{period_text} ago"
        
        return "just now"
        
    except Exception as e:
        logger.error(f"Error formatting relative time {dt}: {str(e)}")
        return "unknown time"

def format_business_hours(hour: int, minute: int = 0, format_24h: bool = False) -> str:
    """
    Format business hours display
    
    Args:
        hour: Hour (0-23)
        minute: Minute (0-59)
        format_24h: Use 24-hour format
        
    Returns:
        Formatted time string
    """
    try:
        if format_24h:
            return f"{hour:02d}:{minute:02d}"
        else:
            if hour == 0:
                return f"12:{minute:02d} AM"
            elif hour < 12:
                return f"{hour}:{minute:02d} AM"
            elif hour == 12:
                return f"12:{minute:02d} PM"
            else:
                return f"{hour-12}:{minute:02d} PM"
                
    except Exception as e:
        logger.error(f"Error formatting business hours {hour}:{minute}: {str(e)}")
        return f"{hour}:{minute:02d}"

# ===============================
# EXPORT FORMATTING FUNCTIONS
# ===============================

def format_data_for_export(data: Any, format_type: str = 'json') -> str:
    """
    Format data for export in various formats
    
    Args:
        data: Data to format
        format_type: Export format ('json', 'csv', 'txt')
        
    Returns:
        Formatted data string
    """
    try:
        if format_type.lower() == 'json':
            return json.dumps(data, indent=2, default=str, ensure_ascii=False)
        
        elif format_type.lower() == 'csv':
            if isinstance(data, dict):
                # Convert dict to CSV
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write headers
                writer.writerow(['Key', 'Value'])
                
                # Write data
                for key, value in data.items():
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value, default=str)
                    writer.writerow([key, value])
                
                return output.getvalue()
            
            elif isinstance(data, list) and data and isinstance(data[0], dict):
                # Convert list of dicts to CSV
                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
                return output.getvalue()
            
            else:
                # Fallback to simple string representation
                return str(data)
        
        elif format_type.lower() == 'txt':
            if isinstance(data, dict):
                lines = []
                for key, value in data.items():
                    lines.append(f"{key}: {value}")
                return '\n'.join(lines)
            elif isinstance(data, list):
                return '\n'.join(str(item) for item in data)
            else:
                return str(data)
        
        else:
            logger.warning(f"Unknown export format: {format_type}")
            return str(data)
            
    except Exception as e:
        logger.error(f"Error formatting data for export: {str(e)}")
        return str(data)

# Export formatter functions
__all__ = [
    # Text formatting
    'clean_text', 'format_legal_text', 'extract_sections', 'format_document_structure', 'truncate_text',
    
    # Numeric formatting
    'format_currency', 'format_percentage', 'format_file_size', 'format_duration', 'format_number_with_ordinal',
    
    # Date/time formatting
    'format_datetime', 'format_relative_time', 'format_business_hours',
    
    # Export formatting
    'format_data_for_export'
]
