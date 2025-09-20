"""
Document utilities and helper functions for multi-document processing
Enhanced with fallbacks and comprehensive error handling
"""

import hashlib
import uuid
import mimetypes
import re
import os
import tempfile
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from datetime import datetime, timezone, timedelta
from pathlib import Path

from config.logging import get_logger

logger = get_logger(__name__)

# Try to import optional dependencies with fallbacks
try:
    import magic
    MAGIC_AVAILABLE = True
    logger.info("✅ python-magic available - advanced file type detection enabled")
except ImportError:
    MAGIC_AVAILABLE = False
    logger.warning("⚠️ python-magic not available - using basic file type detection")

try:
    from PIL import Image
    PIL_AVAILABLE = True
    logger.info("✅ PIL available - image processing enabled")
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("⚠️ PIL not available - image processing disabled")

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
    logger.info("✅ PyPDF2 available - PDF processing enabled")
except ImportError:
    PYPDF2_AVAILABLE = False
    logger.warning("⚠️ PyPDF2 not available - using basic PDF processing")

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
    logger.info("✅ python-docx available - DOCX processing enabled")
except ImportError:
    DOCX_AVAILABLE = False
    logger.warning("⚠️ python-docx not available - DOCX processing disabled")

# Try to import exceptions with fallbacks
try:
    from app.core.exceptions import DocumentValidationError, DocumentProcessingError
    CUSTOM_EXCEPTIONS_AVAILABLE = True
except ImportError:
    CUSTOM_EXCEPTIONS_AVAILABLE = False
    logger.warning("⚠️ Custom exceptions not available - using standard exceptions")
    
    # Define fallback exceptions
    class DocumentValidationError(ValueError):
        """Document validation error"""
        pass
    
    class DocumentProcessingError(RuntimeError):
        """Document processing error"""
        pass

# Enhanced constants with more file types and better organization
SUPPORTED_MIME_TYPES = {
    'application/pdf': {
        'ext': ['.pdf'], 
        'max_size': 50 * 1024 * 1024,  # 50MB
        'description': 'Adobe PDF Document',
        'processing_difficulty': 'medium'
    },
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': {
        'ext': ['.docx'], 
        'max_size': 25 * 1024 * 1024,  # 25MB
        'description': 'Microsoft Word Document (2007+)',
        'processing_difficulty': 'low'
    },
    'application/msword': {
        'ext': ['.doc'], 
        'max_size': 25 * 1024 * 1024,
        'description': 'Microsoft Word Document (97-2003)',
        'processing_difficulty': 'medium'
    },
    'text/plain': {
        'ext': ['.txt'], 
        'max_size': 10 * 1024 * 1024,  # 10MB
        'description': 'Plain Text Document',
        'processing_difficulty': 'low'
    },
    'text/rtf': {
        'ext': ['.rtf'], 
        'max_size': 10 * 1024 * 1024,
        'description': 'Rich Text Format Document',
        'processing_difficulty': 'medium'
    },
    'application/vnd.oasis.opendocument.text': {
        'ext': ['.odt'],
        'max_size': 25 * 1024 * 1024,
        'description': 'OpenDocument Text Document',
        'processing_difficulty': 'medium'
    },
    'text/csv': {
        'ext': ['.csv'],
        'max_size': 5 * 1024 * 1024,  # 5MB
        'description': 'Comma Separated Values',
        'processing_difficulty': 'low'
    }
}

# Enhanced legal document keyword patterns
LEGAL_DOCUMENT_KEYWORDS = {
    'contract_indicators': [
        'agreement', 'contract', 'covenant', 'undertaking', 'obligation',
        'consideration', 'terms and conditions', 'party', 'parties',
        'whereas', 'therefore', 'hereby', 'witnesseth', 'in consideration of'
    ],
    'lease_indicators': [
        'lease', 'rent', 'tenant', 'landlord', 'premises', 'rental',
        'security deposit', 'lease term', 'rental agreement', 'lessor',
        'lessee', 'demised premises', 'rental period', 'occupancy'
    ],
    'employment_indicators': [
        'employment', 'employee', 'employer', 'salary', 'compensation',
        'benefits', 'termination', 'job description', 'work schedule',
        'position', 'duties', 'responsibilities', 'at-will employment'
    ],
    'nda_indicators': [
        'confidential', 'non-disclosure', 'proprietary', 'trade secret',
        'confidentiality agreement', 'non-compete', 'non-solicitation',
        'proprietary information', 'confidential information', 'trade secrets'
    ],
    'loan_indicators': [
        'loan', 'borrower', 'lender', 'principal', 'interest', 'repayment',
        'promissory note', 'credit', 'debt', 'mortgage', 'collateral',
        'default', 'acceleration', 'maturity date'
    ],
    'service_indicators': [
        'services', 'consultant', 'provider', 'client', 'scope of work',
        'deliverables', 'milestones', 'professional services', 'independent contractor'
    ],
    'purchase_indicators': [
        'purchase', 'sale', 'buyer', 'seller', 'goods', 'merchandise',
        'delivery', 'title', 'warranty', 'purchase price', 'closing'
    ],
    'legal_general': [
        'whereas', 'therefore', 'hereby', 'party', 'shall', 'liability',
        'indemnity', 'breach', 'jurisdiction', 'governing law', 'arbitration',
        'force majeure', 'entire agreement', 'severability', 'amendment'
    ]
}

# Enhanced risk indicators with severity levels
RISK_INDICATORS = {
    'critical_risk': [
        'unlimited personal liability', 'criminal penalties', 'forfeiture of all assets',
        'immediate termination without cause', 'waiver of all legal rights',
        'personal guarantee of corporate obligations', 'joint and several liability'
    ],
    'high_risk': [
        'penalty', 'fine', 'liquidated damages', 'breach', 'default',
        'termination', 'forfeiture', 'liability', 'indemnification',
        'personal guarantee', 'unlimited liability', 'automatic renewal',
        'cross-default', 'acceleration clause', 'broad indemnification'
    ],
    'medium_risk': [
        'obligation', 'duty', 'responsibility', 'compliance', 'notice',
        'approval required', 'consent', 'condition precedent',
        'material adverse change', 'cure period', 'specific performance'
    ],
    'financial_risk': [
        'payment', 'fee', 'cost', 'expense', 'interest', 'late fee',
        'collection costs', 'attorney fees', 'court costs', 'prepayment penalty',
        'variable interest rate', 'compound interest', 'balloon payment'
    ],
    'legal_compliance_risk': [
        'regulatory compliance', 'license requirements', 'permit obligations',
        'environmental compliance', 'safety regulations', 'data protection',
        'privacy requirements', 'anti-corruption', 'export controls'
    ],
    'operational_risk': [
        'performance standards', 'service level agreements', 'uptime requirements',
        'key personnel', 'business continuity', 'disaster recovery',
        'technology requirements', 'integration obligations'
    ]
}

# Document validation functions
def validate_file_type(filename: str, content: bytes = None) -> bool:
    """
    Validate if file type is supported with enhanced detection
    
    Args:
        filename: Name of the file
        content: File content bytes (optional, for magic number detection)
        
    Returns:
        True if supported, False otherwise
    """
    try:
        file_ext = Path(filename).suffix.lower()
        
        # Check against supported extensions
        for mime_type, config in SUPPORTED_MIME_TYPES.items():
            if file_ext in config['ext']:
                # If magic is available and content is provided, double-check
                if MAGIC_AVAILABLE and content:
                    detected_mime = magic.from_buffer(content, mime=True)
                    return detected_mime == mime_type
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error validating file type: {str(e)}")
        return False

def validate_file_size(file_content: bytes, filename: str) -> Tuple[bool, str]:
    """
    Validate file size against limits with detailed feedback
    
    Args:
        file_content: File content as bytes
        filename: Name of the file
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        file_size = len(file_content)
        file_ext = Path(filename).suffix.lower()
        
        for mime_type, config in SUPPORTED_MIME_TYPES.items():
            if file_ext in config['ext']:
                if file_size <= config['max_size']:
                    return True, f"File size {format_file_size(file_size)} is within limits"
                else:
                    max_size_formatted = format_file_size(config['max_size'])
                    current_size_formatted = format_file_size(file_size)
                    return False, f"File size {current_size_formatted} exceeds maximum {max_size_formatted} for {config['description']}"
        
        return False, "File type not supported"
        
    except Exception as e:
        logger.error(f"Error validating file size: {str(e)}")
        return False, f"Validation error: {str(e)}"

def validate_document_content(content: str, min_words: int = 50, max_words: int = 100000) -> Tuple[bool, str]:
    """
    Validate document content quality with detailed feedback
    
    Args:
        content: Document text content
        min_words: Minimum word count
        max_words: Maximum word count
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        if not content or not content.strip():
            return False, "Document content is empty"
        
        word_count = len(content.split())
        
        # Check minimum word count
        if word_count < min_words:
            return False, f"Document too short: {word_count} words (minimum {min_words})"
        
        # Check maximum word count
        if word_count > max_words:
            return False, f"Document too long: {word_count} words (maximum {max_words})"
        
        # Check for reasonable text content (not just garbage)
        alpha_ratio = sum(1 for c in content if c.isalpha()) / len(content)
        if alpha_ratio < 0.3:  # At least 30% alphabetic characters
            return False, f"Content appears to be corrupted or not text-based (only {alpha_ratio:.1%} alphabetic characters)"
        
        # Check for excessive repetition
        lines = content.split('\n')
        unique_lines = set(line.strip() for line in lines if line.strip())
        if len(lines) > 10 and len(unique_lines) / len(lines) < 0.1:
            return False, "Content appears to have excessive repetition"
        
        # Check for reasonable sentence structure
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) > 0:
            avg_sentence_length = word_count / len(sentences)
            if avg_sentence_length > 100:  # Very long sentences might indicate formatting issues
                return False, "Content may have formatting issues (very long sentences detected)"
        
        return True, f"Content validation passed: {word_count} words, {len(sentences)} sentences"
        
    except Exception as e:
        logger.error(f"Error validating document content: {str(e)}")
        return False, f"Content validation error: {str(e)}"

# Enhanced document analysis functions
def detect_document_type(content: str, filename: str = "") -> Tuple[str, float]:
    """
    Detect document type with confidence score
    
    Args:
        content: Document text content
        filename: Original filename (optional)
        
    Returns:
        Tuple of (document_type, confidence_score)
    """
    try:
        content_lower = content.lower()
        
        # Score each document type
        scores = {}
        
        for doc_type, keywords in LEGAL_DOCUMENT_KEYWORDS.items():
            doc_type_clean = doc_type.replace('_indicators', '')
            
            # Count keyword matches
            keyword_matches = sum(1 for keyword in keywords if keyword in content_lower)
            
            # Weight by keyword frequency and uniqueness
            total_words = len(content_lower.split())
            if total_words > 0:
                # Base score from keyword frequency
                frequency_score = keyword_matches / len(keywords)
                
                # Boost score for multiple matches of the same keyword
                keyword_density = sum(content_lower.count(keyword) for keyword in keywords) / total_words
                
                scores[doc_type_clean] = frequency_score + (keyword_density * 100)
        
        # Filename hints
        if filename:
            filename_lower = filename.lower()
            for doc_type in scores:
                doc_type_variations = [
                    doc_type,
                    doc_type.replace('_', ''),
                    doc_type.replace('_', '-'),
                    doc_type.replace('_', ' ')
                ]
                
                for variation in doc_type_variations:
                    if variation in filename_lower:
                        scores[doc_type] = scores.get(doc_type, 0) * 1.5  # Boost score
        
        # Special pattern matching
        special_patterns = {
            'lease': [r'lease\s+agreement', r'rental\s+agreement', r'tenant.*landlord'],
            'employment': [r'employment\s+agreement', r'job\s+offer', r'employee.*employer'],
            'nda': [r'non.?disclosure', r'confidentiality\s+agreement', r'proprietary\s+information'],
            'loan': [r'promissory\s+note', r'loan\s+agreement', r'borrower.*lender']
        }
        
        for doc_type, patterns in special_patterns.items():
            pattern_matches = sum(1 for pattern in patterns if re.search(pattern, content_lower))
            if pattern_matches > 0:
                scores[doc_type] = scores.get(doc_type, 0) + pattern_matches * 0.5
        
        # Return highest scoring type with confidence
        if scores:
            best_type, best_score = max(scores.items(), key=lambda x: x[1])
            confidence = min(1.0, best_score)
            
            if confidence > 0.3:  # Minimum confidence threshold
                return best_type, confidence
        
        return 'legal_general', 0.5  # Default fallback with medium confidence
        
    except Exception as e:
        logger.error(f"Error detecting document type: {str(e)}")
        return 'unknown', 0.0

def assess_document_risk_level(content: str, document_type: str = "unknown") -> Tuple[str, float, Dict[str, int]]:
    """
    Assess risk level with detailed breakdown
    
    Args:
        content: Document text content
        document_type: Type of document
        
    Returns:
        Tuple of (risk_level, risk_score, risk_breakdown)
    """
    try:
        content_lower = content.lower()
        
        # Count risk indicators by category
        risk_counts = {}
        total_risk_score = 0.0
        
        for risk_category, indicators in RISK_INDICATORS.items():
            count = sum(1 for indicator in indicators if indicator in content_lower)
            risk_counts[risk_category] = count
            
            # Weight different risk categories
            if risk_category == 'critical_risk':
                total_risk_score += count * 3.0
            elif risk_category == 'high_risk':
                total_risk_score += count * 2.0
            elif risk_category == 'medium_risk':
                total_risk_score += count * 1.0
            elif risk_category == 'financial_risk':
                total_risk_score += count * 1.5
            else:
                total_risk_score += count * 1.0
        
        # Normalize by document length
        total_words = len(content_lower.split())
        if total_words > 0:
            risk_density = total_risk_score / (total_words / 1000)  # Per 1000 words
        else:
            risk_density = 0
        
        # Document type specific adjustments
        type_multipliers = {
            'loan': 1.3,
            'employment': 0.8,
            'lease': 1.1,
            'nda': 0.7,
            'service': 1.2
        }
        
        if document_type in type_multipliers:
            risk_density *= type_multipliers[document_type]
        
        # Determine risk level
        if risk_density >= 8.0 or risk_counts.get('critical_risk', 0) > 0:
            risk_level = 'critical'
        elif risk_density >= 5.0 or risk_counts.get('high_risk', 0) >= 3:
            risk_level = 'high'
        elif risk_density >= 2.0 or risk_counts.get('medium_risk', 0) >= 2:
            risk_level = 'medium'
        elif risk_density >= 0.5:
            risk_level = 'low'
        else:
            risk_level = 'minimal'
        
        return risk_level, min(10.0, risk_density), risk_counts
            
    except Exception as e:
        logger.error(f"Error assessing document risk: {str(e)}")
        return 'medium', 5.0, {}  # Default to medium risk

def extract_key_entities(content: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract key entities with enhanced patterns and metadata
    
    Args:
        content: Document text content
        
    Returns:
        Dictionary of entity types with detailed information
    """
    try:
        entities = {
            'monetary_amounts': [],
            'dates': [],
            'percentages': [],
            'organizations': [],
            'people': [],
            'addresses': [],
            'phone_numbers': [],
            'email_addresses': [],
            'legal_references': [],
            'time_periods': []
        }
        
        # Enhanced money pattern matching
        money_patterns = [
            (r'\$[\d,]+(?:\.\d{2})?', 'USD'),
            (r'USD\s*[\d,]+(?:\.\d{2})?', 'USD'),
            (r'₹[\d,]+(?:\.\d{2})?', 'INR'),
            (r'INR\s*[\d,]+(?:\.\d{2})?', 'INR'),
            (r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD)\b', 'USD'),
            (r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:rupees?|INR)\b', 'INR')
        ]
        
        for pattern, currency in money_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                entities['monetary_amounts'].append({
                    'text': match.group(),
                    'currency': currency,
                    'position': match.span()
                })
        
        # Enhanced date patterns
        date_patterns = [
            (r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', 'MM/DD/YYYY or DD/MM/YYYY'),
            (r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b', 'YYYY/MM/DD'),
            (r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', 'Month DD, YYYY'),
            (r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b', 'DD Month YYYY')
        ]
        
        for pattern, format_type in date_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                entities['dates'].append({
                    'text': match.group(),
                    'format': format_type,
                    'position': match.span()
                })
        
        # Percentages
        percentage_matches = re.finditer(r'\b\d+(?:\.\d+)?%\b', content)
        for match in percentage_matches:
            entities['percentages'].append({
                'text': match.group(),
                'value': float(match.group().replace('%', '')),
                'position': match.span()
            })
        
        # Organizations (enhanced patterns)
        org_patterns = [
            r'\b[A-Z][a-zA-Z\s&,-]+(?:Inc\.?|LLC\.?|Corp\.?|Corporation|Company|Co\.)\b',
            r'\b[A-Z][a-zA-Z\s&,-]+(?:Ltd\.?|Limited|LLP|LP)\b',
            r'\b[A-Z][a-zA-Z\s&,-]+(?:Foundation|Trust|Association|Institute)\b'
        ]
        
        for pattern in org_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                entities['organizations'].append({
                    'text': match.group().strip(),
                    'position': match.span()
                })
        
        # Phone numbers
        phone_pattern = r'(?:\+1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        phone_matches = re.finditer(phone_pattern, content)
        for match in phone_matches:
            entities['phone_numbers'].append({
                'text': match.group(),
                'formatted': f"({match.group(1)}) {match.group(2)}-{match.group(3)}",
                'position': match.span()
            })
        
        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_matches = re.finditer(email_pattern, content)
        for match in email_matches:
            entities['email_addresses'].append({
                'text': match.group(),
                'domain': match.group().split('@')[1],
                'position': match.span()
            })
        
        # Legal references
        legal_ref_patterns = [
            r'Section\s+\d+(?:\.\d+)*',
            r'Article\s+[IVX]+|\bArticle\s+\d+',
            r'Clause\s+\d+(?:\.\d+)*',
            r'Paragraph\s+\d+(?:\.\d+)*',
            r'Schedule\s+[A-Z]|\bSchedule\s+\d+',
            r'Exhibit\s+[A-Z]|\bExhibit\s+\d+'
        ]
        
        for pattern in legal_ref_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                entities['legal_references'].append({
                    'text': match.group(),
                    'position': match.span()
                })
        
        # Time periods
        time_patterns = [
            (r'\b\d+\s*(?:days?|weeks?|months?|years?)\b', 'duration'),
            (r'\b(?:daily|weekly|monthly|quarterly|annually|yearly)\b', 'frequency'),
            (r'\b\d+\s*(?:hours?|minutes?)\b', 'short_duration')
        ]
        
        for pattern, period_type in time_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                entities['time_periods'].append({
                    'text': match.group(),
                    'type': period_type,
                    'position': match.span()
                })
        
        # Remove duplicates and limit results
        for entity_type in entities:
            seen = set()
            unique_entities = []
            for entity in entities[entity_type]:
                entity_text = entity['text'].lower()
                if entity_text not in seen:
                    seen.add(entity_text)
                    unique_entities.append(entity)
            entities[entity_type] = unique_entities[:15]  # Limit to 15 per type
        
        return entities
        
    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}")
        return {}

def extract_key_topics(content: str, max_topics: int = 10) -> List[Dict[str, Any]]:
    """
    Extract key topics with enhanced analysis and scoring
    
    Args:
        content: Document text content
        max_topics: Maximum number of topics to return
        
    Returns:
        List of topic dictionaries with scores and context
    """
    try:
        content_lower = content.lower()
        
        # Enhanced text processing
        # Remove common legal boilerplate
        boilerplate_patterns = [
            r'witness whereof.*?executed',
            r'in witness whereof.*?day of',
            r'this agreement.*?binding upon',
            r'entire agreement.*?supersedes'
        ]
        
        cleaned_content = content_lower
        for pattern in boilerplate_patterns:
            cleaned_content = re.sub(pattern, ' ', cleaned_content, flags=re.IGNORECASE)
        
        # Extract meaningful words (3+ letters, not stop words)
        words = re.findall(r'\b[a-z]{3,}\b', cleaned_content)
        
        # Enhanced stop words list
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 
            'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 
            'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'this', 'that', 'with', 
            'have', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 
            'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 
            'well', 'were', 'will', 'shall', 'may', 'any', 'said', 'each', 'which', 'their', 'what', 'there',
            'would', 'could', 'should', 'being', 'been', 'other', 'after', 'first', 'also', 'back', 'only',
            'hereby', 'whereas', 'therefore', 'pursuant', 'accordance', 'including', 'without', 'limitation'
        }
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get legal-relevant terms with categories
        legal_term_categories = {
            'contract_terms': ['agreement', 'contract', 'obligation', 'consideration', 'covenant', 'undertaking'],
            'payment_terms': ['payment', 'fee', 'cost', 'expense', 'interest', 'penalty'],
            'liability_terms': ['liability', 'indemnity', 'damages', 'breach', 'default', 'termination'],
            'property_terms': ['property', 'title', 'ownership', 'possession', 'rights', 'license'],
            'time_terms': ['term', 'duration', 'period', 'renewal', 'expiration', 'notice'],
            'dispute_terms': ['dispute', 'arbitration', 'litigation', 'jurisdiction', 'governing', 'resolution']
        }
        
        # Score topics with category bonuses
        scored_topics = []
        
        for word, count in word_freq.items():
            if count >= 2:  # Minimum frequency
                score = count
                category = 'general'
                
                # Check if word belongs to a legal category
                for cat_name, cat_words in legal_term_categories.items():
                    if word in cat_words:
                        score *= 2.0  # Boost legal terms
                        category = cat_name
                        break
                
                # Position bonus (words appearing early get slight boost)
                first_occurrence = cleaned_content.find(word)
                if first_occurrence != -1:
                    position_ratio = first_occurrence / len(cleaned_content)
                    if position_ratio < 0.2:  # First 20% of document
                        score *= 1.2
                
                scored_topics.append({
                    'topic': word,
                    'frequency': count,
                    'score': score,
                    'category': category,
                    'importance': 'high' if score >= 8 else 'medium' if score >= 4 else 'low'
                })
        
        # Sort by score and return top topics
        scored_topics.sort(key=lambda x: x['score'], reverse=True)
        return scored_topics[:max_topics]
        
    except Exception as e:
        logger.error(f"Error extracting topics: {str(e)}")
        return []

def generate_document_summary(content: str, max_sentences: int = 3, target_length: int = 300) -> Dict[str, Any]:
    """
    Generate enhanced document summary with multiple strategies
    
    Args:
        content: Document text content
        max_sentences: Maximum number of sentences in summary
        target_length: Target character length for summary
        
    Returns:
        Dictionary with summary and metadata
    """
    try:
        # Clean and prepare text
        cleaned_content = clean_text_content(content)
        
        # Split into sentences
        sentence_endings = r'[.!?]+(?:\s+|$)'
        sentences = re.split(sentence_endings, cleaned_content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return {
                'summary': 'Document content could not be summarized.',
                'method': 'fallback',
                'confidence': 0.0,
                'sentence_count': 0,
                'character_count': 0
            }
        
        if len(sentences) <= max_sentences:
            summary_text = '. '.join(sentences[:max_sentences])
            return {
                'summary': summary_text + '.' if not summary_text.endswith('.') else summary_text,
                'method': 'complete',
                'confidence': 1.0,
                'sentence_count': len(sentences),
                'character_count': len(summary_text)
            }
        
        # Score sentences using multiple criteria
        sentence_scores = {}
        
        # Get important keywords for scoring
        topics = extract_key_topics(content, max_topics=20)
        important_words = {topic['topic'] for topic in topics[:10]}
        
        for i, sentence in enumerate(sentences):
            score = 0
            sentence_lower = sentence.lower()
            sentence_words = set(sentence_lower.split())
            
            # Position scoring (earlier sentences get higher scores)
            position_score = max(0, 20 - i) * 0.1
            score += position_score
            
            # Length scoring (prefer moderate length sentences)
            word_count = len(sentence.split())
            if 10 <= word_count <= 35:
                score += 2
            elif 8 <= word_count <= 40:
                score += 1
            
            # Important keyword scoring
            keyword_matches = len(sentence_words & important_words)
            score += keyword_matches * 1.5
            
            # Legal term scoring
            legal_terms_found = 0
            for category in LEGAL_DOCUMENT_KEYWORDS.values():
                for term in category:
                    if term in sentence_lower:
                        legal_terms_found += 1
            score += legal_terms_found * 0.5
            
            # First/last sentence bonus
            if i == 0:
                score += 3  # First sentence often important
            elif i == len(sentences) - 1:
                score += 1  # Last sentence sometimes important
            
            # Avoid sentences that are too similar to others
            similarity_penalty = 0
            for j, other_sentence in enumerate(sentences[:i]):
                if calculate_text_similarity(sentence, other_sentence) > 0.7:
                    similarity_penalty += 0.5
            score -= similarity_penalty
            
            sentence_scores[i] = score
        
        # Select best sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
        top_sentences.sort(key=lambda x: x[0])  # Maintain original order
        
        # Build summary
        summary_sentences = [sentences[i] for i, _ in top_sentences]
        summary = '. '.join(summary_sentences)
        
        # Ensure proper ending
        if not summary.endswith('.'):
            summary += '.'
        
        # Trim if too long
        if len(summary) > target_length:
            summary = summary[:target_length].rsplit(' ', 1)[0] + '...'
        
        # Clean up summary
        summary = re.sub(r'\s+', ' ', summary)
        summary = summary.strip()
        
        return {
            'summary': summary,
            'method': 'extractive_scoring',
            'confidence': min(1.0, max(sentence_scores.values()) / 10),
            'sentence_count': len(summary_sentences),
            'character_count': len(summary),
            'original_sentence_count': len(sentences)
        }
        
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return {
            'summary': content[:200] + "..." if len(content) > 200 else content,
            'method': 'fallback',
            'confidence': 0.0,
            'sentence_count': 0,
            'character_count': len(content[:203]),
            'error': str(e)
        }

# Enhanced utility functions
def generate_session_id() -> str:
    """Generate unique session ID with timestamp"""
    timestamp = int(time.time())
    return f"session_{timestamp}_{uuid.uuid4().hex[:12]}"

def generate_document_id(prefix: str = "doc") -> str:
    """Generate unique document ID with optional prefix"""
    timestamp = int(time.time())
    return f"{prefix}_{timestamp}_{uuid.uuid4().hex[:8]}"

def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """Generate unique chunk ID"""
    return f"{document_id}_chunk_{chunk_index:04d}"

def calculate_file_hash(content: bytes, algorithm: str = 'sha256') -> str:
    """
    Calculate hash of file content with selectable algorithm
    
    Args:
        content: File content as bytes
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256', 'sha512')
        
    Returns:
        Hex digest of hash
    """
    try:
        if algorithm == 'md5':
            return hashlib.md5(content).hexdigest()
        elif algorithm == 'sha1':
            return hashlib.sha1(content).hexdigest()
        elif algorithm == 'sha256':
            return hashlib.sha256(content).hexdigest()
        elif algorithm == 'sha512':
            return hashlib.sha512(content).hexdigest()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    except Exception as e:
        logger.error(f"Error calculating hash: {str(e)}")
        return ""

def calculate_content_hash(content: str, algorithm: str = 'md5') -> str:
    """Calculate hash of text content"""
    return calculate_file_hash(content.encode('utf-8'), algorithm)

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024.0 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    if i == 0:
        return f"{int(size_bytes)}{size_names[i]}"
    else:
        return f"{size_bytes:.1f}{size_names[i]}"

def estimate_reading_time(content: str, words_per_minute: int = 200) -> Dict[str, Any]:
    """
    Enhanced reading time estimation with different scenarios
    
    Args:
        content: Text content
        words_per_minute: Reading speed
        
    Returns:
        Dictionary with reading time estimates
    """
    try:
        word_count = len(content.split())
        
        # Different reading speeds
        speeds = {
            'slow': 150,      # Slow reader
            'average': 200,   # Average reader
            'fast': 300,      # Fast reader
            'legal': 120      # Legal document (complex content)
        }
        
        estimates = {}
        for speed_type, wpm in speeds.items():
            minutes = max(1, word_count / wpm)
            estimates[speed_type] = {
                'minutes': round(minutes, 1),
                'formatted': f"{int(minutes)}:{int((minutes % 1) * 60):02d}"
            }
        
        return {
            'word_count': word_count,
            'estimates': estimates,
            'recommended_speed': 'legal' if any(indicator in content.lower() for indicator in ['agreement', 'contract', 'legal']) else 'average'
        }
        
    except Exception as e:
        logger.error(f"Error estimating reading time: {str(e)}")
        return {
            'word_count': 0,
            'estimates': {'average': {'minutes': 1, 'formatted': '1:00'}},
            'error': str(e)
        }

def clean_text_content(content: str) -> str:
    """Enhanced text cleaning and normalization"""
    try:
        if not content:
            return ""
        
        # Remove control characters except tabs and newlines
        content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', content)
        
        # Normalize Unicode characters
        import unicodedata
        content = unicodedata.normalize('NFKC', content)
        
        # Fix common OCR errors
        ocr_fixes = {
            r'\b0\b': 'O',  # Zero to letter O in names
            r'\bl\b': 'I',  # lowercase l to uppercase I
            r'(?<=[a-z])1(?=[a-z])': 'l',  # 1 to l in middle of words
            r'\s+': ' ',  # Multiple spaces to single space
            r'([.!?])\s*([A-Z])': r'\1 \2',  # Ensure space after punctuation
        }
        
        for pattern, replacement in ocr_fixes.items():
            content = re.sub(pattern, replacement, content)
        
        # Normalize line endings and remove excessive blank lines
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        # Clean up common formatting artifacts
        content = re.sub(r'_{3,}', '___', content)  # Long underscores
        content = re.sub(r'-{3,}', '---', content)  # Long dashes
        content = re.sub(r'={3,}', '===', content)  # Long equals
        
        return content.strip()
        
    except Exception as e:
        logger.error(f"Error cleaning text content: {str(e)}")
        return content

def extract_document_metadata(content: str, filename: str, file_size: int) -> Dict[str, Any]:
    """
    Extract comprehensive document metadata with enhanced analysis
    
    Args:
        content: Document text content
        filename: Original filename
        file_size: File size in bytes
        
    Returns:
        Dictionary of enhanced metadata
    """
    try:
        # Basic metadata
        metadata = {
            'filename': filename,
            'file_extension': Path(filename).suffix.lower(),
            'file_size': file_size,
            'file_size_formatted': format_file_size(file_size),
            'content_hash': calculate_content_hash(content),
            'file_hash': calculate_file_hash(content.encode('utf-8')),
            'extraction_timestamp': datetime.now(timezone.utc).isoformat(),
            'processing_version': '2.0'
        }
        
        # Content analysis
        document_type, type_confidence = detect_document_type(content, filename)
        risk_level, risk_score, risk_breakdown = assess_document_risk_level(content, document_type)
        reading_time = estimate_reading_time(content)
        topics = extract_key_topics(content, max_topics=15)
        entities = extract_key_entities(content)
        summary_data = generate_document_summary(content)
        
        # Enhanced content metrics
        lines = content.split('\n')
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        metadata.update({
            # Document classification
            'document_type': document_type,
            'document_type_confidence': type_confidence,
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_breakdown': risk_breakdown,
            
            # Content metrics
            'word_count': len(content.split()),
            'character_count': len(content),
            'character_count_no_spaces': len(content.replace(' ', '')),
            'line_count': len(lines),
            'paragraph_count': len(paragraphs),
            'sentence_count': len(sentences),
            'average_sentence_length': len(content.split()) / max(len(sentences), 1),
            'average_paragraph_length': len(content.split()) / max(len(paragraphs), 1),
            
            # Reading analysis
            'reading_time': reading_time,
            'complexity_score': calculate_text_complexity(content),
            'formality_score': calculate_formality_score(content),
            
            # Content features
            'key_topics': topics,
            'key_entities': entities,
            'summary': summary_data,
            'language': detect_language(content),
            
            # Document features
            'contains_tables': detect_tables(content),
            'contains_signatures': detect_signatures(content),
            'contains_dates': len(entities.get('dates', [])) > 0,
            'contains_monetary_amounts': len(entities.get('monetary_amounts', [])) > 0,
            'contains_legal_references': len(entities.get('legal_references', [])) > 0,
            
            # Quality indicators
            'completeness_score': assess_document_completeness(content, document_type),
            'quality_indicators': assess_content_quality(content),
            
            # Estimates
            'page_estimate': max(1, len(content.split()) // 250),  # ~250 words per page
            'processing_difficulty': get_processing_difficulty(content, file_size),
        })
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error extracting document metadata: {str(e)}")
        return {
            'filename': filename,
            'file_size': file_size,
            'extraction_timestamp': datetime.now(timezone.utc).isoformat(),
            'error': str(e),
            'error_type': 'metadata_extraction_failed'
        }

# New helper functions for enhanced metadata
def calculate_text_complexity(content: str) -> float:
    """
    Calculate text complexity score based on sentence length and vocabulary
    
    Returns:
        Complexity score from 0.0 (simple) to 1.0 (complex)
    """
    try:
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.5
        
        # Average sentence length
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # Vocabulary diversity (unique words / total words)
        words = content.lower().split()
        unique_words = set(words)
        vocab_diversity = len(unique_words) / max(len(words), 1)
        
        # Complex word ratio (words > 6 characters)
        complex_words = sum(1 for word in words if len(word) > 6)
        complex_word_ratio = complex_words / max(len(words), 1)
        
        # Legal term density
        legal_term_count = 0
        content_lower = content.lower()
        for category in LEGAL_DOCUMENT_KEYWORDS.values():
            for term in category:
                legal_term_count += content_lower.count(term)
        legal_density = legal_term_count / max(len(words), 1)
        
        # Combine factors
        complexity = (
            min(avg_sentence_length / 20, 1.0) * 0.3 +  # Sentence length factor
            vocab_diversity * 0.2 +  # Vocabulary diversity factor
            complex_word_ratio * 0.3 +  # Complex words factor
            min(legal_density * 10, 1.0) * 0.2  # Legal terms factor
        )
        
        return min(1.0, max(0.0, complexity))
        
    except Exception as e:
        logger.error(f"Error calculating text complexity: {str(e)}")
        return 0.5

def calculate_formality_score(content: str) -> float:
    """
    Calculate formality score of the text
    
    Returns:
        Formality score from 0.0 (informal) to 1.0 (formal)
    """
    try:
        content_lower = content.lower()
        words = content_lower.split()
        
        if not words:
            return 0.5
        
        # Formal indicators
        formal_words = {
            'shall', 'hereby', 'whereas', 'therefore', 'pursuant', 'notwithstanding',
            'furthermore', 'moreover', 'nevertheless', 'subsequently', 'accordingly',
            'consequently', 'respectively', 'heretofore', 'hereafter', 'aforementioned'
        }
        
        # Informal indicators  
        informal_words = {
            'gonna', 'wanna', 'gotta', 'yeah', 'ok', 'okay', 'stuff', 'things',
            'guys', 'folks', 'pretty', 'really', 'very', 'quite', 'sort of', 'kind of'
        }
        
        formal_count = sum(1 for word in words if word in formal_words)
        informal_count = sum(1 for word in words if word in informal_words)
        
        # Passive voice indicators
        passive_indicators = ['is', 'are', 'was', 'were', 'been', 'being']
        past_participles = re.findall(r'\b\w+ed\b', content_lower)
        passive_score = len(past_participles) / max(len(words), 1)
        
        # Calculate formality
        formal_ratio = formal_count / max(len(words), 1)
        informal_ratio = informal_count / max(len(words), 1)
        
        formality = (
            formal_ratio * 10 +  # Boost formal words
            passive_score * 5 +  # Passive voice adds formality
            -informal_ratio * 10  # Informal words reduce formality
        )
        
        return min(1.0, max(0.0, formality))
        
    except Exception as e:
        logger.error(f"Error calculating formality score: {str(e)}")
        return 0.5

def detect_language(content: str) -> str:
    """
    Simple language detection based on character patterns
    
    Returns:
        Detected language code
    """
    try:
        # Check for non-Latin scripts
        if re.search(r'[\u0900-\u097F]', content):  # Devanagari
            return 'hi'
        elif re.search(r'[\u0B80-\u0BFF]', content):  # Tamil
            return 'ta'
        elif re.search(r'[\u0C00-\u0C7F]', content):  # Telugu
            return 'te'
        elif re.search(r'[\u0980-\u09FF]', content):  # Bengali
            return 'bn'
        elif re.search(r'[\u0600-\u06FF]', content):  # Arabic
            return 'ar'
        else:
            return 'en'  # Default to English
            
    except Exception as e:
        logger.error(f"Error detecting language: {str(e)}")
        return 'en'

def detect_tables(content: str) -> Dict[str, Any]:
    """
    Detect presence of tables in content
    
    Returns:
        Dictionary with table detection results
    """
    try:
        # Look for table indicators
        table_indicators = [
            len(re.findall(r'\|', content)),  # Pipe characters
            len(re.findall(r'\t.*\t', content)),  # Tab-separated values
            len(re.findall(r'^\s*\w+\s*:\s*\w+\s*$', content, re.MULTILINE)),  # Key-value pairs
        ]
        
        has_tables = any(count > 5 for count in table_indicators)
        
        return {
            'has_tables': has_tables,
            'pipe_count': table_indicators[0],
            'tab_separated_lines': table_indicators[1],
            'key_value_pairs': table_indicators[2],
            'confidence': 'high' if sum(table_indicators) > 20 else 'medium' if sum(table_indicators) > 10 else 'low'
        }
        
    except Exception as e:
        logger.error(f"Error detecting tables: {str(e)}")
        return {'has_tables': False, 'confidence': 'unknown', 'error': str(e)}

def detect_signatures(content: str) -> Dict[str, Any]:
    """
    Detect signature-related content
    
    Returns:
        Dictionary with signature detection results
    """
    try:
        signature_patterns = [
            r'signature\s*:?\s*_+',
            r'signed\s*:?\s*_+',
            r'date\s*:?\s*_+',
            r'print\s+name\s*:?\s*_+',
            r'executed.*day.*month.*year',
            r'in\s+witness\s+whereof',
            r'/_+/',  # Signature lines
        ]
        
        signature_indicators = []
        for pattern in signature_patterns:
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            signature_indicators.append(matches)
        
        total_indicators = sum(signature_indicators)
        has_signatures = total_indicators > 0
        
        return {
            'has_signatures': has_signatures,
            'signature_line_count': signature_indicators[0] + signature_indicators[1],
            'date_lines': signature_indicators[2],
            'name_lines': signature_indicators[3],
            'execution_clauses': signature_indicators[4] + signature_indicators[5],
            'total_indicators': total_indicators,
            'confidence': 'high' if total_indicators > 3 else 'medium' if total_indicators > 1 else 'low'
        }
        
    except Exception as e:
        logger.error(f"Error detecting signatures: {str(e)}")
        return {'has_signatures': False, 'confidence': 'unknown', 'error': str(e)}

def assess_content_quality(content: str) -> Dict[str, Any]:
    """
    Assess overall content quality
    
    Returns:
        Dictionary with quality indicators
    """
    try:
        quality_indicators = {
            'has_sufficient_content': len(content.split()) >= 100,
            'has_proper_structure': bool(re.search(r'\n\s*\n', content)),  # Has paragraphs
            'has_punctuation': bool(re.search(r'[.!?]', content)),
            'has_capitalization': bool(re.search(r'[A-Z]', content)),
            'reasonable_char_ratio': 0.3 <= (sum(1 for c in content if c.isalpha()) / max(len(content), 1)) <= 0.9,
            'no_excessive_repetition': len(set(content.split())) / max(len(content.split()), 1) > 0.3
        }
        
        quality_score = sum(quality_indicators.values()) / len(quality_indicators)
        
        return {
            'quality_score': quality_score,
            'quality_level': 'high' if quality_score > 0.8 else 'medium' if quality_score > 0.5 else 'low',
            'indicators': quality_indicators,
            'recommendations': _get_quality_recommendations(quality_indicators)
        }
        
    except Exception as e:
        logger.error(f"Error assessing content quality: {str(e)}")
        return {'quality_score': 0.5, 'quality_level': 'unknown', 'error': str(e)}

def get_processing_difficulty(content: str, file_size: int) -> Dict[str, Any]:
    """
    Estimate processing difficulty
    
    Returns:
        Dictionary with difficulty assessment
    """
    try:
        factors = {
            'size': 'high' if file_size > 10*1024*1024 else 'medium' if file_size > 1*1024*1024 else 'low',
            'length': 'high' if len(content) > 50000 else 'medium' if len(content) > 10000 else 'low',
            'complexity': 'high' if calculate_text_complexity(content) > 0.7 else 'medium' if calculate_text_complexity(content) > 0.4 else 'low',
            'structure': 'high' if not detect_tables(content)['has_tables'] and not detect_signatures(content)['has_signatures'] else 'low'
        }
        
        difficulty_scores = {'low': 1, 'medium': 2, 'high': 3}
        total_score = sum(difficulty_scores[level] for level in factors.values())
        avg_score = total_score / len(factors)
        
        if avg_score >= 2.5:
            overall_difficulty = 'high'
        elif avg_score >= 1.5:
            overall_difficulty = 'medium'
        else:
            overall_difficulty = 'low'
        
        return {
            'overall_difficulty': overall_difficulty,
            'factors': factors,
            'estimated_processing_time': {
                'low': '10-30 seconds',
                'medium': '30-60 seconds', 
                'high': '1-3 minutes'
            }[overall_difficulty]
        }
        
    except Exception as e:
        logger.error(f"Error assessing processing difficulty: {str(e)}")
        return {'overall_difficulty': 'medium', 'error': str(e)}

def _get_quality_recommendations(indicators: Dict[str, bool]) -> List[str]:
    """Get quality improvement recommendations"""
    recommendations = []
    
    if not indicators.get('has_sufficient_content'):
        recommendations.append("Document appears to be very short. Ensure complete content was extracted.")
    
    if not indicators.get('has_proper_structure'):
        recommendations.append("Document lacks clear paragraph structure. Check formatting.")
    
    if not indicators.get('has_punctuation'):
        recommendations.append("Document lacks punctuation. May indicate OCR or extraction issues.")
    
    if not indicators.get('reasonable_char_ratio'):
        recommendations.append("Unusual character distribution detected. Check for encoding issues.")
    
    if not indicators.get('no_excessive_repetition'):
        recommendations.append("Excessive repetition detected. Check for duplicate content or extraction errors.")
    
    return recommendations

# Validation functions (enhanced)
def validate_session_request(request_data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Enhanced session request validation
    
    Args:
        request_data: Request data dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check required fields
        required_fields = ['session_id']
        for field in required_fields:
            if field not in request_data or not request_data[field]:
                return False, f"Missing required field: {field}"
        
        # Validate session ID format
        session_id = request_data['session_id']
        if not re.match(r'^session_\d+_[a-f0-9]{12}$', session_id):
            return False, "Invalid session ID format"
        
        # Optional field validation
        if 'language' in request_data:
            supported_languages = ['en', 'hi', 'ta', 'te', 'bn', 'gu', 'kn', 'ml', 'mr', 'or', 'pa', 'ur']
            if request_data['language'] not in supported_languages:
                return False, f"Unsupported language: {request_data['language']}"
        
        return True, "Validation successful"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def create_error_response(error_type: str, message: str, details: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create standardized error response with enhanced details
    
    Args:
        error_type: Type of error
        message: Error message
        details: Additional error details
        
    Returns:
        Enhanced error response dictionary
    """
    return {
        'success': False,
        'error': {
            'type': error_type,
            'message': message,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'details': details or {},
            'request_id': str(uuid.uuid4())
        },
        'suggestions': _get_error_suggestions(error_type),
        'retry_after': _get_retry_delay(error_type)
    }

def create_success_response(data: Dict[str, Any], message: str = "Success") -> Dict[str, Any]:
    """
    Create standardized success response with enhanced metadata
    
    Args:
        data: Response data
        message: Success message
        
    Returns:
        Enhanced success response dictionary
    """
    return {
        'success': True,
        'message': message,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'request_id': str(uuid.uuid4()),
        'data': data,
        'meta': {
            'processing_time': data.get('processing_time', 0),
            'version': '2.0'
        }
    }

def _get_error_suggestions(error_type: str) -> List[str]:
    """Get suggestions based on error type"""
    suggestions_map = {
        'validation_error': [
            "Check that all required fields are provided",
            "Verify data formats match expected patterns"
        ],
        'file_too_large': [
            "Reduce file size or split into smaller documents",
            "Compress the document if possible"
        ],
        'unsupported_format': [
            "Convert to supported format (PDF, DOCX, TXT)",
            "Check file extension matches content type"
        ],
        'processing_timeout': [
            "Try again with a smaller document",
            "Retry the request after a few minutes"
        ]
    }
    return suggestions_map.get(error_type, ["Contact support if the issue persists"])

def _get_retry_delay(error_type: str) -> Optional[int]:
    """Get suggested retry delay in seconds"""
    retry_delays = {
        'rate_limit_exceeded': 60,
        'processing_timeout': 30,
        'server_error': 10
    }
    return retry_delays.get(error_type)

# Document comparison utilities (enhanced)
def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Enhanced text similarity calculation using multiple methods
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    try:
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts
        text1_clean = clean_text_content(text1.lower())
        text2_clean = clean_text_content(text2.lower())
        
        # Word-based similarity (Jaccard)
        words1 = set(text1_clean.split())
        words2 = set(text2_clean.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        jaccard = len(words1 & words2) / len(words1 | words2)
        
        # Character-based similarity
        chars1 = set(text1_clean.replace(' ', ''))
        chars2 = set(text2_clean.replace(' ', ''))
        
        char_similarity = len(chars1 & chars2) / max(len(chars1 | chars2), 1)
        
        # Length similarity
        len1, len2 = len(text1_clean), len(text2_clean)
        length_similarity = 1 - abs(len1 - len2) / max(len1, len2, 1)
        
        # Combined similarity
        combined_similarity = (jaccard * 0.5) + (char_similarity * 0.3) + (length_similarity * 0.2)
        
        return min(1.0, max(0.0, combined_similarity))
        
    except Exception as e:
        logger.error(f"Error calculating text similarity: {str(e)}")
        return 0.0

def find_common_topics(topics1: List[str], topics2: List[str]) -> Dict[str, Any]:
    """
    Enhanced common topic finding with similarity matching
    
    Args:
        topics1: First list of topics
        topics2: Second list of topics
        
    Returns:
        Dictionary with common topics and analysis
    """
    try:
        exact_matches = list(set(topics1) & set(topics2))
        
        # Find similar topics (not exact matches)
        similar_matches = []
        for topic1 in topics1:
            for topic2 in topics2:
                if topic1 != topic2 and calculate_text_similarity(topic1, topic2) > 0.7:
                    similar_matches.append((topic1, topic2))
        
        return {
            'exact_matches': exact_matches,
            'similar_matches': similar_matches,
            'total_common': len(exact_matches) + len(similar_matches),
            'similarity_score': (len(exact_matches) + len(similar_matches) * 0.7) / max(len(set(topics1 + topics2)), 1)
        }
        
    except Exception as e:
        logger.error(f"Error finding common topics: {str(e)}")
        return {'exact_matches': [], 'similar_matches': [], 'total_common': 0, 'similarity_score': 0.0}

def assess_document_completeness(content: str, document_type: str) -> float:
    """
    Enhanced document completeness assessment
    
    Args:
        content: Document content
        document_type: Type of document
        
    Returns:
        Completeness score (0.0 to 1.0)
    """
    try:
        completeness_score = 0.0
        
        # Document type specific requirements
        required_sections = {
            'lease': ['landlord', 'tenant', 'rent', 'term', 'premises'],
            'employment': ['employee', 'employer', 'duties', 'compensation'],
            'loan': ['borrower', 'lender', 'amount', 'interest', 'repayment'],
            'nda': ['confidential', 'parties', 'information'],
            'service': ['services', 'provider', 'client', 'payment']
        }
        
        # General document requirements
        has_header = any(indicator in content[:500] for indicator in ['agreement', 'contract'])
        has_signature_section = any(indicator in content[-500:] for indicator in ['signature', 'signed', 'executed'])
        has_date = bool(re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', content))
        has_parties = len(re.findall(r'\bparty\b|\bparties\b', content, re.IGNORECASE)) >= 2
        
        # Basic structure scoring
                # Basic structure scoring
        if has_header:
            completeness_score += 0.25
        if has_signature_section:
            completeness_score += 0.25
        if has_date:
            completeness_score += 0.15
        if has_parties:
            completeness_score += 0.15
        
        # Document type specific scoring
        if document_type in required_sections:
            content_lower = content.lower()
            required = required_sections[document_type]
            found_sections = sum(1 for section in required if section in content_lower)
            type_completeness = found_sections / len(required)
            completeness_score += type_completeness * 0.2
        
        # Length and content depth scoring
        word_count = len(content.split())
        if word_count > 1000:
            completeness_score += 0.1
        elif word_count > 500:
            completeness_score += 0.05
        
        # Legal formality indicators
        formal_indicators = ['whereas', 'hereby', 'therefore', 'shall', 'party', 'agreement']
        formal_count = sum(1 for indicator in formal_indicators if indicator in content.lower())
        if formal_count >= 3:
            completeness_score += 0.1
        
        return min(1.0, completeness_score)
        
    except Exception as e:
        logger.error(f"Error assessing document completeness: {str(e)}")
        return 0.5

# Enhanced Performance monitoring utilities
class DocumentProcessingMetrics:
    """Enhanced document processing metrics with detailed tracking"""
    
    def __init__(self):
        self.metrics = {
            'documents_processed': 0,
            'total_processing_time': 0,
            'average_processing_time': 0,
            'errors_count': 0,
            'success_rate': 0,
            'processing_stages': {
                'validation': {'count': 0, 'total_time': 0, 'errors': 0},
                'extraction': {'count': 0, 'total_time': 0, 'errors': 0},
                'analysis': {'count': 0, 'total_time': 0, 'errors': 0},
                'summarization': {'count': 0, 'total_time': 0, 'errors': 0}
            },
            'document_types': {},
            'file_sizes': {'small': 0, 'medium': 0, 'large': 0},
            'quality_scores': [],
            'risk_levels': {'minimal': 0, 'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        }
        self.start_time = None
        self.current_stage = None
        self.stage_start_time = None
        self.session_start = datetime.now(timezone.utc)
    
    def start_processing(self, document_type: str = 'unknown', file_size: int = 0):
        """Start timing document processing with context"""
        self.start_time = datetime.now(timezone.utc)
        self.current_document_type = document_type
        self.current_file_size = file_size
        
        # Track file size category
        if file_size < 1024 * 1024:  # < 1MB
            self.metrics['file_sizes']['small'] += 1
        elif file_size < 10 * 1024 * 1024:  # < 10MB
            self.metrics['file_sizes']['medium'] += 1
        else:
            self.metrics['file_sizes']['large'] += 1
    
    def start_stage(self, stage_name: str):
        """Start timing a processing stage"""
        if self.current_stage:
            self.end_stage(success=True)  # End previous stage
        
        self.current_stage = stage_name
        self.stage_start_time = datetime.now(timezone.utc)
        
        if stage_name in self.metrics['processing_stages']:
            self.metrics['processing_stages'][stage_name]['count'] += 1
    
    def end_stage(self, success: bool = True):
        """End timing current stage"""
        if self.current_stage and self.stage_start_time:
            stage_time = (datetime.now(timezone.utc) - self.stage_start_time).total_seconds()
            
            stage_metrics = self.metrics['processing_stages'][self.current_stage]
            stage_metrics['total_time'] += stage_time
            
            if not success:
                stage_metrics['errors'] += 1
            
            self.current_stage = None
            self.stage_start_time = None
    
    def end_processing(self, success: bool = True, quality_score: float = None, risk_level: str = None):
        """End timing and update comprehensive metrics"""
        if self.current_stage:
            self.end_stage(success)
        
        if self.start_time:
            processing_time = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            
            self.metrics['documents_processed'] += 1
            self.metrics['total_processing_time'] += processing_time
            self.metrics['average_processing_time'] = (
                self.metrics['total_processing_time'] / self.metrics['documents_processed']
            )
            
            # Track document type
            doc_type = getattr(self, 'current_document_type', 'unknown')
            if doc_type not in self.metrics['document_types']:
                self.metrics['document_types'][doc_type] = 0
            self.metrics['document_types'][doc_type] += 1
            
            # Track quality score
            if quality_score is not None:
                self.metrics['quality_scores'].append(quality_score)
            
            # Track risk level
            if risk_level and risk_level in self.metrics['risk_levels']:
                self.metrics['risk_levels'][risk_level] += 1
            
            if not success:
                self.metrics['errors_count'] += 1
            
            self.metrics['success_rate'] = (
                (self.metrics['documents_processed'] - self.metrics['errors_count']) / 
                self.metrics['documents_processed']
            ) * 100
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive current metrics"""
        current_time = datetime.now(timezone.utc)
        session_duration = (current_time - self.session_start).total_seconds()
        
        # Calculate stage performance
        stage_performance = {}
        for stage, data in self.metrics['processing_stages'].items():
            if data['count'] > 0:
                stage_performance[stage] = {
                    'average_time': data['total_time'] / data['count'],
                    'error_rate': (data['errors'] / data['count']) * 100,
                    'total_processed': data['count']
                }
        
        # Calculate quality statistics
        quality_stats = {}
        if self.metrics['quality_scores']:
            quality_stats = {
                'average': sum(self.metrics['quality_scores']) / len(self.metrics['quality_scores']),
                'min': min(self.metrics['quality_scores']),
                'max': max(self.metrics['quality_scores']),
                'count': len(self.metrics['quality_scores'])
            }
        
        return {
            **self.metrics,
            'session_duration': session_duration,
            'documents_per_hour': (self.metrics['documents_processed'] / max(session_duration / 3600, 0.001)),
            'stage_performance': stage_performance,
            'quality_statistics': quality_stats,
            'memory_usage': self._get_memory_usage(),
            'performance_grade': self._calculate_performance_grade()
        }
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
                'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0, 'unavailable': True}
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_performance_grade(self) -> str:
        """Calculate overall performance grade"""
        try:
            score = 0
            
            # Success rate (40% of grade)
            if self.metrics['success_rate'] >= 95:
                score += 40
            elif self.metrics['success_rate'] >= 90:
                score += 35
            elif self.metrics['success_rate'] >= 80:
                score += 25
            else:
                score += 10
            
            # Speed (30% of grade)
            avg_time = self.metrics['average_processing_time']
            if avg_time <= 30:
                score += 30
            elif avg_time <= 60:
                score += 25
            elif avg_time <= 120:
                score += 15
            else:
                score += 5
            
            # Quality (30% of grade)
            if self.metrics['quality_scores']:
                avg_quality = sum(self.metrics['quality_scores']) / len(self.metrics['quality_scores'])
                if avg_quality >= 0.8:
                    score += 30
                elif avg_quality >= 0.6:
                    score += 20
                else:
                    score += 10
            else:
                score += 15  # Neutral score if no quality data
            
            # Determine grade
            if score >= 85:
                return 'A'
            elif score >= 75:
                return 'B'
            elif score >= 65:
                return 'C'
            elif score >= 50:
                return 'D'
            else:
                return 'F'
                
        except Exception as e:
            logger.error(f"Error calculating performance grade: {str(e)}")
            return 'Unknown'
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.__init__()
    
    def export_metrics(self, format_type: str = 'json') -> str:
        """Export metrics in specified format"""
        metrics = self.get_metrics()
        
        if format_type.lower() == 'json':
            import json
            return json.dumps(metrics, indent=2, default=str)
        elif format_type.lower() == 'csv':
            # Simple CSV export of key metrics
            csv_data = "Metric,Value\n"
            csv_data += f"Documents Processed,{metrics['documents_processed']}\n"
            csv_data += f"Success Rate,{metrics['success_rate']:.1f}%\n"
            csv_data += f"Average Processing Time,{metrics['average_processing_time']:.2f}s\n"
            csv_data += f"Performance Grade,{metrics['performance_grade']}\n"
            return csv_data
        else:
            return str(metrics)

# Batch processing utilities
class BatchDocumentProcessor:
    """Process multiple documents with progress tracking and error handling"""
    
    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self.metrics = DocumentProcessingMetrics()
        self.logger = logger
        self.processed_documents = []
        self.failed_documents = []
    
    async def process_documents(self, document_list: List[Dict[str, Any]], 
                              progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Process multiple documents concurrently
        
        Args:
            document_list: List of document dictionaries with 'content', 'filename', 'file_size'
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Batch processing results
        """
        try:
            total_documents = len(document_list)
            if total_documents == 0:
                return {'success': True, 'processed': 0, 'failed': 0, 'results': []}
            
            self.logger.info(f"Starting batch processing of {total_documents} documents")
            
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(self.max_concurrent)
            processed_count = 0
            
            async def process_single_document(doc_data: Dict[str, Any], index: int):
                nonlocal processed_count
                
                async with semaphore:
                    try:
                        # Start metrics tracking
                        self.metrics.start_processing(
                            document_type=doc_data.get('document_type', 'unknown'),
                            file_size=doc_data.get('file_size', 0)
                        )
                        
                        # Process document
                        result = await self._process_single_document(doc_data, index)
                        
                        # Update tracking
                        processed_count += 1
                        self.processed_documents.append(result)
                        self.metrics.end_processing(
                            success=True,
                            quality_score=result.get('quality_score'),
                            risk_level=result.get('risk_level')
                        )
                        
                        # Progress callback
                        if progress_callback:
                            await progress_callback(processed_count, total_documents, result)
                        
                        return result
                        
                    except Exception as e:
                        error_result = {
                            'index': index,
                            'filename': doc_data.get('filename', f'document_{index}'),
                            'success': False,
                            'error': str(e),
                            'error_type': type(e).__name__
                        }
                        
                        self.failed_documents.append(error_result)
                        self.metrics.end_processing(success=False)
                        processed_count += 1
                        
                        if progress_callback:
                            await progress_callback(processed_count, total_documents, error_result)
                        
                        self.logger.error(f"Error processing document {index}: {str(e)}")
                        return error_result
            
            # Process all documents concurrently
            tasks = [process_single_document(doc, i) for i, doc in enumerate(document_list)]
            results = await asyncio.gather(*tasks, return_exceptions=False)
            
            # Compile final results
            successful_results = [r for r in results if r.get('success', False)]
            failed_results = [r for r in results if not r.get('success', True)]
            
            batch_result = {
                'success': True,
                'total_documents': total_documents,
                'processed_successfully': len(successful_results),
                'failed': len(failed_results),
                'success_rate': (len(successful_results) / total_documents) * 100,
                'results': results,
                'metrics': self.metrics.get_metrics(),
                'processing_summary': self._generate_processing_summary(results)
            }
            
            self.logger.info(f"Batch processing completed: {len(successful_results)}/{total_documents} successful")
            return batch_result
            
        except Exception as e:
            self.logger.error(f"Error in batch processing: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'processed_successfully': len(self.processed_documents),
                'failed': len(self.failed_documents) + 1,
                'results': self.processed_documents + self.failed_documents
            }
    
    async def _process_single_document(self, doc_data: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Process a single document with full analysis"""
        try:
            filename = doc_data.get('filename', f'document_{index}')
            content = doc_data.get('content', '')
            file_size = doc_data.get('file_size', len(content.encode('utf-8')))
            
            # Validation stage
            self.metrics.start_stage('validation')
            is_valid, validation_message = validate_document_content(content)
            if not is_valid:
                raise DocumentValidationError(validation_message)
            
            # Extraction stage (metadata extraction)
            self.metrics.start_stage('extraction')
            metadata = extract_document_metadata(content, filename, file_size)
            
            # Analysis stage
            self.metrics.start_stage('analysis')
            document_type, type_confidence = detect_document_type(content, filename)
            risk_level, risk_score, risk_breakdown = assess_document_risk_level(content, document_type)
            topics = extract_key_topics(content, max_topics=10)
            entities = extract_key_entities(content)
            
            # Summarization stage
            self.metrics.start_stage('summarization')
            summary_data = generate_document_summary(content)
            
            # Calculate quality score
            quality_assessment = assess_content_quality(content)
            quality_score = quality_assessment['quality_score']
            
            return {
                'index': index,
                'filename': filename,
                'success': True,
                'document_type': document_type,
                'document_type_confidence': type_confidence,
                'risk_level': risk_level,
                'risk_score': risk_score,
                'risk_breakdown': risk_breakdown,
                'quality_score': quality_score,
                'quality_level': quality_assessment['quality_level'],
                'topics': [topic['topic'] for topic in topics[:5]],  # Top 5 topics
                'entities_summary': {
                    'monetary_amounts': len(entities.get('monetary_amounts', [])),
                    'dates': len(entities.get('dates', [])),
                    'organizations': len(entities.get('organizations', [])),
                    'legal_references': len(entities.get('legal_references', []))
                },
                'summary': summary_data['summary'],
                'metadata': {
                    'word_count': metadata.get('word_count', 0),
                    'reading_time': metadata.get('reading_time', {}),
                    'complexity_score': metadata.get('complexity_score', 0),
                    'completeness_score': metadata.get('completeness_score', 0)
                },
                'processing_time': (datetime.now(timezone.utc) - self.metrics.start_time).total_seconds() if self.metrics.start_time else 0
            }
            
        except Exception as e:
            raise DocumentProcessingError(f"Document processing failed: {str(e)}")
    
    def _generate_processing_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of batch processing results"""
        try:
            successful_results = [r for r in results if r.get('success', False)]
            
            if not successful_results:
                return {'message': 'No documents processed successfully'}
            
            # Document type distribution
            doc_types = {}
            risk_levels = {}
            quality_levels = {}
            
            total_words = 0
            total_processing_time = 0
            
            for result in successful_results:
                # Document types
                doc_type = result.get('document_type', 'unknown')
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                
                # Risk levels
                risk_level = result.get('risk_level', 'unknown')
                risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1
                
                # Quality levels
                quality_level = result.get('quality_level', 'unknown')
                quality_levels[quality_level] = quality_levels.get(quality_level, 0) + 1
                
                # Totals
                metadata = result.get('metadata', {})
                total_words += metadata.get('word_count', 0)
                total_processing_time += result.get('processing_time', 0)
            
            return {
                'document_types': doc_types,
                'risk_distribution': risk_levels,
                'quality_distribution': quality_levels,
                'total_words_processed': total_words,
                'average_words_per_document': total_words / len(successful_results),
                'total_processing_time': total_processing_time,
                'average_processing_time': total_processing_time / len(successful_results),
                'most_common_document_type': max(doc_types.items(), key=lambda x: x[1])[0] if doc_types else 'unknown',
                'highest_risk_documents': len([r for r in successful_results if r.get('risk_level') in ['high', 'critical']]),
                'recommendations': self._generate_batch_recommendations(successful_results)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating processing summary: {str(e)}")
            return {'error': str(e)}
    
    def _generate_batch_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on batch processing results"""
        recommendations = []
        
        try:
            high_risk_count = len([r for r in results if r.get('risk_level') in ['high', 'critical']])
            low_quality_count = len([r for r in results if r.get('quality_level') == 'low'])
            
            if high_risk_count > len(results) * 0.5:
                recommendations.append("More than half of your documents have high risk levels. Consider legal review.")
            
            if low_quality_count > len(results) * 0.3:
                recommendations.append("Several documents have quality issues. Check for OCR errors or formatting problems.")
            
            # Check for document type diversity
            doc_types = set(r.get('document_type', 'unknown') for r in results)
            if len(doc_types) > 5:
                recommendations.append("You have diverse document types. Consider organizing them by category.")
            
            # Processing time recommendations
            avg_time = sum(r.get('processing_time', 0) for r in results) / len(results)
            if avg_time > 60:
                recommendations.append("Documents are taking longer to process. Consider reducing file sizes.")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return ["Review individual document results for specific recommendations."]

# Global instances
processing_metrics = DocumentProcessingMetrics()

# Utility function for easy batch processing
async def process_document_batch(documents: List[Dict[str, Any]], 
                                max_concurrent: int = 3,
                                progress_callback: Optional[callable] = None) -> Dict[str, Any]:
    """
    Convenience function for batch document processing
    
    Args:
        documents: List of document dictionaries
        max_concurrent: Maximum concurrent processing
        progress_callback: Optional progress callback
        
    Returns:
        Batch processing results
    """
    processor = BatchDocumentProcessor(max_concurrent=max_concurrent)
    return await processor.process_documents(documents, progress_callback)

# Export important functions and classes
__all__ = [
    # Validation functions
    'validate_file_type',
    'validate_file_size', 
    'validate_document_content',
    'validate_session_request',
    
    # Analysis functions
    'detect_document_type',
    'assess_document_risk_level',
    'extract_key_entities',
    'extract_key_topics',
    'generate_document_summary',
    'extract_document_metadata',
    
    # Utility functions
    'generate_session_id',
    'generate_document_id',
    'generate_chunk_id',
    'calculate_file_hash',
    'calculate_content_hash',
    'format_file_size',
    'estimate_reading_time',
    'clean_text_content',
    
    # Comparison utilities
    'calculate_text_similarity',
    'find_common_topics',
    'assess_document_completeness',
    
    # Response utilities
    'create_error_response',
    'create_success_response',
    
    # Classes
    'DocumentProcessingMetrics',
    'BatchDocumentProcessor',
    
    # Batch processing
    'process_document_batch',
    
    # Exceptions
    'DocumentValidationError',
    'DocumentProcessingError',
    
    # Constants
    'SUPPORTED_MIME_TYPES',
    'LEGAL_DOCUMENT_KEYWORDS',
    'RISK_INDICATORS'
]

