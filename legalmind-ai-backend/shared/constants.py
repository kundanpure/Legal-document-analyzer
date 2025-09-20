"""
Shared constants for LegalMind AI Backend
"""

# Supported file types
SUPPORTED_FILE_TYPES = {
    '.pdf': 'application/pdf',
    '.doc': 'application/msword',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.txt': 'text/plain',
    '.rtf': 'application/rtf',
    '.odt': 'application/vnd.oasis.opendocument.text'
}

# Document processing limits
MAX_FILE_SIZE_MB = 50
MAX_PAGES_PER_CHUNK = 30
MAX_PROCESSING_TIME_MINUTES = 10

# Document types
DOCUMENT_TYPES = [
    'rental_agreement',
    'loan_contract', 
    'employment_contract',
    'terms_of_service',
    'privacy_policy',
    'purchase_agreement',
    'partnership_agreement',
    'nda',
    'other'
]

# Risk levels
RISK_LEVELS = ['high', 'medium', 'low']

# GCS Configuration
GCS_BUCKET_DOCUMENTS = "legalmind-documents"
GCS_BUCKET_RESULTS = "legalmind-results"
GCS_BUCKET_TEMP = "legalmind-temp"
