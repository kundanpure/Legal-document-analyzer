"""
Application constants and enumerations
Enhanced with additional legal document types, risk indicators, and system configurations
"""

from typing import List, Dict, Set, Any, Union
from enum import Enum, IntEnum
from dataclasses import dataclass
import re

# Version and build information
APP_VERSION = "2.0.0"
API_VERSION = "v1"
BUILD_DATE = "2025-09-18"

# Document types (expanded)
LEGAL_DOCUMENT_TYPES = [
    'rental_agreement',
    'lease_agreement', 
    'loan_contract',
    'employment_contract',
    'service_agreement',
    'purchase_agreement',
    'terms_of_service',
    'privacy_policy',
    'partnership_agreement',
    'non_disclosure_agreement',
    'consulting_agreement',
    'licensing_agreement',
    'franchise_agreement',
    'distribution_agreement',
    'supply_agreement',
    'maintenance_agreement',
    'insurance_policy',
    'warranty_agreement',
    'sublease_agreement',
    'joint_venture_agreement',
    'shareholder_agreement',
    'merger_agreement',
    'acquisition_agreement',
    'intellectual_property_license',
    'software_license',
    'data_processing_agreement',
    'vendor_agreement',
    'contractor_agreement',
    'independent_contractor_agreement',
    'non_compete_agreement',
    'general_legal_document'
]

# Document categories for better organization
DOCUMENT_CATEGORIES = {
    'real_estate': [
        'rental_agreement', 'lease_agreement', 'sublease_agreement'
    ],
    'financial': [
        'loan_contract', 'insurance_policy', 'warranty_agreement'
    ],
    'employment': [
        'employment_contract', 'contractor_agreement', 'independent_contractor_agreement', 'non_compete_agreement'
    ],
    'business': [
        'service_agreement', 'consulting_agreement', 'partnership_agreement', 'joint_venture_agreement'
    ],
    'technology': [
        'software_license', 'data_processing_agreement', 'terms_of_service', 'privacy_policy'
    ],
    'intellectual_property': [
        'licensing_agreement', 'intellectual_property_license', 'non_disclosure_agreement'
    ],
    'corporate': [
        'shareholder_agreement', 'merger_agreement', 'acquisition_agreement'
    ],
    'commercial': [
        'purchase_agreement', 'supply_agreement', 'distribution_agreement', 'franchise_agreement'
    ]
}

# Risk categories (expanded)
RISK_CATEGORIES = [
    'financial_risk',
    'legal_liability',
    'termination_risk',
    'compliance_risk',
    'hidden_costs',
    'unfair_terms',
    'enforcement_risk',
    'renewal_risk',
    'intellectual_property_risk',
    'data_privacy_risk',
    'regulatory_risk',
    'operational_risk',
    'reputational_risk',
    'performance_risk',
    'counterparty_risk',
    'force_majeure_risk'
]

# Risk levels with detailed scoring
RISK_LEVELS = {
    'minimal': {'min': 0.0, 'max': 2.0, 'color': '#0f9d58', 'description': 'Very low risk, standard terms'},
    'low': {'min': 2.0, 'max': 4.0, 'color': '#34a853', 'description': 'Low risk, generally favorable'},
    'moderate': {'min': 4.0, 'max': 6.0, 'color': '#fbbc04', 'description': 'Moderate risk, requires attention'},
    'medium': {'min': 6.0, 'max': 7.5, 'color': '#ff9800', 'description': 'Medium risk, significant concerns'},
    'high': {'min': 7.5, 'max': 9.0, 'color': '#ff6d01', 'description': 'High risk, strong concerns'},
    'critical': {'min': 9.0, 'max': 10.0, 'color': '#ea4335', 'description': 'Critical risk, immediate attention required'}
}

# Supported languages (enhanced with additional metadata)
SUPPORTED_LANGUAGES = {
    'en': {
        'name': 'English',
        'native': 'English',
        'iso': 'en-US',
        'direction': 'ltr',
        'family': 'Germanic',
        'script': 'Latin',
        'tts_quality': 'excellent',
        'legal_support': 'complete'
    },
    'hi': {
        'name': 'Hindi',
        'native': 'हिन्दी',
        'iso': 'hi-IN',
        'direction': 'ltr',
        'family': 'Indo-Aryan',
        'script': 'Devanagari',
        'tts_quality': 'excellent',
        'legal_support': 'extensive'
    },
    'ta': {
        'name': 'Tamil',
        'native': 'தமிழ்',
        'iso': 'ta-IN',
        'direction': 'ltr',
        'family': 'Dravidian',
        'script': 'Tamil',
        'tts_quality': 'good',
        'legal_support': 'good'
    },
    'te': {
        'name': 'Telugu',
        'native': 'తెలుగు',
        'iso': 'te-IN',
        'direction': 'ltr',
        'family': 'Dravidian',
        'script': 'Telugu',
        'tts_quality': 'good',
        'legal_support': 'good'
    },
    'bn': {
        'name': 'Bengali',
        'native': 'বাংলা',
        'iso': 'bn-IN',
        'direction': 'ltr',
        'family': 'Indo-Aryan',
        'script': 'Bengali',
        'tts_quality': 'good',
        'legal_support': 'moderate'
    },
    'gu': {
        'name': 'Gujarati',
        'native': 'ગુજરાતી',
        'iso': 'gu-IN',
        'direction': 'ltr',
        'family': 'Indo-Aryan',
        'script': 'Gujarati',
        'tts_quality': 'good',
        'legal_support': 'moderate'
    },
    'kn': {
        'name': 'Kannada',
        'native': 'ಕನ್ನಡ',
        'iso': 'kn-IN',
        'direction': 'ltr',
        'family': 'Dravidian',
        'script': 'Kannada',
        'tts_quality': 'fair',
        'legal_support': 'moderate'
    },
    'ml': {
        'name': 'Malayalam',
        'native': 'മലയാളം',
        'iso': 'ml-IN',
        'direction': 'ltr',
        'family': 'Dravidian',
        'script': 'Malayalam',
        'tts_quality': 'fair',
        'legal_support': 'moderate'
    },
    'mr': {
        'name': 'Marathi',
        'native': 'मराठी',
        'iso': 'mr-IN',
        'direction': 'ltr',
        'family': 'Indo-Aryan',
        'script': 'Devanagari',
        'tts_quality': 'good',
        'legal_support': 'moderate'
    },
    'or': {
        'name': 'Odia',
        'native': 'ଓଡ଼ିଆ',
        'iso': 'or-IN',
        'direction': 'ltr',
        'family': 'Indo-Aryan',
        'script': 'Odia',
        'tts_quality': 'fair',
        'legal_support': 'basic'
    },
    'pa': {
        'name': 'Punjabi',
        'native': 'ਪੰਜਾਬੀ',
        'iso': 'pa-IN',
        'direction': 'ltr',
        'family': 'Indo-Aryan',
        'script': 'Gurmukhi',
        'tts_quality': 'good',
        'legal_support': 'moderate'
    },
    'ur': {
        'name': 'Urdu',
        'native': 'اردو',
        'iso': 'ur-PK',
        'direction': 'rtl',
        'family': 'Indo-Aryan',
        'script': 'Arabic',
        'tts_quality': 'good',
        'legal_support': 'moderate'
    },
    'as': {
        'name': 'Assamese',
        'native': 'অসমীয়া',
        'iso': 'as-IN',
        'direction': 'ltr',
        'family': 'Indo-Aryan',
        'script': 'Bengali',
        'tts_quality': 'fair',
        'legal_support': 'basic'
    }
}

# Language groups for batch operations
LANGUAGE_GROUPS = {
    'indian_languages': ['hi', 'ta', 'te', 'bn', 'gu', 'kn', 'ml', 'mr', 'or', 'pa', 'ur', 'as'],
    'primary_languages': ['en', 'hi', 'ta', 'te', 'bn'],
    'devanagari_script': ['hi', 'mr'],
    'dravidian_languages': ['ta', 'te', 'kn', 'ml'],
    'rtl_languages': ['ur']
}

# Expanded legal terms dictionary with categories
LEGAL_TERMS_GLOSSARY = {
    # Contract basics
    'liability': 'Legal responsibility for damages, debts, or other obligations',
    'indemnity': 'Security or protection against loss or damage; compensation for loss',
    'breach': 'Violation or failure to fulfill a legal duty or contractual obligation',
    'default': 'Failure to meet a legal obligation, especially failure to repay a debt',
    'termination': 'The ending or cancellation of a contract or legal relationship',
    'penalty': 'A punishment imposed for breaking a law, rule, or contract',
    'warranty': 'A promise or guarantee, especially about the condition or quality of something',
    'jurisdiction': 'The official power to make legal decisions and judgments',
    'arbitration': 'The settlement of a dispute by a neutral third party outside of court',
    'force_majeure': 'Unforeseeable circumstances that prevent fulfillment of a contract',
    'consideration': 'Something of value exchanged between parties in a contract',
    'covenant': 'A formal agreement or promise in a contract',
    
    # Advanced legal concepts
    'estoppel': 'A legal principle preventing contradiction of previous actions or words',
    'lien': 'A legal claim on property as security for debt or obligation',
    'novation': 'The substitution of a new contract for an existing one',
    'rescission': 'The cancellation or revocation of a contract',
    'specific_performance': 'A legal remedy requiring exact fulfillment of contract terms',
    'tort': 'A civil wrong causing harm to another person',
    'assignment': 'The transfer of rights or property to another party',
    'subletting': 'Renting out property that one is already renting from someone else',
    
    # Financial and commercial terms
    'liquidated_damages': 'Pre-determined compensation amount for contract breach',
    'earnest_money': 'Deposit made to demonstrate serious intent to purchase',
    'escrow': 'Third-party holding of funds or documents until conditions are met',
    'collateral': 'Assets pledged as security for repayment of a loan',
    'guarantor': 'Person who agrees to be responsible for another\'s debt or obligation',
    'mortgage': 'Legal agreement where property serves as collateral for a loan',
    'amortization': 'The process of gradually paying off a debt over time',
    'compound_interest': 'Interest calculated on both principal and accumulated interest',
    
    # Employment and business terms
    'non_compete': 'Agreement restricting work with competitors after employment ends',
    'confidentiality': 'Obligation to keep certain information secret',
    'intellectual_property': 'Legal rights over creations of the mind',
    'trade_secret': 'Confidential business information providing competitive advantage',
    'fiduciary_duty': 'Legal obligation to act in another party\'s best interest',
    'due_diligence': 'Investigation or audit of potential investment or business',
    'merger': 'Combination of two companies into one entity',
    'acquisition': 'Purchase of one company by another',
    
    # Property and real estate
    'lease': 'Contract granting use of property for specified time and payment',
    'landlord': 'Owner of property who rents it to tenants',
    'tenant': 'Person who rents and occupies property owned by another',
    'security_deposit': 'Money held to cover potential damages or unpaid rent',
    'easement': 'Right to use another person\'s property for specific purpose',
    'title': 'Legal ownership of property',
    'deed': 'Legal document transferring ownership of real estate',
    'zoning': 'Legal restrictions on how property can be used',
    
    # Dispute resolution
    'mediation': 'Process where neutral third party helps resolve disputes',
    'litigation': 'Process of taking legal action through courts',
    'settlement': 'Agreement resolving dispute without going to trial',
    'injunction': 'Court order requiring party to do or stop doing something',
    'damages': 'Monetary compensation for loss or injury',
    'restitution': 'Restoration of something to its original state or owner'
}

# Legal term categories for better organization
LEGAL_TERM_CATEGORIES = {
    'contract_basics': ['liability', 'breach', 'termination', 'warranty', 'consideration'],
    'financial_terms': ['liquidated_damages', 'earnest_money', 'escrow', 'collateral', 'mortgage'],
    'employment_law': ['non_compete', 'confidentiality', 'intellectual_property', 'fiduciary_duty'],
    'property_law': ['lease', 'landlord', 'tenant', 'security_deposit', 'title', 'deed'],
    'dispute_resolution': ['arbitration', 'mediation', 'litigation', 'settlement', 'injunction']
}

# Common contract clauses (expanded)
COMMON_CLAUSES = {
    'payment_terms': [
        'payment schedule',
        'due date',
        'late payment',
        'interest rate',
        'collection costs',
        'payment method',
        'currency',
        'invoice requirements',
        'disputed payments',
        'set-off rights'
    ],
    'termination_clauses': [
        'termination notice',
        'early termination',
        'breach termination',
        'mutual termination',
        'automatic renewal',
        'termination for convenience',
        'cure periods',
        'survival clauses',
        'return of property',
        'final payment'
    ],
    'liability_clauses': [
        'limitation of liability',
        'indemnification',
        'hold harmless',
        'insurance requirements',
        'damage caps',
        'consequential damages',
        'third party claims',
        'mutual indemnification',
        'defense obligations',
        'notification requirements'
    ],
    'dispute_resolution': [
        'governing law',
        'jurisdiction',
        'arbitration',
        'mediation',
        'attorney fees',
        'venue selection',
        'choice of law',
        'dispute escalation',
        'injunctive relief',
        'class action waiver'
    ],
    'confidentiality': [
        'non-disclosure',
        'confidential information',
        'trade secrets',
        'proprietary information',
        'return of materials',
        'permitted disclosures',
        'duration of confidentiality',
        'residual information',
        'third party information',
        'disclosure obligations'
    ],
    'intellectual_property': [
        'ownership of work product',
        'license grants',
        'moral rights',
        'patent indemnification',
        'trademark usage',
        'copyright assignment',
        'derivative works',
        'improvements',
        'prior inventions',
        'publicity rights'
    ],
    'force_majeure': [
        'unforeseeable events',
        'natural disasters',
        'government actions',
        'notification requirements',
        'mitigation efforts',
        'suspension of obligations',
        'termination rights',
        'partial performance',
        'force majeure cure',
        'cost allocation'
    ]
}

# Enhanced risk indicators with severity levels
RISK_INDICATORS = {
    'critical_risk': [
        'unlimited personal liability',
        'personal guarantees of corporate debt',
        'waiver of all legal rights',
        'irrevocable power of attorney',
        'blanket lien on all assets',
        'criminal liability exposure',
        'forfeiture of all payments',
        'permanent injunctive relief',
        'liquidated damages exceeding contract value',
        'absolute liability regardless of fault'
    ],
    'high_risk': [
        'unlimited liability',
        'personal guarantee',
        'broad indemnification',
        'no termination right',
        'automatic renewal without notice',
        'penalty fees exceeding damages',
        'liquidated damages',
        'waiver of jury trial',
        'unilateral modification rights',
        'exclusive dealing requirements',
        'non-compete for life',
        'assignment of all IP rights',
        'broad confidentiality obligations',
        'joint and several liability',
        'cross-default provisions'
    ],
    'medium_risk': [
        'limited liability caps',
        'reasonable indemnification',
        'termination for cause only',
        'renewal with notice',
        'late fees and interest',
        'specific damage measures',
        'binding arbitration',
        'modification with consent',
        'reasonable non-compete',
        'conditional assignment rights',
        'mutual indemnification',
        'force majeure exclusions',
        'minimum purchase requirements',
        'exclusivity in defined territory',
        'liquidated damages formula'
    ],
    'financial_risk': [
        'hidden fees and charges',
        'variable interest rates',
        'compound interest calculation',
        'prepayment penalties',
        'default interest rates',
        'collection costs and fees',
        'large security deposits',
        'earnest money at risk',
        'payment acceleration clauses',
        'attorney fees provisions',
        'cost of funds adjustments',
        'currency fluctuation risk',
        'minimum payment requirements',
        'balloon payment provisions',
        'cross-collateralization clauses'
    ],
    'operational_risk': [
        'unrealistic performance standards',
        'insufficient time for delivery',
        'broad scope of work',
        'unlimited change orders',
        'resource availability constraints',
        'technology dependency risks',
        'regulatory compliance burden',
        'third party dependency',
        'key person dependency',
        'seasonal performance risks'
    ],
    'legal_compliance_risk': [
        'regulatory change exposure',
        'license and permit requirements',
        'industry standard compliance',
        'data protection obligations',
        'environmental liability',
        'safety and health requirements',
        'employment law compliance',
        'tax obligation shifting',
        'import/export restrictions',
        'anti-corruption compliance'
    ]
}

# Industry standards (expanded with more document types)
INDUSTRY_STANDARDS = {
    'rental_agreement': {
        'average_risk_score': 5.2,
        'common_issues': [
            'excessive security deposits',
            'broad entry rights',
            'unclear maintenance responsibilities',
            'unfair termination clauses',
            'automatic renewal clauses',
            'tenant improvement restrictions'
        ],
        'red_flags': ['unlimited landlord entry', 'tenant pays all repairs', 'no notice termination']
    },
    'loan_contract': {
        'average_risk_score': 6.8,
        'common_issues': [
            'high interest rates',
            'prepayment penalties',
            'personal guarantees',
            'cross-default clauses',
            'acceleration clauses',
            'variable rate provisions'
        ],
        'red_flags': ['compound daily interest', 'unlimited personal liability', 'immediate acceleration']
    },
    'employment_contract': {
        'average_risk_score': 4.9,
        'common_issues': [
            'broad non-compete clauses',
            'unclear termination terms',
            'intellectual property assignments',
            'confidentiality overreach',
            'at-will employment',
            'commission calculation disputes'
        ],
        'red_flags': ['lifetime non-compete', 'forfeiture of all compensation', 'unlimited IP assignment']
    },
    'service_agreement': {
        'average_risk_score': 5.5,
        'common_issues': [
            'unlimited liability',
            'broad indemnification',
            'unclear scope of work',
            'unfavorable payment terms',
            'IP ownership disputes',
            'change order provisions'
        ],
        'red_flags': ['unlimited indemnification', 'all work for hire', 'payment only on completion']
    },
    'nda_agreement': {
        'average_risk_score': 4.1,
        'common_issues': [
            'overly broad confidentiality definition',
            'long confidentiality periods',
            'unclear exceptions',
            'one-sided obligations',
            'return of information requirements'
        ],
        'red_flags': ['perpetual confidentiality', 'no residual knowledge exception', 'criminal penalties']
    },
    'software_license': {
        'average_risk_score': 6.2,
        'common_issues': [
            'license scope limitations',
            'maintenance and support terms',
            'data ownership questions',
            'liability limitations',
            'termination and data retrieval'
        ],
        'red_flags': ['no data portability', 'unlimited license fees', 'immediate termination right']
    }
}

# Enhanced file constraints with security measures
FILE_CONSTRAINTS = {
    'max_file_size': 50 * 1024 * 1024,  # 50MB
    'allowed_extensions': ['.pdf', '.doc', '.docx', '.txt'],
    'allowed_mime_types': [
        'application/pdf',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'text/plain'
    ],
    'max_pages': 500,
    'max_text_length': 1000000,  # 1M characters
    'min_text_length': 100,      # Minimum meaningful content
    'virus_scan_required': True,
    'content_validation': True,
    'ocr_max_pages': 100,        # Max pages for OCR processing
    'processing_timeout': 300     # 5 minutes max processing time
}

# Enhanced API rate limits with burst allowances
RATE_LIMITS = {
    'document_upload': {
        'requests': 10, 
        'window': 3600, 
        'burst': 3,
        'penalty': 300  # 5 minute penalty for exceeding
    },
    'chat_message': {
        'requests': 100, 
        'window': 3600, 
        'burst': 10,
        'penalty': 60
    },
    'report_generation': {
        'requests': 5, 
        'window': 3600, 
        'burst': 2,
        'penalty': 600  # 10 minute penalty
    },
    'voice_generation': {
        'requests': 20, 
        'window': 3600, 
        'burst': 5,
        'penalty': 180  # 3 minute penalty
    },
    'translation': {
        'requests': 200,
        'window': 3600,
        'burst': 20,
        'penalty': 120
    },
    'general_api': {
        'requests': 1000, 
        'window': 3600, 
        'burst': 50,
        'penalty': 60
    }
}

# Enhanced processing timeouts with retry logic
PROCESSING_TIMEOUTS = {
    'document_analysis': {'timeout': 300, 'retries': 2, 'backoff': 'exponential'},
    'report_generation': {'timeout': 180, 'retries': 1, 'backoff': 'linear'},
    'voice_generation': {'timeout': 120, 'retries': 2, 'backoff': 'exponential'},
    'chat_response': {'timeout': 30, 'retries': 3, 'backoff': 'immediate'},
    'translation': {'timeout': 15, 'retries': 2, 'backoff': 'linear'},
    'ocr_processing': {'timeout': 240, 'retries': 1, 'backoff': 'none'},
    'ai_analysis': {'timeout': 180, 'retries': 2, 'backoff': 'exponential'}
}

# Enhanced cache settings with invalidation policies
CACHE_SETTINGS = {
    'document_analysis': {
        'ttl': 3600, 
        'prefix': 'doc_', 
        'max_size': '100MB',
        'invalidate_on': ['document_update']
    },
    'translations': {
        'ttl': 7200, 
        'prefix': 'trans_', 
        'max_size': '50MB',
        'invalidate_on': ['language_update']
    },
    'voice_summaries': {
        'ttl': 86400, 
        'prefix': 'voice_', 
        'max_size': '200MB',
        'invalidate_on': ['content_update']
    },
    'reports': {
        'ttl': 86400, 
        'prefix': 'report_', 
        'max_size': '100MB',
        'invalidate_on': ['analysis_update']
    },
    'suggestions': {
        'ttl': 1800, 
        'prefix': 'suggest_', 
        'max_size': '10MB',
        'invalidate_on': ['model_update']
    },
    'user_sessions': {
        'ttl': 3600,
        'prefix': 'session_',
        'max_size': '20MB',
        'invalidate_on': ['logout', 'timeout']
    }
}

# Enhanced response templates with more languages and contexts
RESPONSE_TEMPLATES = {
    'welcome_message': {
        'en': "Hello! I've analyzed your {document_type} and identified {risk_count} key areas that need attention. What would you like to know?",
        'hi': "नमस्ते! मैंने आपके {document_type} का विश्लेषण किया है और {risk_count} मुख्य क्षेत्रों की पहचान की है जिन पर ध्यान देने की आवश्यकता है। आप क्या जानना चाहते हैं?",
        'ta': "வணக்கம்! நான் உங்கள் {document_type} ஐ ஆய்வு செய்து {risk_count} முக்கிய பகுதிகளை கண்டறிந்துள்ளேன். நீங்கள் என்ன தெரிந்துகொள்ள விரும்புகிறீர்கள்?",
        'te': "నమస్కారం! నేను మీ {document_type} ను విశ్లేషించి {risk_count} ముఖ్య రంగాలను గుర్తించాను। మీరు ఏమి తెలుసుకోవాలనుకుంటున్నారు?",
        'bn': "নমস্কার! আমি আপনার {document_type} বিশ্লেষণ করেছি এবং {risk_count}টি মূল ক্ষেত্র চিহ্নিত করেছি যেগুলির প্রতি মনোযোগ প্রয়োজন। আপনি কী জানতে চান?"
    },
    'processing_message': {
        'en': "I'm analyzing your document. This usually takes 30-45 seconds...",
        'hi': "मैं आपके दस्तावेज़ का विश्लेषण कर रहा हूं। इसमें आमतौर पर 30-45 सेकंड लगते हैं...",
        'ta': "நான் உங்கள் ஆவணத்தை ஆய்வு செய்து வருகிறேன். இதற்கு பொதுவாக 30-45 வினாடிகள் ஆகும்...",
        'te': "నేను మీ పత్రాన్ని విశ్లేషిస్తున్నాను। దీనికి సాధారణంగా 30-45 సెకన్లు పడుతుంది...",
        'bn': "আমি আপনার নথি বিশ্লেষণ করছি। এতে সাধারণত ৩০-৪৫ সেকেন্ড লাগে..."
    },
    'error_message': {
        'en': "I apologize, but I encountered an error. Please try again or contact support.",
        'hi': "मुझे खेद है, लेकिन मुझे एक त्रुटि का सामना करना पড़ा। कृपया पुनः प्रयास करें या सहायता से संपर्क करें।",
        'ta': "மன்னிக்கவும், எனக்கு ஒரு பிழை ஏற்பட்டது। தயவுசெய்து மீண்டும் முயற்சிக்கவும் அல்லது ஆதரவைத் தொடர்பு கொள்ளவும்.",
        'te': "క్షమించండి, నాకు ఒక దోషం ఎదురైంది। దయచేసి మళ్లీ ప్రయత్నించండి లేదా మద్దతును సంప్రదించండి।",
        'bn': "দুঃখিত, আমি একটি ত্রুটির সম্মুখীন হয়েছি। অনুগ্রহ করে আবার চেষ্টা করুন বা সহায়তার সাথে যোগাযোগ করুন।"
    },
    'completion_message': {
        'en': "Analysis complete! I found {risk_count} areas of concern. Here's what you should know:",
        'hi': "विश्लेषण पूरा हो गया! मुझे {risk_count} चिंता के क्षेत्र मिले। आपको यह जानना चाहिए:",
        'ta': "ஆய்வு முடிவடைந்தது! நான் {risk_count} கவலைக்குரிய பகுதிகளைக் கண்டேன். நீங்கள் தெரிந்து கொள்ள வேண்டியது:",
        'te': "విశ్లేషణ పూర్తయింది! నేను {risk_count} ఆందోళనకర ప్రాంతాలను కనుగొన్నాను। మీరు తెలుసుకోవాల్సినవి:",
        'bn': "বিশ্লেষণ সম্পন্ন! আমি {risk_count}টি উদ্বেগের ক্ষেত্র খুঁজে পেয়েছি। আপনার যা জানা উচিত:"
    }
}

# Enhanced system defaults with environment-specific settings
SYSTEM_DEFAULTS = {
    'language': 'en',
    'risk_threshold': 5.0,
    'confidence_threshold': 0.7,
    'max_suggestions': 5,
    'session_timeout': 3600,        # 1 hour
    'report_expiry': 2592000,       # 30 days
    'voice_expiry': 604800,         # 7 days
    'cleanup_interval': 86400,      # 24 hours
    'max_concurrent_analyses': 10,
    'ai_model_temperature': 0.3,
    'max_retry_attempts': 3,
    'batch_size': 50,
    'log_retention_days': 30,
    'backup_frequency_hours': 24
}

# Enhanced feature flags with versioning
FEATURE_FLAGS = {
    'multilingual_support': True,
    'voice_generation': True,
    'industry_comparison': True,
    'advanced_analytics': True,
    'real_time_analysis': True,
    'batch_processing': False,      # Future feature
    'api_versioning': True,
    'user_authentication': False,   # Future feature
    'premium_features': False,      # Future feature
    'ai_explanations': True,
    'document_comparison': True,
    'risk_trending': False,         # Future feature
    'collaboration_tools': False,   # Future feature
    'mobile_optimization': True,
    'offline_mode': False,          # Future feature
    'custom_risk_profiles': False,  # Future feature
    'audit_logging': True,
    'data_export': True,
    'webhook_notifications': False  # Future feature
}

# Performance monitoring thresholds
PERFORMANCE_THRESHOLDS = {
    'response_time_warning': 5.0,    # seconds
    'response_time_critical': 15.0,  # seconds
    'memory_usage_warning': 80,      # percentage
    'memory_usage_critical': 95,     # percentage
    'error_rate_warning': 5,         # percentage
    'error_rate_critical': 15,       # percentage
    'queue_size_warning': 100,
    'queue_size_critical': 500
}

# Security configuration
SECURITY_CONFIG = {
    'max_login_attempts': 5,
    'lockout_duration': 900,         # 15 minutes
    'password_min_length': 8,
    'session_rotation': 3600,        # 1 hour
    'csrf_protection': True,
    'xss_protection': True,
    'content_security_policy': True,
    'rate_limit_enforcement': True,
    'audit_logging': True,
    'encryption_at_rest': True,
    'encryption_in_transit': True
}

# Enums for better type safety
class RiskLevel(IntEnum):
    MINIMAL = 0
    LOW = 1
    MODERATE = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5

class DocumentStatus(Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    ANALYZED = "analyzed"
    FAILED = "failed"
    EXPIRED = "expired"

class ProcessingStage(Enum):
    RECEIVED = "received"
    VALIDATING = "validating"
    EXTRACTING = "extracting"
    ANALYZING = "analyzing"
    GENERATING_REPORT = "generating_report"
    COMPLETED = "completed"
    FAILED = "failed"

# Data classes for structured configuration
@dataclass
class LanguageConfig:
    code: str
    name: str
    native_name: str
    iso_code: str
    direction: str
    tts_support: bool = True
    translation_support: bool = True

@dataclass
class RiskConfig:
    category: str
    weight: float
    threshold: float
    indicators: List[str]

# Utility functions for constants
def get_risk_level_by_score(score: float) -> str:
    """Get risk level name by numeric score"""
    for level, config in RISK_LEVELS.items():
        if config['min'] <= score < config['max']:
            return level
    return 'unknown'

def get_supported_language_codes() -> List[str]:
    """Get list of supported language codes"""
    return list(SUPPORTED_LANGUAGES.keys())

def get_document_types_by_category(category: str) -> List[str]:
    """Get document types for a specific category"""
    return DOCUMENT_CATEGORIES.get(category, [])

def is_high_risk_indicator(text: str) -> bool:
    """Check if text contains high-risk indicators"""
    text_lower = text.lower()
    return any(indicator in text_lower for indicator in RISK_INDICATORS['high_risk'])

def get_legal_term_definition(term: str) -> str:
    """Get definition for a legal term"""
    return LEGAL_TERMS_GLOSSARY.get(term.lower(), f"Definition not available for '{term}'")

def validate_file_constraints(file_size: int, file_extension: str, mime_type: str) -> Dict[str, Any]:
    """Validate file against constraints"""
    issues = []
    
    if file_size > FILE_CONSTRAINTS['max_file_size']:
        issues.append(f"File size {file_size} exceeds maximum {FILE_CONSTRAINTS['max_file_size']}")
    
    if file_extension.lower() not in FILE_CONSTRAINTS['allowed_extensions']:
        issues.append(f"File extension {file_extension} not allowed")
    
    if mime_type not in FILE_CONSTRAINTS['allowed_mime_types']:
        issues.append(f"MIME type {mime_type} not allowed")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues
    }

# Regular expressions for text processing
REGEX_PATTERNS = {
    'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    'phone': re.compile(r'(\+\d{1,3}\s?)?(\(\d{3}\)|\d{3})[\s\-]?\d{3}[\s\-]?\d{4}'),
    'currency': re.compile(r'[\$₹£€¥]\s?\d{1,3}(,\d{3})*(\.\d{2})?'),
    'percentage': re.compile(r'\d+(\.\d+)?%'),
    'date': re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
    'legal_reference': re.compile(r'Section\s+\d+|Article\s+\d+|Clause\s+\d+', re.IGNORECASE)
}

# Export commonly used constants for easy importing
__all__ = [
    'LEGAL_DOCUMENT_TYPES',
    'RISK_CATEGORIES', 
    'RISK_LEVELS',
    'SUPPORTED_LANGUAGES',
    'LEGAL_TERMS_GLOSSARY',
    'COMMON_CLAUSES',
    'RISK_INDICATORS',
    'INDUSTRY_STANDARDS',
    'FILE_CONSTRAINTS',
    'RATE_LIMITS',
    'PROCESSING_TIMEOUTS',
    'CACHE_SETTINGS',
    'RESPONSE_TEMPLATES',
    'SYSTEM_DEFAULTS',
    'FEATURE_FLAGS',
    'RiskLevel',
    'DocumentStatus',
    'ProcessingStage'
]
