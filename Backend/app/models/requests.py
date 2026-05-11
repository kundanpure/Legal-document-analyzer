"""
Enhanced Pydantic request models for LegalMind AI API endpoints
Production-ready with comprehensive validation, documentation, and type safety
"""

import re
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, validator, root_validator, EmailStr
from pydantic.types import UUID4, constr, confloat, conint


# Enhanced Enums

class LanguageCode(str, Enum):
    """Supported language codes"""
    ENGLISH = "en"
    HINDI = "hi"
    TAMIL = "ta"
    TELUGU = "te"
    BENGALI = "bn"
    GUJARATI = "gu"
    KANNADA = "kn"
    MALAYALAM = "ml"
    MARATHI = "mr"
    ODIA = "or"
    PUNJABI = "pa"
    URDU = "ur"
    ASSAMESE = "as"


class DocumentType(str, Enum):
    """Document type classifications"""
    RENTAL_AGREEMENT = "rental_agreement"
    LEASE_DEED = "lease_deed"
    LOAN_CONTRACT = "loan_contract"
    MORTGAGE_AGREEMENT = "mortgage_agreement"
    EMPLOYMENT_CONTRACT = "employment_contract"
    SERVICE_AGREEMENT = "service_agreement"
    NDA = "nda"
    CONFIDENTIALITY_AGREEMENT = "confidentiality_agreement"
    TERMS_OF_SERVICE = "terms_of_service"
    PRIVACY_POLICY = "privacy_policy"
    PURCHASE_AGREEMENT = "purchase_agreement"
    SALE_DEED = "sale_deed"
    PARTNERSHIP_AGREEMENT = "partnership_agreement"
    FRANCHISE_AGREEMENT = "franchise_agreement"
    LICENSE_AGREEMENT = "license_agreement"
    INSURANCE_POLICY = "insurance_policy"
    WARRANTY_AGREEMENT = "warranty_agreement"
    GENERAL = "general"
    OTHER = "other"


class ReportTemplate(str, Enum):
    """Report template types"""
    EXECUTIVE = "executive"
    DETAILED = "detailed"
    COMPLIANCE = "compliance"
    COMPARISON = "comparison"
    RISK_ASSESSMENT = "risk_assessment"
    LEGAL_SUMMARY = "legal_summary"
    CUSTOM = "custom"


class ExportFormat(str, Enum):
    """Export format options"""
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"


class VoiceType(str, Enum):
    """Voice synthesis options"""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class VoiceContentType(str, Enum):
    """Voice content types"""
    SUMMARY = "summary"
    EXECUTIVE_SUMMARY = "executive"
    RISKS = "risks"
    RECOMMENDATIONS = "recommendations"
    FULL = "full"


class ProcessingPriority(str, Enum):
    """Processing priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class AnalysisDepth(str, Enum):
    """Analysis depth levels"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    EXPERT = "expert"


# Base Models

class BaseRequest(BaseModel):
    """Base request model with common fields"""
    request_id: Optional[str] = Field(None, description="Optional request tracking ID")
    client_info: Optional[Dict[str, Any]] = Field(None, description="Client metadata")
    
    class Config:
        extra = "forbid"  # Prevent extra fields
        validate_assignment = True


# Authentication and User Models

class LoginRequest(BaseModel):
    """User login request"""
    email: EmailStr = Field(..., description="User email address")
    password: constr(min_length=8, max_length=128) = Field(..., description="User password")
    remember_me: bool = Field(False, description="Remember login session")
    
    @validator('password')
    def validate_password(cls, v):
        if not re.search(r"[A-Za-z]", v) or not re.search(r"\d", v):
            raise ValueError('Password must contain both letters and numbers')
        return v


class RegisterRequest(BaseModel):
    """User registration request"""
    email: EmailStr = Field(..., description="User email address")
    password: constr(min_length=8, max_length=128) = Field(..., description="User password")
    confirm_password: str = Field(..., description="Password confirmation")
    full_name: constr(min_length=2, max_length=100) = Field(..., description="Full name")
    organization: Optional[constr(max_length=100)] = Field(None, description="Organization name")
    terms_accepted: bool = Field(..., description="Terms and conditions acceptance")
    
    @root_validator
    def validate_passwords_match(cls, values):
        if values.get('password') != values.get('confirm_password'):
            raise ValueError('Passwords do not match')
        return values
    
    @validator('terms_accepted')
    def validate_terms(cls, v):
        if not v:
            raise ValueError('Terms and conditions must be accepted')
        return v


class RefreshTokenRequest(BaseModel):
    """Token refresh request"""
    refresh_token: str = Field(..., description="Valid refresh token")


# Document Upload and Processing

class DocumentUploadRequest(BaseRequest):
    """Enhanced document upload request"""
    query: constr(min_length=5, max_length=1000) = Field(
        default="Analyze this legal document for risks and key terms",
        description="Specific analysis question or instruction"
    )
    language: LanguageCode = Field(
        default=LanguageCode.ENGLISH,
        description="Preferred language for analysis"
    )
    priority: ProcessingPriority = Field(
        default=ProcessingPriority.NORMAL,
        description="Processing priority level"
    )
    analysis_depth: AnalysisDepth = Field(
        default=AnalysisDepth.STANDARD,
        description="Depth of analysis to perform"
    )
    document_type_hint: Optional[DocumentType] = Field(
        None, 
        description="Optional hint about document type"
    )
    tags: Optional[List[constr(max_length=50)]] = Field(
        None,
        description="Optional tags for categorization"
    )
    max_length: Optional[conint(ge=100, le=10000)] = Field(
        None,
        description="Maximum text length to analyze"
    )
    include_ocr: bool = Field(
        True,
        description="Enable OCR for scanned documents"
    )
    extract_entities: bool = Field(
        True,
        description="Extract legal entities and key terms"
    )
    
    @validator('tags')
    def validate_tags(cls, v):
        if v and len(v) > 10:
            raise ValueError('Maximum 10 tags allowed')
        return v


class DocumentAnalysisRequest(BaseRequest):
    """Document re-analysis request"""
    document_id: constr(regex=r'^[a-fA-F0-9-]+$') = Field(..., description="Valid document ID")
    query: constr(min_length=5, max_length=1000) = Field(
        ..., 
        description="New analysis question"
    )
    language: LanguageCode = Field(
        default=LanguageCode.ENGLISH,
        description="Analysis language"
    )
    focus_areas: List[str] = Field(
        default=["risks", "obligations", "rights"],
        description="Specific areas to focus analysis on"
    )
    analysis_depth: AnalysisDepth = Field(
        default=AnalysisDepth.STANDARD,
        description="Analysis depth level"
    )
    
    @validator('focus_areas')
    def validate_focus_areas(cls, v):
        valid_areas = ["risks", "obligations", "rights", "financial", "termination", "compliance", "liability", "intellectual_property"]
        invalid_areas = [area for area in v if area not in valid_areas]
        if invalid_areas:
            raise ValueError(f'Invalid focus areas: {invalid_areas}. Valid options: {valid_areas}')
        return v


# Chat and Conversation

class ChatRequest(BaseRequest):
    """Enhanced chat message request"""
    session_id: constr(regex=r'^[a-fA-F0-9-]+$') = Field(..., description="Valid session ID")
    document_id: constr(regex=r'^[a-fA-F0-9-]+$') = Field(..., description="Valid document ID")
    message: constr(min_length=1, max_length=2000) = Field(..., description="User message")
    language: LanguageCode = Field(default=LanguageCode.ENGLISH, description="Response language")
    context_length: Optional[conint(ge=1, le=10)] = Field(
        5, 
        description="Number of previous messages to include as context"
    )
    temperature: Optional[confloat(ge=0.0, le=1.0)] = Field(
        None,
        description="AI response creativity level"
    )
    max_response_length: Optional[conint(ge=100, le=2000)] = Field(
        None,
        description="Maximum response length"
    )
    include_citations: bool = Field(True, description="Include source citations")
    include_suggestions: bool = Field(True, description="Include follow-up suggestions")


class ChatSessionRequest(BaseRequest):
    """Chat session creation request"""
    document_id: constr(regex=r'^[a-fA-F0-9-]+$') = Field(..., description="Valid document ID")
    language: LanguageCode = Field(default=LanguageCode.ENGLISH, description="Session language")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    welcome_message: Optional[constr(max_length=500)] = Field(
        None,
        description="Custom welcome message"
    )


# Report Generation

class ReportRequest(BaseRequest):
    """Enhanced report generation request"""
    document_id: constr(regex=r'^[a-fA-F0-9-]+$') = Field(..., description="Valid document ID")
    template: ReportTemplate = Field(
        default=ReportTemplate.DETAILED,
        description="Report template to use"
    )
    language: LanguageCode = Field(
        default=LanguageCode.ENGLISH,
        description="Report language"
    )
    export_format: ExportFormat = Field(
        default=ExportFormat.PDF,
        description="Export format"
    )
    include_sections: List[str] = Field(
        default=["summary", "risks", "recommendations", "obligations"],
        description="Sections to include in report"
    )
    custom_branding: Optional[Dict[str, Any]] = Field(
        None,
        description="Custom branding options"
    )
    include_charts: bool = Field(True, description="Include charts and visualizations")
    include_appendix: bool = Field(True, description="Include detailed appendix")
    watermark: Optional[str] = Field(None, description="Custom watermark text")
    
    @validator('include_sections')
    def validate_sections(cls, v):
        valid_sections = [
            "summary", "executive_summary", "risks", "recommendations", 
            "obligations", "rights", "financial", "termination", "compliance",
            "key_terms", "parties", "timeline", "appendix"
        ]
        invalid_sections = [section for section in v if section not in valid_sections]
        if invalid_sections:
            raise ValueError(f'Invalid sections: {invalid_sections}. Valid options: {valid_sections}')
        return v


class CustomReportRequest(BaseRequest):
    """Custom report generation request"""
    document_id: constr(regex=r'^[a-fA-F0-9-]+$') = Field(..., description="Valid document ID")
    template_config: Dict[str, Any] = Field(..., description="Custom template configuration")
    language: LanguageCode = Field(default=LanguageCode.ENGLISH, description="Report language")
    export_format: ExportFormat = Field(default=ExportFormat.PDF, description="Export format")
    sections: List[str] = Field(..., description="Custom sections to include")
    branding: Optional[Dict[str, Any]] = Field(None, description="Branding configuration")
    
    @validator('template_config')
    def validate_template_config(cls, v):
        required_fields = ["name", "layout", "styles"]
        missing_fields = [field for field in required_fields if field not in v]
        if missing_fields:
            raise ValueError(f'Missing required template config fields: {missing_fields}')
        return v


class BulkReportRequest(BaseRequest):
    """Bulk report generation request"""
    document_ids: List[constr(regex=r'^[a-fA-F0-9-]+$')] = Field(
        ..., 
        description="List of document IDs"
    )
    template: ReportTemplate = Field(
        default=ReportTemplate.DETAILED,
        description="Report template for all documents"
    )
    language: LanguageCode = Field(
        default=LanguageCode.ENGLISH,
        description="Report language"
    )
    export_format: ExportFormat = Field(
        default=ExportFormat.PDF,
        description="Export format"
    )
    include_sections: Optional[List[str]] = Field(
        None,
        description="Sections to include (uses template default if not specified)"
    )
    merge_reports: bool = Field(
        False,
        description="Merge all reports into single file"
    )
    
    @validator('document_ids')
    def validate_document_ids(cls, v):
        if len(v) > 50:
            raise ValueError('Maximum 50 documents allowed per bulk request')
        if len(set(v)) != len(v):
            raise ValueError('Duplicate document IDs not allowed')
        return v


# Voice Generation

class VoiceRequest(BaseRequest):
    """Enhanced voice generation request"""
    document_id: constr(regex=r'^[a-fA-F0-9-]+$') = Field(..., description="Valid document ID")
    language: LanguageCode = Field(default=LanguageCode.ENGLISH, description="Voice language")
    voice_type: VoiceType = Field(default=VoiceType.FEMALE, description="Voice type")
    speed: confloat(ge=0.5, le=2.0) = Field(default=1.0, description="Speech speed multiplier")
    content_type: VoiceContentType = Field(
        default=VoiceContentType.SUMMARY,
        description="Type of content to vocalize"
    )
    max_length: Optional[conint(ge=100, le=5000)] = Field(
        None,
        description="Maximum text length for voice generation"
    )
    include_ssml: bool = Field(
        False,
        description="Include SSML tags for enhanced pronunciation"
    )
    audio_format: str = Field(
        default="mp3",
        description="Audio output format"
    )
    
    @validator('audio_format')
    def validate_audio_format(cls, v):
        valid_formats = ["mp3", "wav", "ogg"]
        if v not in valid_formats:
            raise ValueError(f'Invalid audio format. Valid options: {valid_formats}')
        return v


class MultilingualVoiceRequest(BaseRequest):
    """Multilingual voice generation request"""
    document_id: constr(regex=r'^[a-fA-F0-9-]+$') = Field(..., description="Valid document ID")
    languages: List[LanguageCode] = Field(..., description="List of languages for voice generation")
    voice_type: VoiceType = Field(default=VoiceType.FEMALE, description="Voice type")
    content_type: VoiceContentType = Field(
        default=VoiceContentType.SUMMARY,
        description="Content type"
    )
    speed: confloat(ge=0.5, le=2.0) = Field(default=1.0, description="Speech speed")
    
    @validator('languages')
    def validate_languages(cls, v):
        if len(v) > 10:
            raise ValueError('Maximum 10 languages allowed per request')
        if len(set(v)) != len(v):
            raise ValueError('Duplicate languages not allowed')
        return v


# Translation

class TranslationRequest(BaseRequest):
    """Enhanced translation request"""
    document_id: constr(regex=r'^[a-fA-F0-9-]+$') = Field(..., description="Valid document ID")
    target_language: LanguageCode = Field(..., description="Target language")
    source_language: Optional[LanguageCode] = Field(None, description="Source language (auto-detect if not specified)")
    content_type: str = Field(
        default="summary",
        description="Type of content to translate"
    )
    preserve_formatting: bool = Field(True, description="Preserve original formatting")
    include_glossary: bool = Field(True, description="Include legal term glossary")
    
    @validator('content_type')
    def validate_content_type(cls, v):
        valid_types = ["summary", "full", "risks", "recommendations", "obligations", "rights"]
        if v not in valid_types:
            raise ValueError(f'Invalid content type. Valid options: {valid_types}')
        return v


class BulkTranslationRequest(BaseRequest):
    """Bulk translation request"""
    document_id: constr(regex=r'^[a-fA-F0-9-]+$') = Field(..., description="Valid document ID")
    target_languages: List[LanguageCode] = Field(..., description="List of target languages")
    content_type: str = Field(default="summary", description="Content type to translate")
    
    @validator('target_languages')
    def validate_target_languages(cls, v):
        if len(v) > 5:
            raise ValueError('Maximum 5 target languages allowed per request')
        return v


# Comparison and Analysis

class ComparisonRequest(BaseRequest):
    """Document comparison request"""
    document_id: constr(regex=r'^[a-fA-F0-9-]+$') = Field(..., description="Primary document ID")
    comparison_type: str = Field(
        default="industry_standard",
        description="Type of comparison to perform"
    )
    document_type: DocumentType = Field(..., description="Document type for comparison")
    industry: Optional[str] = Field(None, description="Industry context")
    jurisdiction: Optional[str] = Field(None, description="Legal jurisdiction")
    comparison_criteria: List[str] = Field(
        default=["fairness", "risk_level", "standard_terms"],
        description="Criteria for comparison"
    )
    
    @validator('comparison_type')
    def validate_comparison_type(cls, v):
        valid_types = ["industry_standard", "best_practices", "regulatory_compliance", "peer_comparison"]
        if v not in valid_types:
            raise ValueError(f'Invalid comparison type. Valid options: {valid_types}')
        return v


class RiskAnalysisRequest(BaseRequest):
    """Enhanced risk analysis request"""
    document_id: constr(regex=r'^[a-fA-F0-9-]+$') = Field(..., description="Valid document ID")
    analysis_depth: AnalysisDepth = Field(
        default=AnalysisDepth.COMPREHENSIVE,
        description="Depth of risk analysis"
    )
    risk_categories: List[str] = Field(
        default=["financial", "legal", "operational", "compliance"],
        description="Risk categories to analyze"
    )
    jurisdiction: Optional[str] = Field(None, description="Legal jurisdiction context")
    industry_context: Optional[str] = Field(None, description="Industry-specific risk factors")
    risk_tolerance: str = Field(
        default="moderate",
        description="Risk tolerance level"
    )
    
    @validator('risk_tolerance')
    def validate_risk_tolerance(cls, v):
        valid_levels = ["low", "moderate", "high", "aggressive"]
        if v not in valid_levels:
            raise ValueError(f'Invalid risk tolerance. Valid options: {valid_levels}')
        return v


# System and Admin Requests

class SystemMaintenanceRequest(BaseRequest):
    """System maintenance request"""
    maintenance_type: str = Field(..., description="Type of maintenance")
    duration_minutes: conint(ge=5, le=240) = Field(..., description="Estimated duration in minutes")
    message: Optional[constr(max_length=200)] = Field(None, description="Maintenance message")
    
    @validator('maintenance_type')
    def validate_maintenance_type(cls, v):
        valid_types = ["planned", "emergency", "security_update", "feature_deployment"]
        if v not in valid_types:
            raise ValueError(f'Invalid maintenance type. Valid options: {valid_types}')
        return v


class BulkProcessingRequest(BaseRequest):
    """Bulk processing request"""
    operation: str = Field(..., description="Bulk operation type")
    parameters: Dict[str, Any] = Field(..., description="Operation parameters")
    batch_size: conint(ge=1, le=100) = Field(default=10, description="Batch processing size")
    priority: ProcessingPriority = Field(default=ProcessingPriority.NORMAL, description="Processing priority")
    
    @validator('operation')
    def validate_operation(cls, v):
        valid_operations = ["document_analysis", "report_generation", "voice_synthesis", "translation", "cleanup"]
        if v not in valid_operations:
            raise ValueError(f'Invalid operation. Valid options: {valid_operations}')
        return v


# Feedback and Support

class FeedbackRequest(BaseModel):
    """User feedback request"""
    category: str = Field(..., description="Feedback category")
    rating: conint(ge=1, le=5) = Field(..., description="Rating (1-5)")
    message: constr(min_length=10, max_length=2000) = Field(..., description="Feedback message")
    feature_related: Optional[str] = Field(None, description="Related feature")
    contact_email: Optional[EmailStr] = Field(None, description="Contact email for follow-up")
    
    @validator('category')
    def validate_category(cls, v):
        valid_categories = ["bug_report", "feature_request", "general_feedback", "performance", "usability"]
        if v not in valid_categories:
            raise ValueError(f'Invalid category. Valid options: {valid_categories}')
        return v


class SupportTicketRequest(BaseModel):
    """Support ticket request"""
    subject: constr(min_length=5, max_length=100) = Field(..., description="Ticket subject")
    description: constr(min_length=20, max_length=2000) = Field(..., description="Detailed description")
    priority: str = Field(default="normal", description="Ticket priority")
    category: str = Field(..., description="Issue category")
    user_email: EmailStr = Field(..., description="User email")
    attachments: Optional[List[str]] = Field(None, description="Attachment URLs")
    
    @validator('priority')
    def validate_priority(cls, v):
        valid_priorities = ["low", "normal", "high", "urgent"]
        if v not in valid_priorities:
            raise ValueError(f'Invalid priority. Valid options: {valid_priorities}')
        return v


# Validation Helpers

def validate_uuid_format(value: str) -> str:
    """Validate UUID format"""
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
    if not re.match(uuid_pattern, value, re.IGNORECASE):
        raise ValueError('Invalid UUID format')
    return value.lower()


def validate_document_id(value: str) -> str:
    """Validate document ID format"""
    return validate_uuid_format(value)


def validate_session_id(value: str) -> str:
    """Validate session ID format"""
    return validate_uuid_format(value)


# Add validators to relevant models using root_validator for cross-field validation
for model_class in [DocumentUploadRequest, ChatRequest, ReportRequest, VoiceRequest]:
    if hasattr(model_class, 'document_id'):
        model_class.__validators__['validate_document_id'] = validator('document_id', allow_reuse=True)(validate_document_id)
    
    if hasattr(model_class, 'session_id'):
        model_class.__validators__['validate_session_id'] = validator('session_id', allow_reuse=True)(validate_session_id)
