"""
Enhanced Pydantic response models for LegalMind AI API endpoints
Production-ready with comprehensive response structures, validation, and documentation
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, validator
from pydantic.types import UUID4, confloat, conint


# Enhanced Enums

class ProcessingStatus(str, Enum):
    """Document and task processing status"""
    UPLOADED = "uploaded"
    UPLOADING = "uploading"
    QUEUED = "queued"
    PROCESSING = "processing"
    EXTRACTING_TEXT = "extracting_text"
    ANALYZING = "analyzing"
    TRANSLATING = "translating"
    GENERATING = "generating"
    RENDERING = "rendering"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class RiskLevel(str, Enum):
    """Risk level classifications"""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    """AI confidence levels"""
    VERY_LOW = "very_low"
    LOW = "low" 
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ResponseStatus(str, Enum):
    """General response status"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PARTIAL = "partial"


# Base Response Models

class BaseResponse(BaseModel):
    """Base response model with common fields"""
    success: bool = Field(default=True, description="Operation success status")
    message: Optional[str] = Field(None, description="Optional response message")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp"
    )
    request_id: Optional[str] = Field(None, description="Request tracking ID")
    version: str = Field(default="2.0", description="API version")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PaginatedResponse(BaseResponse):
    """Base paginated response"""
    total_count: int = Field(..., ge=0, description="Total number of items")
    page: int = Field(..., ge=1, description="Current page number")
    per_page: int = Field(..., ge=1, le=100, description="Items per page")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(default=False, description="Has next page")
    has_prev: bool = Field(default=False, description="Has previous page")


# Authentication Response Models

class LoginResponse(BaseResponse):
    """User login response"""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user: Dict[str, Any] = Field(..., description="User information")


class RefreshTokenResponse(BaseResponse):
    """Token refresh response"""
    access_token: str = Field(..., description="New JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")


# Document Response Models

class FileInfo(BaseModel):
    """File information details"""
    filename: str = Field(..., description="File name")
    original_filename: str = Field(..., description="Original uploaded filename")
    size: int = Field(..., ge=0, description="File size in bytes")
    size_formatted: str = Field(..., description="Human-readable file size")
    content_type: str = Field(..., description="MIME content type")
    checksum: Optional[str] = Field(None, description="File checksum for integrity")


class ProcessingStage(BaseModel):
    """Processing stage information"""
    stage: str = Field(..., description="Current processing stage")
    description: str = Field(..., description="Stage description")
    progress: int = Field(..., ge=0, le=100, description="Stage progress percentage")
    started_at: Optional[datetime] = Field(None, description="Stage start time")
    completed_at: Optional[datetime] = Field(None, description="Stage completion time")
    duration: Optional[float] = Field(None, description="Stage duration in seconds")


class DocumentUploadResponse(BaseResponse):
    """Enhanced document upload response"""
    document_id: str = Field(..., description="Unique document identifier")
    session_id: str = Field(..., description="Chat session identifier")
    status: ProcessingStatus = Field(..., description="Initial processing status")
    estimated_time: str = Field(..., description="Estimated processing time")
    priority: str = Field(..., description="Processing priority level")
    file_info: FileInfo = Field(..., description="Uploaded file information")
    processing_stages: Optional[List[str]] = Field(None, description="Expected processing stages")


class DocumentStatusResponse(BaseModel):
    """Detailed document processing status"""
    document_id: str = Field(..., description="Document identifier")
    status: ProcessingStatus = Field(..., description="Current status")
    progress: int = Field(..., ge=0, le=100, description="Overall progress percentage")
    current_stage: Optional[str] = Field(None, description="Current processing stage")
    stage_description: Optional[str] = Field(None, description="Stage description")
    processing_time: Optional[str] = Field(None, description="Total processing time")
    estimated_remaining: Optional[str] = Field(None, description="Estimated remaining time")
    error_message: Optional[str] = Field(None, description="Error details if failed")
    retry_count: int = Field(default=0, description="Number of retry attempts")
    priority: str = Field(default="normal", description="Processing priority")
    last_updated: datetime = Field(..., description="Last status update time")
    can_retry: bool = Field(default=False, description="Whether retry is possible")
    stages: Optional[List[ProcessingStage]] = Field(None, description="Detailed stage information")


class DocumentInfo(BaseModel):
    """Document information summary"""
    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Document filename")
    title: Optional[str] = Field(None, description="Document title")
    upload_date: datetime = Field(..., description="Upload timestamp")
    document_type: Optional[str] = Field(None, description="Classified document type")
    document_subtype: Optional[str] = Field(None, description="Document subtype")
    status: ProcessingStatus = Field(..., description="Processing status")
    risk_level: RiskLevel = Field(..., description="Overall risk level")
    risk_score: Optional[float] = Field(None, ge=0, le=10, description="Risk score (0-10)")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Analysis confidence")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    file_size_formatted: Optional[str] = Field(None, description="Formatted file size")
    page_count: Optional[int] = Field(None, description="Number of pages")
    word_count: Optional[int] = Field(None, description="Word count")
    language: str = Field(default="en", description="Document language")
    tags: List[str] = Field(default=[], description="Document tags")
    processing_time: Optional[str] = Field(None, description="Total processing time")
    has_chat: bool = Field(default=False, description="Has active chat session")
    has_reports: bool = Field(default=False, description="Has generated reports")
    has_voice: bool = Field(default=False, description="Has voice summaries")


class DocumentLibraryResponse(PaginatedResponse):
    """Document library with pagination and filtering"""
    documents: List[DocumentInfo] = Field(..., description="List of documents")
    filters_applied: Dict[str, Any] = Field(default={}, description="Applied filters")
    statistics: Dict[str, Any] = Field(default={}, description="Library statistics")
    sort_options: List[Dict[str, str]] = Field(default=[], description="Available sort options")


# Document Analysis Response Models

class RiskInfo(BaseModel):
    """Individual risk information"""
    risk_id: str = Field(..., description="Unique risk identifier")
    risk_level: RiskLevel = Field(..., description="Risk severity level")
    category: str = Field(..., description="Risk category")
    title: str = Field(..., description="Risk title")
    description: str = Field(..., description="Detailed risk description")
    impact: str = Field(..., description="Potential impact description")
    likelihood: Optional[str] = Field(None, description="Likelihood of occurrence")
    recommendation: str = Field(..., description="Recommended action")
    clause_reference: Optional[str] = Field(None, description="Referenced clause")
    page_number: Optional[int] = Field(None, description="Page number where found")
    section: Optional[str] = Field(None, description="Document section")
    priority: int = Field(default=1, ge=1, le=5, description="Priority level (1-5)")
    confidence: float = Field(..., ge=0, le=1, description="AI confidence in this risk")


class RiskCategory(BaseModel):
    """Risk category summary"""
    category: str = Field(..., description="Category name")
    score: float = Field(..., ge=0, le=10, description="Category risk score")
    risk_count: int = Field(..., ge=0, description="Number of risks in category")
    severity_distribution: Dict[str, int] = Field(default={}, description="Risk severity breakdown")
    risks: List[RiskInfo] = Field(..., description="Individual risks in category")


class KeyTerm(BaseModel):
    """Extracted key legal term"""
    term: str = Field(..., description="Legal term")
    definition: Optional[str] = Field(None, description="Term definition")
    category: Optional[str] = Field(None, description="Term category")
    importance: float = Field(..., ge=0, le=1, description="Term importance score")
    occurrences: int = Field(..., ge=1, description="Number of occurrences")
    context: Optional[str] = Field(None, description="Usage context")


class DocumentStatistics(BaseModel):
    """Document processing statistics"""
    page_count: int = Field(..., ge=0, description="Number of pages")
    word_count: int = Field(..., ge=0, description="Total word count")
    paragraph_count: int = Field(..., ge=0, description="Number of paragraphs")
    sentence_count: int = Field(..., ge=0, description="Number of sentences")
    character_count: int = Field(..., ge=0, description="Total character count")
    estimated_reading_time: str = Field(..., description="Estimated reading time")
    complexity_score: Optional[float] = Field(None, ge=0, le=1, description="Text complexity score")


class ProcessingMetadata(BaseModel):
    """Processing metadata"""
    processing_time: str = Field(..., description="Total processing time")
    completed_at: datetime = Field(..., description="Processing completion time")
    language: str = Field(..., description="Detected/specified language")
    version: str = Field(..., description="Analysis version")
    model_used: str = Field(..., description="AI model identifier")
    extraction_method: str = Field(..., description="Text extraction method")
    confidence_metrics: Dict[str, float] = Field(default={}, description="Various confidence scores")


class DocumentSummaryResponse(BaseModel):
    """Comprehensive document analysis summary"""
    document_id: str = Field(..., description="Document identifier")
    document_type: str = Field(..., description="Classified document type")
    document_subtype: Optional[str] = Field(None, description="Document subtype")
    title: str = Field(..., description="Document title")
    summary: str = Field(..., description="Executive summary")
    executive_summary: Optional[str] = Field(None, description="Detailed executive summary")
    
    # Risk Analysis
    risk_score: float = Field(..., ge=0, le=10, description="Overall risk score")
    risk_level: RiskLevel = Field(..., description="Overall risk level")
    risk_categories: Dict[str, float] = Field(..., description="Risk scores by category")
    key_risks: List[str] = Field(..., description="Top identified risks")
    risk_details: List[RiskInfo] = Field(..., description="Detailed risk analysis")
    
    # Recommendations and Obligations
    recommendations: List[str] = Field(..., description="Key recommendations")
    user_obligations: List[str] = Field(..., description="User obligations identified")
    user_rights: List[str] = Field(..., description="User rights identified")
    
    # Financial and Legal
    financial_implications: Dict[str, Any] = Field(..., description="Financial impact analysis")
    key_terms: List[KeyTerm] = Field(..., description="Important legal terms")
    
    # Document Analysis
    fairness_score: Optional[float] = Field(None, ge=0, le=10, description="Contract fairness score")
    complexity_analysis: Dict[str, Any] = Field(default={}, description="Document complexity metrics")
    
    # Statistics and Metadata
    document_statistics: DocumentStatistics = Field(..., description="Document statistics")
    processing_metadata: ProcessingMetadata = Field(..., description="Processing information")
    confidence_score: float = Field(..., ge=0, le=1, description="Overall analysis confidence")


class DocumentAnalysisResponse(BaseModel):
    """Detailed document analysis response"""
    document_id: str = Field(..., description="Document identifier")
    analysis_id: str = Field(..., description="Analysis session identifier")
    query: str = Field(..., description="Analysis query")
    response: str = Field(..., description="Analysis response")
    confidence: float = Field(..., ge=0, le=1, description="Response confidence")
    analysis_depth: str = Field(..., description="Analysis depth level")
    focus_areas: List[str] = Field(..., description="Analysis focus areas")
    findings: Dict[str, Any] = Field(..., description="Detailed findings")
    citations: List[str] = Field(..., description="Source citations")
    follow_up_suggestions: List[str] = Field(..., description="Suggested follow-up questions")
    processing_time: float = Field(..., description="Analysis processing time")
    timestamp: datetime = Field(..., description="Analysis timestamp")


# Chat Response Models

class ChatMessage(BaseModel):
    """Individual chat message"""
    message_id: str = Field(..., description="Message identifier")
    type: str = Field(..., description="Message type (user/assistant/system)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="Message timestamp")
    language: str = Field(default="en", description="Message language")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="AI confidence")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    citations: List[str] = Field(default=[], description="Source citations")
    suggestions: List[str] = Field(default=[], description="Follow-up suggestions")
    user_rating: Optional[int] = Field(None, ge=-1, le=1, description="User feedback rating")
    tokens_used: Optional[int] = Field(None, description="Tokens consumed")


class ChatResponse(BaseResponse):
    """Chat AI response"""
    message_id: str = Field(..., description="Response message ID")
    session_id: str = Field(..., description="Chat session ID")
    response: str = Field(..., description="AI response content")
    citations: List[str] = Field(..., description="Source citations")
    confidence: float = Field(..., ge=0, le=1, description="Response confidence")
    suggestions: List[str] = Field(default=[], description="Follow-up suggestions")
    follow_up_questions: List[str] = Field(default=[], description="Suggested questions")
    processing_time: float = Field(..., description="Response generation time")
    tokens_used: int = Field(..., description="Tokens consumed")
    language: str = Field(..., description="Response language")
    context_used: Optional[str] = Field(None, description="Context information used")


class ChatHistoryResponse(BaseModel):
    """Chat conversation history"""
    session_id: str = Field(..., description="Chat session identifier")
    document_id: str = Field(..., description="Associated document ID")
    messages: List[ChatMessage] = Field(..., description="Conversation messages")
    total_messages: int = Field(..., description="Total message count")
    session_start: datetime = Field(..., description="Session start time")
    last_activity: datetime = Field(..., description="Last activity time")
    is_active: bool = Field(..., description="Session active status")
    language: str = Field(..., description="Session language")
    message_count_by_type: Dict[str, int] = Field(default={}, description="Message count by type")


class SuggestedQuestionsResponse(BaseModel):
    """AI-suggested questions for document"""
    document_id: str = Field(..., description="Document identifier")
    document_type: str = Field(..., description="Document type")
    questions: List[str] = Field(..., description="Suggested questions")
    categories: Dict[str, List[str]] = Field(..., description="Questions by category")
    context_specific: List[str] = Field(default=[], description="Context-specific questions")
    generated_at: datetime = Field(..., description="Generation timestamp")
    confidence: float = Field(..., ge=0, le=1, description="Suggestion quality confidence")


# Report Response Models

class ChartData(BaseModel):
    """Chart/visualization data"""
    chart_id: str = Field(..., description="Chart identifier")
    chart_type: str = Field(..., description="Chart type (bar, pie, line, etc.)")
    title: str = Field(..., description="Chart title")
    data: Dict[str, Any] = Field(..., description="Chart data")
    labels: Optional[List[str]] = Field(None, description="Data labels")
    colors: Optional[List[str]] = Field(None, description="Color scheme")
    description: Optional[str] = Field(None, description="Chart description")
    insights: List[str] = Field(default=[], description="Key insights from chart")


class ReportSection(BaseModel):
    """Report section content"""
    section_id: str = Field(..., description="Section identifier")
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content")
    subsections: List[Dict[str, str]] = Field(default=[], description="Subsection content")
    charts: List[ChartData] = Field(default=[], description="Associated charts")
    key_points: List[str] = Field(default=[], description="Key section points")
    references: List[str] = Field(default=[], description="References and citations")


class ReportMetadata(BaseModel):
    """Report generation metadata"""
    template: str = Field(..., description="Report template used")
    language: str = Field(..., description="Report language")
    export_format: str = Field(..., description="Export format")
    generation_time: float = Field(..., description="Generation time in seconds")
    page_count: Optional[int] = Field(None, description="Report page count")
    word_count: Optional[int] = Field(None, description="Report word count")
    sections_included: List[str] = Field(..., description="Included sections")
    custom_branding: bool = Field(default=False, description="Custom branding applied")
    watermark: Optional[str] = Field(None, description="Watermark text")


class ReportGenerationResponse(BaseResponse):
    """Report generation initiation response"""
    report_id: str = Field(..., description="Report identifier")
    status: ProcessingStatus = Field(..., description="Generation status")
    estimated_time: str = Field(..., description="Estimated completion time")
    template: str = Field(..., description="Selected template")
    language: str = Field(..., description="Report language")
    export_format: str = Field(..., description="Export format")
    sections_included: int = Field(..., description="Number of sections")
    priority: str = Field(default="normal", description="Processing priority")


class ReportStatusResponse(BaseModel):
    """Report generation status"""
    report_id: str = Field(..., description="Report identifier")
    status: ProcessingStatus = Field(..., description="Current status")
    progress: int = Field(..., ge=0, le=100, description="Progress percentage")
    current_stage: Optional[str] = Field(None, description="Current generation stage")
    stage_description: Optional[str] = Field(None, description="Stage description")
    processing_time: Optional[str] = Field(None, description="Processing time")
    estimated_remaining: Optional[str] = Field(None, description="Estimated remaining time")
    error_message: Optional[str] = Field(None, description="Error details if failed")
    file_size: Optional[int] = Field(None, description="Generated file size")
    file_size_formatted: Optional[str] = Field(None, description="Formatted file size")
    download_available: bool = Field(default=False, description="Download ready")
    expires_at: Optional[datetime] = Field(None, description="Download expiration")


class ReportLibraryResponse(PaginatedResponse):
    """User's report library"""
    reports: List[Dict[str, Any]] = Field(..., description="Report summaries")
    statistics: Dict[str, Any] = Field(default={}, description="Library statistics")


class ReportTemplateInfo(BaseModel):
    """Report template information"""
    template_id: str = Field(..., description="Template identifier")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    category: str = Field(..., description="Template category")
    sections: List[str] = Field(..., description="Available sections")
    languages: List[str] = Field(..., description="Supported languages")
    export_formats: List[str] = Field(..., description="Supported export formats")
    preview_url: Optional[str] = Field(None, description="Template preview URL")
    custom_options: Dict[str, Any] = Field(default={}, description="Customization options")


class ReportTemplateResponse(BaseModel):
    """Available report templates"""
    templates: List[ReportTemplateInfo] = Field(..., description="Available templates")
    total_count: int = Field(..., description="Total template count")
    supported_languages: List[str] = Field(..., description="All supported languages")
    categories: List[str] = Field(..., description="Template categories")


# Voice Response Models

class VoiceGenerationResponse(BaseResponse):
    """Voice generation initiation response"""
    voice_id: str = Field(..., description="Voice generation identifier")
    status: ProcessingStatus = Field(..., description="Generation status")
    estimated_time: str = Field(..., description="Estimated completion time")
    estimated_duration: str = Field(..., description="Estimated audio duration")
    language: str = Field(..., description="Voice language")
    voice_type: str = Field(..., description="Voice type")
    content_info: Dict[str, Any] = Field(..., description="Content information")


class VoiceProgressResponse(BaseModel):
    """Voice generation progress"""
    voice_id: str = Field(..., description="Voice identifier")
    status: ProcessingStatus = Field(..., description="Current status")
    progress: int = Field(..., ge=0, le=100, description="Progress percentage")
    current_stage: Optional[str] = Field(None, description="Current stage")
    stage_description: Optional[str] = Field(None, description="Stage description")
    processing_time: Optional[str] = Field(None, description="Processing time")
    estimated_remaining: Optional[str] = Field(None, description="Estimated remaining time")
    error_message: Optional[str] = Field(None, description="Error details")
    audio_preview_available: bool = Field(default=False, description="Preview available")
    metadata: Dict[str, Any] = Field(default={}, description="Generation metadata")


class VoiceSummaryResponse(BaseModel):
    """Voice summary details"""
    voice_id: str = Field(..., description="Voice identifier")
    status: ProcessingStatus = Field(..., description="Generation status")
    audio_url: Optional[str] = Field(None, description="Audio stream URL")
    download_url: Optional[str] = Field(None, description="Download URL")
    duration: Optional[str] = Field(None, description="Audio duration")
    language: str = Field(..., description="Audio language")
    voice_type: str = Field(..., description="Voice type used")
    content_type: str = Field(..., description="Content type vocalized")
    file_size: Optional[int] = Field(None, description="Audio file size")
    file_size_formatted: Optional[str] = Field(None, description="Formatted file size")
    audio_format: str = Field(default="mp3", description="Audio format")
    transcript: Optional[str] = Field(None, description="Audio transcript")
    created_at: datetime = Field(..., description="Creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    metadata: Dict[str, Any] = Field(default={}, description="Voice metadata")


class VoiceLibraryResponse(PaginatedResponse):
    """User's voice library"""
    voice_summaries: List[Dict[str, Any]] = Field(..., description="Voice summary list")
    statistics: Dict[str, Any] = Field(default={}, description="Library statistics")


# Translation Response Models

class TranslationResponse(BaseModel):
    """Translation operation response"""
    translation_id: str = Field(..., description="Translation identifier")
    status: ProcessingStatus = Field(..., description="Translation status")
    source_language: str = Field(..., description="Source language")
    target_language: str = Field(..., description="Target language")
    content_type: str = Field(..., description="Content type translated")
    translated_content: Dict[str, str] = Field(..., description="Translated content")
    confidence: float = Field(..., ge=0, le=1, description="Translation confidence")
    processing_time: float = Field(..., description="Translation time")
    word_count: int = Field(..., description="Word count")
    character_count: int = Field(..., description="Character count")
    timestamp: datetime = Field(..., description="Translation timestamp")


# Comparison and Analysis Response Models

class KeyDifference(BaseModel):
    """Key difference in document comparison"""
    clause_type: str = Field(..., description="Type of clause")
    section: str = Field(..., description="Document section")
    user_terms: str = Field(..., description="User's document terms")
    industry_standard: str = Field(..., description="Industry standard terms")
    favorability: str = Field(..., description="Favorability assessment")
    impact_score: float = Field(..., ge=0, le=10, description="Impact score")
    recommendation: str = Field(..., description="Recommended action")
    priority: str = Field(..., description="Priority level")
    risk_level: RiskLevel = Field(..., description="Associated risk level")


class ComparisonResponse(BaseModel):
    """Document comparison analysis"""
    comparison_id: str = Field(..., description="Comparison identifier")
    document_id: str = Field(..., description="Analyzed document ID")
    comparison_type: str = Field(..., description="Type of comparison")
    document_favorability: float = Field(..., ge=0, le=10, description="Document favorability score")
    industry_average: float = Field(..., ge=0, le=10, description="Industry average score")
    better_than_percent: int = Field(..., ge=0, le=100, description="Percentile ranking")
    key_differences: List[KeyDifference] = Field(..., description="Key differences identified")
    overall_assessment: str = Field(..., description="Overall comparison assessment")
    recommendations: List[str] = Field(..., description="Improvement recommendations")
    benchmark_data: Dict[str, Any] = Field(default={}, description="Benchmark statistics")
    analysis_date: datetime = Field(..., description="Analysis timestamp")
    confidence: float = Field(..., ge=0, le=1, description="Analysis confidence")


class RiskAnalysisResponse(BaseModel):
    """Comprehensive risk analysis response"""
    document_id: str = Field(..., description="Document identifier")
    analysis_id: str = Field(..., description="Analysis identifier")
    overall_risk_score: float = Field(..., ge=0, le=10, description="Overall risk score")
    risk_level: RiskLevel = Field(..., description="Overall risk level")
    risk_categories: Dict[str, float] = Field(..., description="Risk scores by category")
    risk_distribution: Dict[str, int] = Field(..., description="Risk count by severity")
    flagged_clauses: List[Dict[str, Any]] = Field(..., description="Flagged problematic clauses")
    risk_details: List[RiskInfo] = Field(..., description="Detailed risk information")
    mitigation_strategies: List[str] = Field(..., description="Risk mitigation strategies")
    risk_summary: str = Field(..., description="Executive risk summary")
    compliance_issues: List[str] = Field(default=[], description="Compliance concerns")
    legal_precedents: List[str] = Field(default=[], description="Relevant legal precedents")
    analysis_depth: str = Field(..., description="Analysis depth level")
    analysis_date: datetime = Field(..., description="Analysis timestamp")
    confidence: float = Field(..., ge=0, le=1, description="Analysis confidence")
    recommendations: List[str] = Field(..., description="Risk mitigation recommendations")


# System Response Models

class ServiceStatus(BaseModel):
    """Individual service health status"""
    service_name: str = Field(..., description="Service identifier")
    status: str = Field(..., description="Service status")
    response_time: Optional[float] = Field(None, description="Response time in seconds")
    last_check: datetime = Field(..., description="Last health check time")
    error: Optional[str] = Field(None, description="Error message if unhealthy")
    details: Dict[str, Any] = Field(default={}, description="Additional service details")


class HealthResponse(BaseModel):
    """System health check response"""
    status: str = Field(..., description="Overall system status")
    services: List[ServiceStatus] = Field(..., description="Individual service statuses")
    uptime: str = Field(..., description="System uptime")
    app_uptime: str = Field(..., description="Application uptime")
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Deployment environment")
    timestamp: datetime = Field(..., description="Health check timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier")
    system_info: Dict[str, Any] = Field(default={}, description="System information")
    performance_metrics: Dict[str, Any] = Field(default={}, description="Performance metrics")
    checks_performed: int = Field(..., description="Number of checks performed")
    error: Optional[str] = Field(None, description="Overall error if unhealthy")


class SystemStatusResponse(BaseModel):
    """Comprehensive system status"""
    system_info: Dict[str, Any] = Field(..., description="System information")
    service_statistics: Dict[str, Any] = Field(..., description="Service statistics")
    system_resources: Dict[str, Any] = Field(..., description="Resource utilization")
    application_metrics: Dict[str, Any] = Field(..., description="Application metrics")
    storage: Dict[str, Any] = Field(..., description="Storage information")
    network: Dict[str, Any] = Field(default={}, description="Network information")
    configuration: Dict[str, Any] = Field(..., description="System configuration")
    feature_flags: Dict[str, bool] = Field(..., description="Feature flags status")
    last_updated: datetime = Field(..., description="Last update timestamp")


class SystemMetricsResponse(BaseModel):
    """System metrics and analytics"""
    time_range: str = Field(..., description="Metrics time range")
    start_time: datetime = Field(..., description="Metrics start time")
    end_time: datetime = Field(..., description="Metrics end time")
    performance_data: Dict[str, Any] = Field(..., description="Performance metrics")
    error_statistics: Dict[str, Any] = Field(..., description="Error statistics")
    usage_analytics: Dict[str, Any] = Field(..., description="Usage analytics")
    resource_trends: Dict[str, Any] = Field(..., description="Resource utilization trends")
    service_metrics: Dict[str, Any] = Field(..., description="Service-specific metrics")
    summary: Dict[str, Any] = Field(..., description="Metrics summary")


class LanguageInfo(BaseModel):
    """Language support information"""
    code: str = Field(..., description="Language code")
    name: str = Field(..., description="Language name")
    native_name: str = Field(..., description="Native language name")
    supported_features: List[str] = Field(..., description="Supported features")
    is_primary: bool = Field(default=False, description="Primary language")
    script: Optional[str] = Field(None, description="Writing script")
    family: Optional[str] = Field(None, description="Language family")
    quality: str = Field(..., description="Support quality level")
    regional_variants: List[str] = Field(default=[], description="Regional variants")


class SupportedLanguagesResponse(BaseModel):
    """Supported languages information"""
    languages: List[LanguageInfo] = Field(..., description="Supported languages")
    total_languages: int = Field(..., description="Total language count")
    default_language: str = Field(..., description="Default language")
    primary_languages: List[str] = Field(..., description="Primary languages")
    features_by_language: Dict[str, List[str]] = Field(..., description="Features by language")
    quality_levels: Dict[str, int] = Field(..., description="Quality level distribution")


# Session and User Response Models

class SessionInfo(BaseModel):
    """Session information"""
    session_id: str = Field(..., description="Session identifier")
    document_id: str = Field(..., description="Associated document")
    user_id: str = Field(..., description="User identifier")
    language: str = Field(..., description="Session language")
    is_active: bool = Field(..., description="Session active status")
    created_at: datetime = Field(..., description="Session creation time")
    last_activity: datetime = Field(..., description="Last activity time")
    expires_at: datetime = Field(..., description="Session expiration time")
    message_count: int = Field(..., description="Total message count")
    preferences: Dict[str, Any] = Field(default={}, description="Session preferences")
    capabilities: List[str] = Field(default=[], description="Session capabilities")


class SessionResponse(BaseResponse):
    """Session creation response"""
    session_info: SessionInfo = Field(..., description="Session information")
    welcome_message: str = Field(..., description="Welcome message")
    suggested_questions: List[str] = Field(default=[], description="Suggested starter questions")


# Analytics and Reporting Response Models

class AnalyticsTimePoint(BaseModel):
    """Analytics data point"""
    timestamp: datetime = Field(..., description="Data point timestamp")
    value: float = Field(..., description="Metric value")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")


class AnalyticsResponse(BaseModel):
    """Analytics data response"""
    metric_name: str = Field(..., description="Metric name")
    time_range: str = Field(..., description="Time range")
    data_points: List[AnalyticsTimePoint] = Field(..., description="Time series data")
    summary_stats: Dict[str, float] = Field(..., description="Summary statistics")
    trends: Dict[str, str] = Field(default={}, description="Trend analysis")
    generated_at: datetime = Field(..., description="Report generation time")


class ReportAnalyticsResponse(BaseModel):
    """Report analytics response"""
    analytics: Dict[str, Any] = Field(..., description="Analytics data")
    generated_at: datetime = Field(..., description="Generation timestamp")


# Error Response Models

class ValidationErrorDetail(BaseModel):
    """Validation error detail"""
    field: str = Field(..., description="Field name")
    message: str = Field(..., description="Error message")
    invalid_value: Optional[str] = Field(None, description="Invalid value")
    expected_type: Optional[str] = Field(None, description="Expected type/format")


class ErrorResponse(BaseModel):
    """Enhanced error response"""
    success: bool = Field(default=False, description="Operation success status")
    error: Dict[str, Any] = Field(..., description="Error details")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Error timestamp"
    )
    path: Optional[str] = Field(None, description="Request path")
    method: Optional[str] = Field(None, description="HTTP method")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Utility Response Models

class BulkOperationResponse(BaseResponse):
    """Bulk operation response"""
    batch_id: str = Field(..., description="Batch operation identifier")
    total_items: int = Field(..., description="Total items to process")
    successful: int = Field(default=0, description="Successfully processed items")
    failed: int = Field(default=0, description="Failed items")
    skipped: int = Field(default=0, description="Skipped items")
    in_progress: int = Field(default=0, description="Items in progress")
    results: List[Dict[str, Any]] = Field(default=[], description="Individual results")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")


class ExportResponse(BaseModel):
    """Data export response"""
    export_id: str = Field(..., description="Export identifier")
    format: str = Field(..., description="Export format")
    status: ProcessingStatus = Field(..., description="Export status")
    download_url: Optional[str] = Field(None, description="Download URL")
    file_size: Optional[int] = Field(None, description="File size")
    expires_at: datetime = Field(..., description="Download expiration")
    records_count: int = Field(..., description="Number of exported records")
    created_at: datetime = Field(..., description="Export creation time")


# Webhook and Notification Response Models

class NotificationResponse(BaseModel):
    """Notification response"""
    notification_id: str = Field(..., description="Notification identifier")
    type: str = Field(..., description="Notification type")
    title: str = Field(..., description="Notification title")
    message: str = Field(..., description="Notification message")
    priority: str = Field(..., description="Notification priority")
    read: bool = Field(default=False, description="Read status")
    created_at: datetime = Field(..., description="Creation timestamp")
    data: Dict[str, Any] = Field(default={}, description="Additional notification data")


# Add response validation
@validator('timestamp', pre=True, always=True)
def ensure_utc_timezone(cls, v):
    """Ensure timestamp is in UTC"""
    if isinstance(v, datetime) and v.tzinfo is None:
        return v.replace(tzinfo=timezone.utc)
    return v


# Apply timestamp validator to all response models with timestamp fields
for model_name in globals():
    model = globals()[model_name]
    if (isinstance(model, type) and 
        issubclass(model, BaseModel) and 
        hasattr(model, '__fields__') and 
        'timestamp' in model.__fields__):
        model.__validators__['ensure_utc_timestamp'] = ensure_utc_timezone
