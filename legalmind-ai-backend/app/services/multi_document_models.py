"""
Data models for multi-document conversational assistant
Enhanced with timezone awareness and comprehensive validation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
from enum import Enum

import re
from dotenv import load_dotenv
load_dotenv() 
import uuid

# Enums for better type safety
class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    TXT = "txt"
    RTF = "rtf"
    MD = "md"  # Added markdown support
    HTML = "html"  # Added HTML support

class RiskLevel(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"  # For when risk cannot be determined

class RelationshipType(str, Enum):
    HIGHLY_RELATED = "highly_related"
    RELATED = "related"
    LOOSELY_RELATED = "loosely_related"
    COMPLEMENTARY = "complementary"
    CONTRADICTORY = "contradictory"
    SUPERSEDES = "supersedes"  # When one document replaces another
    REFERENCES = "references"  # When one document references another

class IntentType(str, Enum):
    QUESTION = "question"
    COMPARISON = "comparison"
    RISK_ASSESSMENT = "risk_assessment"
    EXPLANATION = "explanation"
    ACTION_REQUEST = "action_request"
    DOCUMENT_UPLOAD = "document_upload"
    GENERAL_INQUIRY = "general_inquiry"
    CLARIFICATION = "clarification"  # Added for follow-up questions
    SUMMARY_REQUEST = "summary_request"  # Added for summary requests

class SuggestionPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"  # Added for informational suggestions

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # When only part of document was processed

@dataclass
class StandardDocument:
    """Standardized document representation across all formats"""
    id: str
    session_id: str
    filename: str
    content_type: str
    content: str
    metadata: Dict[str, Any]
    page_count: int
    word_count: int
    file_hash: str
    created_at: datetime
    processing_status: ProcessingStatus = ProcessingStatus.COMPLETED
    file_size: Optional[int] = None
    language: str = "en"
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Ensure timezone awareness and validation"""
        if self.created_at and self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)
        
        if self.updated_at and self.updated_at.tzinfo is None:
            self.updated_at = self.updated_at.replace(tzinfo=timezone.utc)
        
        # Set updated_at to created_at if not provided
        if not self.updated_at:
            self.updated_at = self.created_at
        
        # Validate required fields
        if not self.id:
            self.id = f"doc_{uuid.uuid4().hex[:12]}"
        
        # Ensure processing_status is enum
        if isinstance(self.processing_status, str):
            self.processing_status = ProcessingStatus(self.processing_status)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'filename': self.filename,
            'content_type': self.content_type,
            'page_count': self.page_count,
            'word_count': self.word_count,
            'file_hash': self.file_hash,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'processing_status': self.processing_status.value if isinstance(self.processing_status, ProcessingStatus) else self.processing_status,
            'file_size': self.file_size,
            'language': self.language,
            'metadata': self.metadata
        }
    
    def update_status(self, status: ProcessingStatus):
        """Update processing status and timestamp"""
        self.processing_status = status
        self.updated_at = datetime.now(timezone.utc)

@dataclass
class DocumentChunk:
    """Represents a chunk of a document with intelligent boundaries"""
    id: str
    document_id: str
    chunk_index: int
    total_chunks: int
    content: str
    page_range: str
    char_range: str
    word_count: int
    metadata: Dict[str, Any]
    created_at: datetime
    embedding_vector: Optional[List[float]] = None
    semantic_tags: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    
    def __post_init__(self):
        """Ensure timezone awareness and validation"""
        if self.created_at and self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)
        
        # Generate ID if not provided
        if not self.id:
            self.id = f"chunk_{self.document_id}_{self.chunk_index}"
        
        # Calculate word count if not provided
        if not self.word_count and self.content:
            self.word_count = len(self.content.split())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'document_id': self.document_id,
            'chunk_index': self.chunk_index,
            'total_chunks': self.total_chunks,
            'page_range': self.page_range,
            'char_range': self.char_range,
            'word_count': self.word_count,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'has_embedding': self.embedding_vector is not None,
            'semantic_tags': self.semantic_tags,
            'quality_score': self.quality_score
        }
    
    def add_semantic_tag(self, tag: str):
        """Add semantic tag if not already present"""
        if tag and tag not in self.semantic_tags:
            self.semantic_tags.append(tag)

@dataclass
class DocumentRelationship:
    """Represents relationship between two documents"""
    id: str
    document_1_id: str
    document_2_id: str
    relationship_type: RelationshipType
    similarity_score: float
    common_topics: List[str]
    relationship_details: Dict[str, Any]
    created_at: datetime
    confidence_score: Optional[float] = None
    strength: str = "medium"  # weak, medium, strong
    
    def __post_init__(self):
        """Ensure timezone awareness and validation"""
        if self.created_at and self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)
        
        # Generate ID if not provided
        if not self.id:
            self.id = f"rel_{self.document_1_id}_{self.document_2_id}"
        
        # Ensure relationship_type is enum
        if isinstance(self.relationship_type, str):
            self.relationship_type = RelationshipType(self.relationship_type)
        
        # Calculate strength based on similarity score
        if self.similarity_score >= 0.8:
            self.strength = "strong"
        elif self.similarity_score >= 0.5:
            self.strength = "medium"
        else:
            self.strength = "weak"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'document_1_id': self.document_1_id,
            'document_2_id': self.document_2_id,
            'relationship_type': self.relationship_type.value if isinstance(self.relationship_type, RelationshipType) else self.relationship_type,
            'similarity_score': self.similarity_score,
            'common_topics': self.common_topics,
            'relationship_details': self.relationship_details,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'confidence_score': self.confidence_score,
            'strength': self.strength
        }

@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation"""
    turn_id: int
    timestamp: datetime
    user_message: str
    assistant_response: str
    source_documents: List[str]
    topics_discussed: List[str]
    user_intent: IntentType
    response_metadata: Dict[str, Any] = field(default_factory=dict)
    confidence_score: Optional[float] = None
    processing_time: float = 0.0
    language: str = "en"
    
    def __post_init__(self):
        """Ensure timezone awareness and validation"""
        if self.timestamp and self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
        
        # Ensure user_intent is enum
        if isinstance(self.user_intent, str):
            self.user_intent = IntentType(self.user_intent)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'turn_id': self.turn_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'user_message': self.user_message,
            'assistant_response': self.assistant_response,
            'source_documents': self.source_documents,
            'topics_discussed': self.topics_discussed,
            'user_intent': self.user_intent.value if isinstance(self.user_intent, IntentType) else self.user_intent,
            'response_metadata': self.response_metadata,
            'confidence_score': self.confidence_score,
            'processing_time': self.processing_time,
            'language': self.language
        }

@dataclass
class SourceAttribution:
    """Attribution information for response sources"""
    document_id: str
    document_title: str
    page_range: str
    chunk_id: str
    relevance_score: float
    confidence_score: float
    key_concepts: List[str]
    excerpt: str
    citation_text: str = ""  # How to cite this source
    
    def __post_init__(self):
        """Generate citation text if not provided"""
        if not self.citation_text:
            self.citation_text = f"{self.document_title}, p. {self.page_range}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'document_id': self.document_id,
            'document_title': self.document_title,
            'page_range': self.page_range,
            'chunk_id': self.chunk_id,
            'relevance_score': self.relevance_score,
            'confidence_score': self.confidence_score,
            'key_concepts': self.key_concepts,
            'excerpt': self.excerpt,
            'citation_text': self.citation_text
        }

@dataclass
class SynthesizedResponse:
    """Complete synthesized response with all metadata"""
    query: str
    answer: str
    source_attributions: List[SourceAttribution]
    synthesis_strategy: str
    confidence_score: float
    quality_metrics: Dict[str, Any]
    follow_up_suggestions: List[str]
    cross_references: List[Dict[str, Any]]
    response_metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    language: str = "en"
    word_count: int = 0
    
    def __post_init__(self):
        """Calculate metrics and ensure timezone awareness"""
        if self.timestamp and self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
        
        # Calculate word count
        if not self.word_count and self.answer:
            self.word_count = len(self.answer.split())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'query': self.query,
            'answer': self.answer,
            'source_attributions': [attr.to_dict() for attr in self.source_attributions],
            'synthesis_strategy': self.synthesis_strategy,
            'confidence_score': self.confidence_score,
            'quality_metrics': self.quality_metrics,
            'follow_up_suggestions': self.follow_up_suggestions,
            'cross_references': self.cross_references,
            'response_metadata': self.response_metadata,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'language': self.language,
            'word_count': self.word_count
        }

@dataclass
class IntegrationSuggestion:
    """Suggestion for integrating new document with existing knowledge"""
    type: str
    priority: SuggestionPriority
    message: str
    suggested_queries: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    action_required: bool = False
    expires_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Ensure timezone awareness and validation"""
        if self.expires_at and self.expires_at.tzinfo is None:
            self.expires_at = self.expires_at.replace(tzinfo=timezone.utc)
        
        # Ensure priority is enum
        if isinstance(self.priority, str):
            self.priority = SuggestionPriority(self.priority)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'type': self.type,
            'priority': self.priority.value if isinstance(self.priority, SuggestionPriority) else self.priority,
            'message': self.message,
            'suggested_queries': self.suggested_queries,
            'metadata': self.metadata,
            'action_required': self.action_required,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }
    
    def is_expired(self) -> bool:
        """Check if suggestion has expired"""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at

@dataclass
class DocumentSummary:
    """High-level summary of document for conversation context"""
    document_id: str
    title: str
    document_type: str
    key_topics: List[str]
    main_entities: List[str]
    risk_level: RiskLevel
    summary_text: str
    chunk_count: int
    created_at: datetime
    word_count: Optional[int] = None
    page_count: Optional[int] = None
    confidence_score: float = 0.8
    processing_notes: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Ensure timezone awareness and validation"""
        if self.created_at and self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)
        
        # Ensure risk_level is enum
        if isinstance(self.risk_level, str):
            self.risk_level = RiskLevel(self.risk_level)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'document_id': self.document_id,
            'title': self.title,
            'document_type': self.document_type,
            'key_topics': self.key_topics,
            'main_entities': self.main_entities,
            'risk_level': self.risk_level.value if isinstance(self.risk_level, RiskLevel) else self.risk_level,
            'summary_text': self.summary_text,
            'chunk_count': self.chunk_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'word_count': self.word_count,
            'page_count': self.page_count,
            'confidence_score': self.confidence_score,
            'processing_notes': self.processing_notes
        }
    
    def add_processing_note(self, note: str):
        """Add processing note"""
        if note and note not in self.processing_notes:
            self.processing_notes.append(note)

@dataclass
class ContextualQuery:
    """Query with conversation context"""
    query: str
    session_id: str
    context_turns: List[ConversationTurn]
    current_topics: List[str]
    active_documents: List[str]
    user_intent: IntentType
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    language: str = "en"
    
    def __post_init__(self):
        """Ensure timezone awareness and validation"""
        if self.timestamp and self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
        
        # Ensure user_intent is enum
        if isinstance(self.user_intent, str):
            self.user_intent = IntentType(self.user_intent)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'query': self.query,
            'session_id': self.session_id,
            'context_turns': [turn.to_dict() for turn in self.context_turns],
            'current_topics': self.current_topics,
            'active_documents': self.active_documents,
            'user_intent': self.user_intent.value if isinstance(self.user_intent, IntentType) else self.user_intent,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'language': self.language
        }

@dataclass
class SessionState:
    """Complete state of a multi-document session"""
    session_id: str
    created_at: datetime
    last_updated: datetime
    document_count: int
    total_chunks: int
    conversation_turn_count: int
    dominant_topics: List[str]
    risk_profile: Dict[str, Any]
    user_expertise_level: str
    session_metadata: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    language: str = "en"
    
    def __post_init__(self):
        """Ensure timezone awareness"""
        if self.created_at and self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)
        
        if self.last_updated and self.last_updated.tzinfo is None:
            self.last_updated = self.last_updated.replace(tzinfo=timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'document_count': self.document_count,
            'total_chunks': self.total_chunks,
            'conversation_turn_count': self.conversation_turn_count,
            'dominant_topics': self.dominant_topics,
            'risk_profile': self.risk_profile,
            'user_expertise_level': self.user_expertise_level,
            'session_metadata': self.session_metadata,
            'active': self.active,
            'language': self.language
        }
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_updated = datetime.now(timezone.utc)

# Request/Response models for API endpoints
@dataclass
class MultiDocumentChatRequest:
    """Request model for multi-document chat"""
    session_id: str
    message: str
    language: str = "en"
    response_style: str = "comprehensive"
    max_sources: int = 5
    include_cross_references: bool = True
    temperature: float = 0.7  # For AI response generation
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'message': self.message,
            'language': self.language,
            'response_style': self.response_style,
            'max_sources': self.max_sources,
            'include_cross_references': self.include_cross_references,
            'temperature': self.temperature
        }

@dataclass
class MultiDocumentChatResponse:
    """Response model for multi-document chat"""
    session_id: str
    response: SynthesizedResponse
    session_state: SessionState
    integration_opportunities: List[IntegrationSuggestion]
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'response': self.response.to_dict(),
            'session_state': self.session_state.to_dict(),
            'integration_opportunities': [opp.to_dict() for opp in self.integration_opportunities],
            'processing_time': self.processing_time,
            'success': self.success,
            'error_message': self.error_message,
            'warnings': self.warnings
        }

@dataclass
class DocumentUploadRequest:
    """Request model for document upload"""
    session_id: Optional[str] = None
    filename: str = ""
    language: str = "en"
    auto_analyze: bool = True
    integration_mode: str = "intelligent"  # "intelligent", "append", "replace"
    preserve_formatting: bool = True
    extract_tables: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'filename': self.filename,
            'language': self.language,
            'auto_analyze': self.auto_analyze,
            'integration_mode': self.integration_mode,
            'preserve_formatting': self.preserve_formatting,
            'extract_tables': self.extract_tables
        }

@dataclass
class DocumentUploadResponse:
    """Response model for document upload"""
    document_id: str
    session_id: str
    document_summary: DocumentSummary
    integration_suggestions: List[IntegrationSuggestion]
    relationship_updates: List[Dict[str, Any]]
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'document_id': self.document_id,
            'session_id': self.session_id,
            'document_summary': self.document_summary.to_dict(),
            'integration_suggestions': [sug.to_dict() for sug in self.integration_suggestions],
            'relationship_updates': self.relationship_updates,
            'processing_time': self.processing_time,
            'success': self.success,
            'error_message': self.error_message,
            'warnings': self.warnings
        }

@dataclass
class SessionOverviewResponse:
    """Response model for session overview"""
    session_id: str
    session_state: SessionState
    documents: List[DocumentSummary]
    relationships: List[DocumentRelationship]
    conversation_insights: Dict[str, Any]
    portfolio_analysis: Dict[str, Any]
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'session_state': self.session_state.to_dict(),
            'documents': [doc.to_dict() for doc in self.documents],
            'relationships': [rel.to_dict() for rel in self.relationships],
            'conversation_insights': self.conversation_insights,
            'portfolio_analysis': self.portfolio_analysis,
            'performance_metrics': self.performance_metrics
        }

# Utility functions for model conversion
def create_document_from_dict(data: Dict[str, Any]) -> StandardDocument:
    """Create StandardDocument from dictionary"""
    # Handle datetime fields
    if 'created_at' in data and isinstance(data['created_at'], str):
        try:
            data['created_at'] = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
        except ValueError:
            data['created_at'] = datetime.now(timezone.utc)
    
    if 'updated_at' in data and isinstance(data['updated_at'], str):
        try:
            data['updated_at'] = datetime.fromisoformat(data['updated_at'].replace('Z', '+00:00'))
        except ValueError:
            data['updated_at'] = None
    
    # Handle enum fields
    if 'processing_status' in data and isinstance(data['processing_status'], str):
        try:
            data['processing_status'] = ProcessingStatus(data['processing_status'])
        except ValueError:
            data['processing_status'] = ProcessingStatus.COMPLETED
    
    return StandardDocument(**data)

def create_chunk_from_dict(data: Dict[str, Any]) -> DocumentChunk:
    """Create DocumentChunk from dictionary"""
    if 'created_at' in data and isinstance(data['created_at'], str):
        try:
            data['created_at'] = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
        except ValueError:
            data['created_at'] = datetime.now(timezone.utc)
    
    # Ensure lists exist
    data.setdefault('semantic_tags', [])
    
    return DocumentChunk(**data)

def create_relationship_from_dict(data: Dict[str, Any]) -> DocumentRelationship:
    """Create DocumentRelationship from dictionary"""
    if 'created_at' in data and isinstance(data['created_at'], str):
        try:
            data['created_at'] = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
        except ValueError:
            data['created_at'] = datetime.now(timezone.utc)
    
    if 'relationship_type' in data and isinstance(data['relationship_type'], str):
        try:
            data['relationship_type'] = RelationshipType(data['relationship_type'])
        except ValueError:
            data['relationship_type'] = RelationshipType.RELATED
    
    return DocumentRelationship(**data)

def create_conversation_turn_from_dict(data: Dict[str, Any]) -> ConversationTurn:
    """Create ConversationTurn from dictionary"""
    if 'timestamp' in data and isinstance(data['timestamp'], str):
        try:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        except ValueError:
            data['timestamp'] = datetime.now(timezone.utc)
    
    if 'user_intent' in data and isinstance(data['user_intent'], str):
        try:
            data['user_intent'] = IntentType(data['user_intent'])
        except ValueError:
            data['user_intent'] = IntentType.GENERAL_INQUIRY
    
    # Ensure required defaults
    data.setdefault('response_metadata', {})
    data.setdefault('language', 'en')
    
    return ConversationTurn(**data)

def create_document_summary_from_dict(data: Dict[str, Any]) -> DocumentSummary:
    """Create DocumentSummary from dictionary"""
    if 'created_at' in data and isinstance(data['created_at'], str):
        try:
            data['created_at'] = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
        except ValueError:
            data['created_at'] = datetime.now(timezone.utc)
    
    if 'risk_level' in data and isinstance(data['risk_level'], str):
        try:
            data['risk_level'] = RiskLevel(data['risk_level'])
        except ValueError:
            data['risk_level'] = RiskLevel.MEDIUM
    
    # Ensure lists exist
    data.setdefault('processing_notes', [])
    
    return DocumentSummary(**data)

# Enhanced validation functions
def validate_document_upload_request(data: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Validate document upload request data"""
    errors = []
    
    if not data.get('filename'):
        errors.append("Filename is required")
    elif not isinstance(data['filename'], str):
        errors.append("Filename must be a string")
    
    if 'language' in data and data['language'] not in ['en', 'es', 'fr', 'de', 'hi', 'zh']:
        errors.append("Unsupported language")
    
    if 'integration_mode' in data and data['integration_mode'] not in ['intelligent', 'append', 'replace']:
        errors.append("Invalid integration mode")
    
    return len(errors) == 0, errors

def validate_chat_request(data: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Validate chat request data"""
    errors = []
    
    if not data.get('session_id'):
        errors.append("Session ID is required")
    elif not validate_session_id(data['session_id']):
        errors.append("Invalid session ID format")
    
    if not data.get('message'):
        errors.append("Message is required")
    elif not isinstance(data['message'], str):
        errors.append("Message must be a string")
    elif len(data['message'].strip()) == 0:
        errors.append("Message cannot be empty")
    elif len(data['message']) > 5000:
        errors.append("Message too long (max 5000 characters)")
    
    if 'max_sources' in data:
        try:
            max_sources = int(data['max_sources'])
            if max_sources < 1 or max_sources > 10:
                errors.append("max_sources must be between 1 and 10")
        except (ValueError, TypeError):
            errors.append("max_sources must be an integer")
    
    return len(errors) == 0, errors

def validate_session_id(session_id: str) -> bool:
    """Validate session ID format"""
    if not isinstance(session_id, str):
        return False
    
    return bool(re.match(r'^[a-zA-Z0-9\-_]{8,64}$', session_id))

def validate_document_id(document_id: str) -> bool:
    """Validate document ID format"""
    if not isinstance(document_id, str):
        return False
    
    return bool(re.match(r'^[a-zA-Z0-9\-_]{8,64}$', document_id))

# Error models with enhanced details
@dataclass
class ValidationError:
    """Validation error details"""
    field: str
    message: str
    code: str
    value: Any = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'field': self.field,
            'message': self.message,
            'code': self.code,
            'value': self.value
        }

@dataclass
class ProcessingError:
    """Processing error details"""
    stage: str
    message: str
    code: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'stage': self.stage,
            'message': self.message,
            'code': self.code,
            'details': self.details,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

# Statistics and metrics models
@dataclass
class SessionMetrics:
    """Session performance and usage metrics"""
    session_id: str
    total_documents: int
    total_chunks: int
    total_conversations: int
    average_response_time: float
    user_satisfaction: Optional[float] = None
    most_active_topics: List[str] = field(default_factory=list)
    error_count: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'total_documents': self.total_documents,
            'total_chunks': self.total_chunks,
            'total_conversations': self.total_conversations,
            'average_response_time': self.average_response_time,
            'user_satisfaction': self.user_satisfaction,
            'most_active_topics': self.most_active_topics,
            'error_count': self.error_count,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

@dataclass
class DocumentProcessingMetrics:
    """Metrics for document processing performance"""
    document_id: str
    processing_time: float
    chunk_count: int
    extraction_quality: float
    analysis_quality: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'document_id': self.document_id,
            'processing_time': self.processing_time,
            'chunk_count': self.chunk_count,
            'extraction_quality': self.extraction_quality,
            'analysis_quality': self.analysis_quality,
            'errors': self.errors,
            'warnings': self.warnings,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

# Helper functions for common operations
def generate_session_id() -> str:
    """Generate a unique session ID"""
    return f"session_{uuid.uuid4().hex[:16]}"

def generate_document_id() -> str:
    """Generate a unique document ID"""
    return f"doc_{uuid.uuid4().hex[:12]}"

def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """Generate a unique chunk ID"""
    return f"chunk_{document_id}_{chunk_index:04d}"

def generate_relationship_id(doc1_id: str, doc2_id: str) -> str:
    """Generate a unique relationship ID"""
    # Ensure consistent ordering for bidirectional relationships
    ids = sorted([doc1_id, doc2_id])
    return f"rel_{ids[0]}_{ids[1]}"

def calculate_risk_level(risk_factors: List[str], weights: Dict[str, float] = None) -> RiskLevel:
    """Calculate overall risk level from risk factors"""
    if not risk_factors:
        return RiskLevel.LOW
    
    if not weights:
        weights = {
            'critical': 1.0,
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        }
    
    total_score = sum(weights.get(factor.lower(), 0.3) for factor in risk_factors)
    
    if total_score >= 2.0:
        return RiskLevel.CRITICAL
    elif total_score >= 1.5:
        return RiskLevel.HIGH
    elif total_score >= 0.8:
        return RiskLevel.MEDIUM
    else:
        return RiskLevel.LOW

def merge_metadata(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Safely merge metadata dictionaries"""
    merged = base.copy()
    
    for key, value in updates.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_metadata(merged[key], value)
        else:
            merged[key] = value
    
    return merged
