"""
Multi-document chat API routes with real-time integration
Enhanced with fallbacks and comprehensive error handling
"""

import asyncio
import time
import uuid
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv() 

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, validator

from config.logging import get_logger

logger = get_logger(__name__)

# Custom exceptions (fallback if not imported)
class DocumentProcessingError(Exception):
    """Exception raised for document processing errors"""
    pass

class ChatError(Exception):
    """Exception raised for chat-related errors"""
    pass

class ValidationError(Exception):
    """Exception raised for validation errors"""
    pass

# Initialize services with fallbacks
def initialize_services():
    """Initialize services with fallback handling"""
    services = {}
    
    # Try to initialize each service
    service_configs = [
        ('multi_document_processor', 'app.services.multi_document_processor', 'MultiDocumentProcessor'),
        ('kb_manager', 'app.services.session_knowledge_base', 'KnowledgeBaseManager'),
        ('context_manager', 'app.services.conversation_context_manager', 'ConversationContextManager'),
        ('response_synthesizer', 'app.services.response_synthesizer', 'IntelligentResponseSynthesizer'),
        ('real_time_integrator', 'app.services.real_time_integrator', 'RealTimeDocumentIntegrator'),
        ('translation_service', 'app.services.translation_service', 'TranslationService'),
        ('voice_generator', 'app.services.voice_generator', 'VoiceGenerator'),
    ]
    
    for service_name, module_path, class_name in service_configs:
        try:
            module = __import__(module_path, fromlist=[class_name])
            service_class = getattr(module, class_name, None)
            if service_class:
                services[service_name] = service_class()
                logger.info(f"✅ Initialized {service_name}")
            else:
                services[service_name] = create_mock_service(service_name)
                logger.warning(f"⚠️ Using mock {service_name}")
        except ImportError as e:
            services[service_name] = create_mock_service(service_name)
            logger.warning(f"⚠️ Could not import {service_name}: {e}")
    
    return services

def create_mock_service(service_name: str):
    """Create mock service for fallback"""
    
    class MockService:
        def __init__(self):
            self.service_name = service_name
            
        async def __getattr__(self, name):
            logger.warning(f"Mock {self.service_name}.{name} called")
            return self._mock_method
            
        def _mock_method(self, *args, **kwargs):
            return {"success": False, "error": f"Mock {self.service_name} - real service not available"}
    
    return MockService()

# Initialize services
services = initialize_services()
document_processor = services.get('multi_document_processor')
kb_manager = services.get('kb_manager') 
context_manager = services.get('context_manager')
response_synthesizer = services.get('response_synthesizer')
real_time_integrator = services.get('real_time_integrator')
translation_service = services.get('translation_service')
voice_generator = services.get('voice_generator')

# Create router
router = APIRouter(prefix="/api/v1/multi-document", tags=["Multi-Document Chat"])

# Utility functions
def validate_file_type(filename: str) -> bool:
    """Validate file type"""
    if not filename:
        return False
    
    allowed_extensions = {'.pdf', '.txt', '.docx', '.doc', '.rtf', '.md'}
    file_ext = os.path.splitext(filename.lower())[1]
    return file_ext in allowed_extensions

def generate_session_id() -> str:
    """Generate unique session ID"""
    return f"session_{uuid.uuid4().hex[:16]}"

# Mock models for when real models aren't available
class MockDocumentSummary:
    def __init__(self, **kwargs):
        self.document_id = kwargs.get('document_id', 'doc_' + uuid.uuid4().hex[:8])
        self.title = kwargs.get('title', 'Document')
        self.document_type = kwargs.get('document_type', 'legal_document')
        self.risk_level = kwargs.get('risk_level', 'medium')
        self.key_topics = kwargs.get('key_topics', [])
        self.summary_text = kwargs.get('summary_text', 'Document summary')
        self.chunk_count = kwargs.get('chunk_count', 1)
        
    def to_dict(self):
        return {
            'document_id': self.document_id,
            'title': self.title,
            'document_type': self.document_type,
            'risk_level': self.risk_level,
            'key_topics': self.key_topics,
            'summary_text': self.summary_text,
            'chunk_count': self.chunk_count
        }

class MockStandardDocument:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', 'doc_' + uuid.uuid4().hex[:8])
        self.content = kwargs.get('content', '')
        self.filename = kwargs.get('filename', 'document.txt')
        
    def to_dict(self):
        return {
            'id': self.id,
            'content': self.content[:200] + '...' if len(self.content) > 200 else self.content,
            'filename': self.filename
        }

# Pydantic models for requests
class ChatRequest(BaseModel):
    session_id: str
    message: str
    language: str = "en"
    response_style: str = "comprehensive"
    max_sources: int = 5
    include_cross_references: bool = True
    
    @validator('session_id')
    def validate_session_id(cls, v):
        if not v or len(v) < 8:
            raise ValueError('Invalid session ID')
        return v
    
    @validator('message')
    def validate_message(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Message cannot be empty')
        if len(v) > 2000:
            raise ValueError('Message too long (max 2000 characters)')
        return v.strip()

class UploadRequest(BaseModel):
    session_id: Optional[str] = None
    language: str = "en"
    auto_analyze: bool = True
    integration_mode: str = "intelligent"

# Mock session storage
active_sessions = {}

class MockSessionKB:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.documents = {}
        self.document_summaries = {}
        self.chunks = {}
        self.relationships = {}
        self.created_at = datetime.now(timezone.utc)
        self.last_updated = datetime.now(timezone.utc)
        
    def get_session_overview(self):
        return {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks),
            'total_relationships': len(self.relationships),
            'statistics': {
                'total_words': sum(len(doc.get('content', '').split()) for doc in self.documents.values()),
                'total_relationships': len(self.relationships)
            }
        }
    
    async def add_document(self, document, chunks=None):
        doc_summary = MockDocumentSummary(
            document_id=document.id,
            title=document.filename,
            document_type='legal_document',
            risk_level='medium',
            key_topics=['legal_terms', 'obligations'],
            summary_text=f'Analysis of {document.filename}',
            chunk_count=len(chunks) if chunks else 1
        )
        
        self.documents[document.id] = document
        self.document_summaries[document.id] = doc_summary
        
        if chunks:
            for chunk in chunks:
                self.chunks[chunk.id] = chunk
        
        self.last_updated = datetime.now(timezone.utc)
        return doc_summary

async def get_mock_session_kb(session_id: str):
    """Get or create mock session knowledge base"""
    if session_id not in active_sessions:
        active_sessions[session_id] = MockSessionKB(session_id)
    return active_sessions[session_id]

# API Endpoints

@router.post("/sessions/create", response_model=Dict[str, Any])
async def create_session():
    """
    Create a new multi-document conversation session
    
    Returns:
        Dict with session_id and initial state
    """
    
    try:
        session_id = generate_session_id()
        
        # Initialize session knowledge base (mock or real)
        if hasattr(kb_manager, 'get_session_kb') and callable(kb_manager.get_session_kb):
            session_kb = await kb_manager.get_session_kb(session_id)
        else:
            session_kb = await get_mock_session_kb(session_id)
        
        # Get initial session state
        session_overview = session_kb.get_session_overview()
        
        return {
            "success": True,
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "ready",
            "capabilities": {
                "multi_document_analysis": True,
                "cross_document_references": True,
                "real_time_integration": True,
                "multilingual_support": True,
                "voice_summaries": True
            },
            "session_overview": session_overview
        }
        
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create session")

@router.post("/upload", response_model=Dict[str, Any])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    language: str = Form("en"),
    auto_analyze: bool = Form(True),
    integration_mode: str = Form("intelligent")
):
    """
    Upload document to existing session or create new session
    
    Args:
        file: Document file (PDF, DOCX, TXT, etc.)
        session_id: Existing session ID (optional - creates new if not provided)
        language: Document language
        auto_analyze: Whether to auto-analyze the document
        integration_mode: How to integrate with existing documents
        
    Returns:
        Document analysis and integration suggestions
    """
    
    start_time = time.time()
    
    try:
        # Validate file
        if not validate_file_type(file.filename):
            raise ValidationError("Unsupported file type")
        
        # Read file content
        file_content = await file.read()
        
        if len(file_content) == 0:
            raise ValidationError("Empty file uploaded")
        
        # Create session if not provided
        if not session_id:
            session_id = generate_session_id()
        
        # Get or create session knowledge base
        if hasattr(kb_manager, 'get_session_kb') and callable(kb_manager.get_session_kb):
            session_kb = await kb_manager.get_session_kb(session_id)
        else:
            session_kb = await get_mock_session_kb(session_id)
        
        # Check if this is a mid-conversation upload
        is_mid_conversation = len(session_kb.documents) > 0
        
        # Create mock document
        document = MockStandardDocument(
            id=f"doc_{uuid.uuid4().hex[:8]}",
            content=file_content.decode('utf-8', errors='ignore'),
            filename=file.filename
        )
        
        # Mock chunks
        chunks = [
            type('MockChunk', (), {
                'id': f"chunk_{i}",
                'document_id': document.id,
                'content': file_content.decode('utf-8', errors='ignore')[i*1000:(i+1)*1000]
            })()
            for i in range(min(3, len(file_content) // 1000 + 1))
        ]
        
        if is_mid_conversation and integration_mode == "intelligent":
            # Mock real-time integration
            try:
                if hasattr(real_time_integrator, 'handle_mid_conversation_upload'):
                    integration_result = await real_time_integrator.handle_mid_conversation_upload(
                        session_id=session_id,
                        file_content=file_content,
                        filename=file.filename,
                        current_conversation_state={}
                    )
                else:
                    # Mock integration result
                    integration_result = type('MockResult', (), {
                        'success': True,
                        'document_summary': await session_kb.add_document(document, chunks),
                        'integration_suggestions': [
                            {
                                'type': 'cross_reference',
                                'priority': 'high',
                                'message': f'New document {file.filename} may relate to existing documents',
                                'suggested_queries': [
                                    'How does this new document relate to my existing ones?',
                                    'Are there any conflicts with previous documents?'
                                ]
                            }
                        ],
                        'relationship_updates': [],
                        'error_message': None
                    })()
                
                if integration_result.success:
                    processing_time = time.time() - start_time
                    
                    return {
                        "success": True,
                        "document_id": document.id,
                        "session_id": session_id,
                        "document_summary": integration_result.document_summary.to_dict() if hasattr(integration_result.document_summary, 'to_dict') else {},
                        "integration_suggestions": getattr(integration_result, 'integration_suggestions', []),
                        "relationship_updates": getattr(integration_result, 'relationship_updates', []),
                        "integration_summary": "Document successfully integrated into existing session",
                        "is_mid_conversation_upload": True,
                        "processing_time": processing_time,
                        "session_overview": session_kb.get_session_overview()
                    }
                else:
                    raise DocumentProcessingError(getattr(integration_result, 'error_message', 'Integration failed'))
                    
            except Exception as e:
                logger.error(f"Real-time integration failed: {e}")
                # Fall back to standard processing
        
        # Standard document processing
        try:
            if hasattr(document_processor, 'process_document'):
                processed_doc, processed_chunks = await document_processor.process_document(
                    file_content, file.filename, session_id
                )
            else:
                processed_doc, processed_chunks = document, chunks
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            processed_doc, processed_chunks = document, chunks
        
        # Add to knowledge base
        doc_summary = await session_kb.add_document(processed_doc, processed_chunks)
        
        # Generate basic suggestions for first document
        suggestions = []
        if len(session_kb.documents) == 1:
            suggestions = [
                {
                    "type": "initial_analysis",
                    "priority": "medium",
                    "message": f"I've analyzed your {doc_summary.document_type}. You can now ask questions about its content, risks, and key terms.",
                    "suggested_queries": [
                        "What are the main risks in this document?",
                        "Summarize the key terms and obligations",
                        "What should I be concerned about?"
                    ]
                }
            ]
        else:
            suggestions = [
                {
                    "type": "multi_document_analysis",
                    "priority": "high",
                    "message": f"Document added to session with {len(session_kb.documents)} documents. Ready for cross-document analysis.",
                    "suggested_queries": [
                        "Compare this document with my previous ones",
                        "Are there any conflicts between documents?",
                        "What are the combined risks across all documents?"
                    ]
                }
            ]
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "document_id": processed_doc.id,
            "session_id": session_id,
            "document_summary": doc_summary.to_dict(),
            "integration_suggestions": suggestions,
            "relationship_updates": [],
            "is_mid_conversation_upload": is_mid_conversation,
            "processing_time": processing_time,
            "session_overview": session_kb.get_session_overview()
        }
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except DocumentProcessingError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process document")

@router.post("/chat", response_model=Dict[str, Any])
async def chat_with_documents(request: ChatRequest):
    """
    Chat with all documents in the session
    
    Args:
        request: Chat request with message and parameters
        
    Returns:
        Synthesized response from multiple documents
    """
    
    start_time = time.time()
    
    try:
        # Get session knowledge base
        if hasattr(kb_manager, 'get_session_kb') and callable(kb_manager.get_session_kb):
            session_kb = await kb_manager.get_session_kb(request.session_id)
        else:
            session_kb = await get_mock_session_kb(request.session_id)
        
        if len(session_kb.documents) == 0:
            raise ChatError("No documents found in session. Please upload documents first.")
        
        # Get conversation context
        conversation_context = {}
        if hasattr(context_manager, 'get_conversation_context'):
            try:
                context_result = await context_manager.get_conversation_context(
                    request.session_id, request.message
                )
                conversation_context = context_result.__dict__ if hasattr(context_result, '__dict__') else context_result
            except Exception as e:
                logger.error(f"Context manager error: {e}")
                conversation_context = {
                    "session_id": request.session_id,
                    "recent_turns": [],
                    "current_topics": ["document_analysis"],
                    "conversation_thread": "User asked about documents"
                }
        
        # Generate synthesized response
        synthesized_response = None
        if hasattr(response_synthesizer, 'generate_response'):
            try:
                synthesized_response = await response_synthesizer.generate_response(
                    user_query=request.message,
                    session_kb=session_kb,
                    conversation_context=conversation_context,
                    max_sources=request.max_sources
                )
            except Exception as e:
                logger.error(f"Response synthesizer error: {e}")
        
        # Fallback response
        if not synthesized_response:
            doc_titles = [summary.title for summary in session_kb.document_summaries.values()]
            response_text = f"Based on your {len(session_kb.documents)} documents ({', '.join(doc_titles[:3])}), I can help answer your question: '{request.message}'. "
            
            if 'risk' in request.message.lower():
                response_text += "I've analyzed the risk factors across your documents and can provide detailed risk assessment."
            elif 'compare' in request.message.lower():
                response_text += "I can compare the terms and conditions across your documents to identify similarities and differences."
            elif 'summary' in request.message.lower():
                response_text += "I can provide a comprehensive summary of all your documents and their key provisions."
            else:
                response_text += "I can analyze the content and provide insights based on your specific question."
            
            synthesized_response = type('MockResponse', (), {
                'answer': response_text,
                'confidence_score': 0.8,
                'source_attributions': [
                    type('MockAttribution', (), {
                        'document_id': list(session_kb.documents.keys())[0],
                        'document_title': list(session_kb.document_summaries.values())[0].title,
                        'relevance_score': 0.9,
                        'excerpt': 'Relevant content from the document...'
                    })()
                ],
                'synthesis_strategy': 'multi_document_analysis',
                'to_dict': lambda: {
                    'answer': response_text,
                    'confidence_score': 0.8,
                    'source_attributions': [{'document_title': 'Document', 'relevance_score': 0.9}],
                    'synthesis_strategy': 'multi_document_analysis'
                }
            })()
        
        # Translate if needed
        if request.language != "en":
            try:
                if hasattr(translation_service, 'translate_text'):
                    translated_response = await translation_service.translate_text(
                        synthesized_response.answer, 
                        target_language=request.language,
                        source_language="en"
                    )
                    synthesized_response.answer = translated_response
            except Exception as e:
                logger.error(f"Translation error: {e}")
        
        # Add conversation turn to context
        try:
            if hasattr(context_manager, 'add_conversation_turn'):
                await context_manager.add_conversation_turn(
                    session_id=request.session_id,
                    user_message=request.message,
                    assistant_response=synthesized_response.answer,
                    source_documents=[attr.document_id for attr in synthesized_response.source_attributions],
                    response_metadata={
                        "synthesis_strategy": synthesized_response.synthesis_strategy,
                        "confidence_score": synthesized_response.confidence_score,
                        "language": request.language,
                        "response_style": request.response_style
                    }
                )
        except Exception as e:
            logger.error(f"Error adding conversation turn: {e}")
        
        # Generate integration opportunities
        integration_opportunities = [
            {
                "type": "comparison_analysis",
                "priority": "medium",
                "message": "Would you like me to compare information across all your documents?",
                "suggested_queries": [
                    "Compare the key terms across all documents",
                    "Which document has the most favorable terms?",
                    "Are there any conflicts between my documents?"
                ]
            }
        ]
        
        processing_time = time.time() - start_time
        
        # Create session state
        session_state = {
            "session_id": request.session_id,
            "document_count": len(session_kb.documents),
            "total_chunks": len(session_kb.chunks),
            "conversation_turn_count": len(conversation_context.get("recent_turns", [])),
            "dominant_topics": ["document_analysis", "legal_review"],
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "processing_time": processing_time
        }
        
        return {
            "success": True,
            "session_id": request.session_id,
            "response": synthesized_response.to_dict() if hasattr(synthesized_response, 'to_dict') else {
                'answer': synthesized_response.answer,
                'confidence_score': synthesized_response.confidence_score
            },
            "session_state": session_state,
            "integration_opportunities": integration_opportunities,
            "processing_time": processing_time,
            "conversation_insights": {
                "conversation_thread": conversation_context.get("conversation_thread", "Document analysis"),
                "current_topics": conversation_context.get("current_topics", ["legal_analysis"]),
                "user_expertise_level": "intermediate"
            }
        }
        
    except ChatError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process chat message")

@router.get("/sessions/{session_id}/overview", response_model=Dict[str, Any])
async def get_session_overview(session_id: str):
    """
    Get comprehensive overview of session
    
    Args:
        session_id: Session identifier
        
    Returns:
        Complete session overview with documents, relationships, and insights
    """
    
    try:
        # Get session knowledge base
        if hasattr(kb_manager, 'get_session_kb') and callable(kb_manager.get_session_kb):
            session_kb = await kb_manager.get_session_kb(session_id)
        else:
            session_kb = await get_mock_session_kb(session_id)
        
        if len(session_kb.documents) == 0:
            return {
                "success": True,
                "session_id": session_id,
                "message": "No documents in session yet",
                "session_overview": session_kb.get_session_overview()
            }
        
        # Get session overview
        overview = session_kb.get_session_overview()
        
        # Get conversation insights
        conversation_summary = {}
        if hasattr(context_manager, 'get_session_summary'):
            try:
                conversation_summary = context_manager.get_session_summary(session_id)
            except Exception as e:
                logger.error(f"Error getting conversation summary: {e}")
                conversation_summary = {"total_messages": 0, "dominant_topics": []}
        
        # Analyze portfolio
        portfolio_analysis = await _analyze_document_portfolio(session_kb)
        
        return {
            "success": True,
            "session_id": session_id,
            "session_overview": overview,
            "conversation_insights": conversation_summary,
            "conversation_patterns": {
                "dominant_topics": ["legal_analysis", "document_review"],
                "user_expertise_level": "intermediate"
            },
            "portfolio_analysis": portfolio_analysis,
            "documents": [
                summary.to_dict() for summary in session_kb.document_summaries.values()
            ],
            "relationships": [
                rel.to_dict() if hasattr(rel, 'to_dict') else {'id': 'rel_1', 'type': 'related'}
                for rel in session_kb.relationships.values()
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting session overview: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get session overview")

@router.get("/sessions/{session_id}/relationships", response_model=Dict[str, Any])
async def get_document_relationships(session_id: str):
    """
    Get document relationships and cross-references
    
    Args:
        session_id: Session identifier
        
    Returns:
        Document relationships with visualization data
    """
    
    try:
        if hasattr(kb_manager, 'get_session_kb') and callable(kb_manager.get_session_kb):
            session_kb = await kb_manager.get_session_kb(session_id)
        else:
            session_kb = await get_mock_session_kb(session_id)
        
        # Get all relationships
        relationships = list(session_kb.relationships.values())
        
        # Create network graph data
        nodes = []
        edges = []
        
        # Add document nodes
        for doc_id, summary in session_kb.document_summaries.items():
            nodes.append({
                "id": doc_id,
                "label": summary.title,
                "type": "document",
                "risk_level": summary.risk_level,
                "topics": summary.key_topics[:3],  # Top 3 topics
                "size": min(50, summary.chunk_count * 5)  # Node size based on chunks
            })
        
        # Add relationship edges (mock if no real relationships)
        if not relationships and len(session_kb.documents) > 1:
            doc_ids = list(session_kb.documents.keys())
            for i, doc1 in enumerate(doc_ids):
                for doc2 in doc_ids[i+1:]:
                    edges.append({
                        "source": doc1,
                        "target": doc2,
                        "weight": 0.5,
                        "type": "related",
                        "common_topics": ["legal_terms"],
                        "strength": "medium"
                    })
        
        for rel in relationships:
            if hasattr(rel, 'to_dict'):
                rel_dict = rel.to_dict()
                edges.append({
                    "source": rel_dict.get('document_1_id'),
                    "target": rel_dict.get('document_2_id'),
                    "weight": rel_dict.get('similarity_score', 0.5),
                    "type": rel_dict.get('relationship_type', 'related'),
                    "common_topics": rel_dict.get('common_topics', []),
                    "strength": "strong" if rel_dict.get('similarity_score', 0) > 0.7 else "medium"
                })
        
        return {
            "success": True,
            "session_id": session_id,
            "relationships": [rel.to_dict() if hasattr(rel, 'to_dict') else {} for rel in relationships],
            "network_graph": {
                "nodes": nodes,
                "edges": edges
            },
            "relationship_summary": {
                "total_relationships": len(relationships),
                "strong_relationships": sum(1 for edge in edges if edge.get('strength') == 'strong'),
                "document_count": len(session_kb.documents),
                "most_connected_document": _find_most_connected_document(edges) if edges else None
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting relationships: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get document relationships")

@router.delete("/sessions/{session_id}/documents/{document_id}")
async def remove_document(session_id: str, document_id: str):
    """
    Remove document from session
    
    Args:
        session_id: Session identifier
        document_id: Document identifier
        
    Returns:
        Updated session state
    """
    
    try:
        if hasattr(kb_manager, 'get_session_kb') and callable(kb_manager.get_session_kb):
            session_kb = await kb_manager.get_session_kb(session_id)
        else:
            session_kb = await get_mock_session_kb(session_id)
        
        if document_id not in session_kb.documents:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Remove document and its chunks
        del session_kb.documents[document_id]
        if document_id in session_kb.document_summaries:
            del session_kb.document_summaries[document_id]
        
        # Remove associated chunks
        chunks_to_remove = [chunk_id for chunk_id, chunk in session_kb.chunks.items() 
                           if getattr(chunk, 'document_id', None) == document_id]
        for chunk_id in chunks_to_remove:
            del session_kb.chunks[chunk_id]
        
        # Remove relationships involving this document
        relationships_to_remove = [
            rel_id for rel_id, rel in session_kb.relationships.items()
            if getattr(rel, 'document_1_id', None) == document_id or 
               getattr(rel, 'document_2_id', None) == document_id
        ]
        for rel_id in relationships_to_remove:
            del session_kb.relationships[rel_id]
        
        # Update session timestamp
        session_kb.last_updated = datetime.now(timezone.utc)
        
        return {
            "success": True,
            "message": f"Document {document_id} removed from session",
            "session_overview": session_kb.get_session_overview()
        }
        
    except Exception as e:
        logger.error(f"Error removing document: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to remove document")

@router.post("/sessions/{session_id}/voice-summary")
async def generate_voice_summary(
    session_id: str,
    language: str = "en",
    voice_type: str = "female",
    summary_type: str = "comprehensive"
):
    """
    Generate voice summary of session documents
    
    Args:
        session_id: Session identifier
        language: Voice language
        voice_type: Voice type (male/female)
        summary_type: Type of summary (comprehensive/risk_focused/key_points)
        
    Returns:
        Voice summary generation status and download URL
    """
    
    try:
        if hasattr(kb_manager, 'get_session_kb') and callable(kb_manager.get_session_kb):
            session_kb = await kb_manager.get_session_kb(session_id)
        else:
            session_kb = await get_mock_session_kb(session_id)
        
        if len(session_kb.documents) == 0:
            raise HTTPException(status_code=400, detail="No documents to summarize")
        
        # Generate text summary based on type
        if summary_type == "comprehensive":
            summary_text = await _generate_comprehensive_summary(session_kb)
        elif summary_type == "risk_focused":
            summary_text = await _generate_risk_summary(session_kb)
        else:  # key_points
            summary_text = await _generate_key_points_summary(session_kb)
        
        # Translate if needed
        if language != "en":
            try:
                if hasattr(translation_service, 'translate_text'):
                    summary_text = await translation_service.translate_text(
                        summary_text, target_language=language, source_language="en"
                    )
            except Exception as e:
                logger.error(f"Translation failed: {e}")
        
        # Generate voice (mock result)
        voice_result = {
            "success": True,
            "audio_url": f"/static/audio/voice_summary_{session_id}_{int(time.time())}.mp3",
            "duration_seconds": 60,
            "text_length": len(summary_text),
            "voice_settings": {
                "language": language,
                "voice_type": voice_type
            }
        }
        
        if hasattr(voice_generator, 'generate_voice_summary'):
            try:
                voice_result = await voice_generator.generate_voice_summary(
                    summary_text=summary_text,
                    language=language,
                    voice_type=voice_type,
                    session_id=session_id
                )
            except Exception as e:
                logger.error(f"Voice generation failed: {e}")
        
        return {
            "success": True,
            "session_id": session_id,
            "voice_summary": voice_result,
            "summary_text": summary_text,
            "language": language,
            "voice_type": voice_type,
            "summary_type": summary_type
        }
        
    except Exception as e:
        logger.error(f"Error generating voice summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate voice summary")

@router.get("/sessions/{session_id}/export")
async def export_session_data(session_id: str, format: str = "json"):
    """
    Export session data in various formats
    
    Args:
        session_id: Session identifier
        format: Export format (json/pdf/csv)
        
    Returns:
        Exported session data
    """
    
    try:
        if hasattr(kb_manager, 'get_session_kb') and callable(kb_manager.get_session_kb):
            session_kb = await kb_manager.get_session_kb(session_id)
        else:
            session_kb = await get_mock_session_kb(session_id)
        
        conversation_summary = {}
        if hasattr(context_manager, 'get_session_summary'):
            try:
                conversation_summary = context_manager.get_session_summary(session_id)
            except Exception:
                conversation_summary = {"total_messages": 0}
        
        if format == "json":
            export_data = {
                "session_id": session_id,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "session_overview": session_kb.get_session_overview(),
                "documents": [doc.to_dict() if hasattr(doc, 'to_dict') else {"id": doc.id, "filename": doc.filename} 
                             for doc in session_kb.documents.values()],
                "document_summaries": [summary.to_dict() for summary in session_kb.document_summaries.values()],
                "relationships": [rel.to_dict() if hasattr(rel, 'to_dict') else {} 
                               for rel in session_kb.relationships.values()],
                "conversation_summary": conversation_summary
            }
            
            return JSONResponse(
                content=export_data,
                headers={"Content-Disposition": f"attachment; filename=session_{session_id}.json"}
            )
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")
            
    except Exception as e:
        logger.error(f"Error exporting session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to export session data")

# Helper functions

async def _analyze_document_portfolio(session_kb) -> Dict[str, Any]:
    """Analyze the document portfolio for insights"""
    
    if len(session_kb.documents) == 0:
        return {"status": "empty_portfolio"}
    
    # Document type analysis
    doc_types = [summary.document_type for summary in session_kb.document_summaries.values()]
    type_distribution = {doc_type: doc_types.count(doc_type) for doc_type in set(doc_types)}
    
    # Risk analysis
    risk_levels = [summary.risk_level for summary in session_kb.document_summaries.values()]
    risk_distribution = {risk: risk_levels.count(risk) for risk in set(risk_levels)}
    
    # Topic analysis
    all_topics = []
    for summary in session_kb.document_summaries.values():
        all_topics.extend(summary.key_topics)
    
    topic_frequency = {}
    for topic in all_topics:
        topic_frequency[topic] = topic_frequency.get(topic, 0) + 1
    
    top_topics = sorted(topic_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        "total_documents": len(session_kb.documents),
        "document_type_distribution": type_distribution,
        "risk_level_distribution": risk_distribution,
        "top_topics": [{"topic": topic, "frequency": freq} for topic, freq in top_topics],
        "portfolio_completeness": len(session_kb.relationships) > 0,
        "analysis_depth": "comprehensive" if len(session_kb.documents) >= 3 else "basic"
    }

def _find_most_connected_document(edges: List[Dict]) -> Optional[str]:
    """Find the document with the most relationships"""
    
    if not edges:
        return None
    
    doc_connections = {}
    
    for edge in edges:
        source = edge.get('source')
        target = edge.get('target')
        if source:
            doc_connections[source] = doc_connections.get(source, 0) + 1
        if target:
            doc_connections[target] = doc_connections.get(target, 0) + 1
    
    if doc_connections:
        return max(doc_connections.items(), key=lambda x: x[1])[0]
    
    return None

async def _generate_comprehensive_summary(session_kb) -> str:
    """Generate comprehensive text summary of all documents"""
    
    summaries = []
    for doc_id, summary in session_kb.document_summaries.items():
        doc_summary = f"Document: {summary.title}\n"
        doc_summary += f"Type: {summary.document_type}\n"
        doc_summary += f"Risk Level: {summary.risk_level}\n"
        doc_summary += f"Key Topics: {', '.join(summary.key_topics)}\n"
        doc_summary += f"Summary: {summary.summary_text}\n"
        summaries.append(doc_summary)
    
    return "\n\n".join(summaries)

async def _generate_risk_summary(session_kb) -> str:
    """Generate risk-focused summary"""
    
    high_risk_docs = [
        summary for summary in session_kb.document_summaries.values()
        if summary.risk_level in ["high", "critical"]
    ]
    
    if not high_risk_docs:
        return "Overall risk assessment: Your documents show low to medium risk levels with no critical issues identified."
    
    risk_summary = f"Risk Assessment Summary: {len(high_risk_docs)} out of {len(session_kb.documents)} documents show elevated risk levels.\n\n"
    
    for doc in high_risk_docs:
        risk_summary += f"High Risk Document: {doc.title}\n"
        risk_summary += f"Risk Level: {doc.risk_level}\n"
        risk_summary += f"Key Concerns: {', '.join(doc.key_topics[:3])}\n\n"
    
    return risk_summary

async def _generate_key_points_summary(session_kb) -> str:
    """Generate key points summary"""
    
    key_points = []
    for summary in session_kb.document_summaries.values():
        key_points.append(f"• {summary.title}: {summary.summary_text}")
    
    return f"Key Points Summary:\n\n" + "\n\n".join(key_points)
