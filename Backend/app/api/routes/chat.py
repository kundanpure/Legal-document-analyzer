"""
Enhanced Chat API routes for interactive document conversations
Production-ready with advanced features, error handling, and monitoring
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional, List, Dict, Any
import asyncio
import json
from datetime import datetime, timezone

from config.logging import get_logger, log_api_request
from config.settings import get_settings
from app.models.requests import ChatRequest, ChatSessionRequest
from app.models.responses import (
    ChatResponse, ChatHistoryResponse, SuggestedQuestionsResponse,
    SessionResponse, ErrorResponse, StreamingChatResponse
)
from app.services.chat_handler import ChatHandler
from app.services.gemini_analyzer import GeminiAnalyzer
from app.utils.validators import validate_message_length, validate_session_id
from app.utils.helpers import generate_request_id, mask_sensitive_data
from app.core.exceptions import ChatError, ValidationError
from app.core.dependencies import get_current_user, rate_limit

settings = get_settings()
logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

# Service instances
chat_handler = ChatHandler()
gemini_analyzer = GeminiAnalyzer()

# External reference to documents (from documents.py)
from app.api.routes.documents import documents_db


@router.post("/message", response_model=ChatResponse)
async def send_chat_message(
    chat_request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict] = Depends(get_current_user),
    _rate_limit: None = Depends(rate_limit("chat_message"))
):
    """
    Send a message to the AI assistant about the document
    
    - **session_id**: Chat session identifier
    - **document_id**: Document identifier  
    - **message**: User's message/question
    - **language**: Response language preference
    - **stream**: Enable streaming response
    """
    
    request_id = generate_request_id()
    start_time = datetime.now(timezone.utc)
    
    try:
        logger.info(
            f"Chat message received",
            extra={
                'request_id': request_id,
                'session_id': chat_request.session_id,
                'document_id': chat_request.document_id,
                'user_id': current_user.get('user_id') if current_user else None,
                'message_length': len(chat_request.message),
                'language': chat_request.language
            }
        )
        
        # Enhanced validation
        validation_result = await _validate_chat_request(chat_request)
        if not validation_result['valid']:
            raise HTTPException(
                status_code=400,
                detail={
                    'error': 'VALIDATION_ERROR',
                    'message': validation_result['error'],
                    'request_id': request_id
                }
            )
        
        # Check document availability
        document_data = await _get_and_validate_document(chat_request.document_id)
        
        # Check session validity
        session_valid = await chat_handler.validate_session(chat_request.session_id)
        if not session_valid:
            raise HTTPException(
                status_code=404,
                detail={
                    'error': 'SESSION_NOT_FOUND',
                    'message': 'Chat session not found or expired',
                    'request_id': request_id
                }
            )
        
        # Rate limiting check for user
        user_id = current_user.get('user_id') if current_user else chat_request.session_id
        rate_limit_result = await chat_handler.check_user_rate_limit(user_id)
        if not rate_limit_result['allowed']:
            raise HTTPException(
                status_code=429,
                detail={
                    'error': 'RATE_LIMIT_EXCEEDED',
                    'message': f"Too many messages. Try again in {rate_limit_result['retry_after']} seconds",
                    'retry_after': rate_limit_result['retry_after'],
                    'request_id': request_id
                }
            )
        
        # Get AI response
        response_data = await chat_handler.get_ai_response(
            document_id=chat_request.document_id,
            message=chat_request.message,
            session_id=chat_request.session_id,
            language=chat_request.language,
            document_context=document_data,
            user_context=current_user,
            request_id=request_id
        )
        
        # Log successful interaction
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        background_tasks.add_task(
            log_api_request,
            'POST',
            '/chat/message',
            200,
            processing_time,
            user_id,
            chat_request.session_id,
            request_id
        )
        
        return ChatResponse(
            message_id=response_data['message_id'],
            response=response_data['response'],
            citations=response_data.get('citations', []),
            confidence=response_data.get('confidence', 0.8),
            suggestions=response_data.get('suggestions', []),
            timestamp=response_data.get('timestamp', datetime.now(timezone.utc).isoformat()),
            processing_time=processing_time,
            request_id=request_id,
            follow_up_questions=response_data.get('follow_up_questions', [])
        )
        
    except HTTPException:
        raise
    except ChatError as e:
        logger.error(f"Chat error: {str(e)}", extra={'request_id': request_id})
        raise HTTPException(
            status_code=400,
            detail={
                'error': 'CHAT_ERROR',
                'message': str(e),
                'request_id': request_id
            }
        )
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}", extra={'request_id': request_id})
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'INTERNAL_ERROR',
                'message': "Failed to process chat message",
                'request_id': request_id
            }
        )


@router.post("/stream", response_class=StreamingResponse)
async def send_streaming_chat_message(
    chat_request: ChatRequest,
    current_user: Optional[Dict] = Depends(get_current_user),
    _rate_limit: None = Depends(rate_limit("chat_message"))
):
    """
    Send a message with streaming response for real-time chat experience
    """
    
    request_id = generate_request_id()
    
    try:
        # Validate request
        validation_result = await _validate_chat_request(chat_request)
        if not validation_result['valid']:
            # Return error as streaming response
            async def error_stream():
                error_data = {
                    'error': 'VALIDATION_ERROR',
                    'message': validation_result['error'],
                    'request_id': request_id
                }
                yield f"data: {json.dumps(error_data)}\n\n"
            
            return StreamingResponse(
                error_stream(),
                media_type="text/plain",
                status_code=400
            )
        
        # Check document and session
        document_data = await _get_and_validate_document(chat_request.document_id)
        session_valid = await chat_handler.validate_session(chat_request.session_id)
        
        if not session_valid:
            async def error_stream():
                error_data = {
                    'error': 'SESSION_NOT_FOUND',
                    'message': 'Chat session not found or expired',
                    'request_id': request_id
                }
                yield f"data: {json.dumps(error_data)}\n\n"
            
            return StreamingResponse(
                error_stream(),
                media_type="text/plain",
                status_code=404
            )
        
        # Create streaming response
        async def generate_streaming_response():
            async for chunk in chat_handler.get_streaming_ai_response(
                document_id=chat_request.document_id,
                message=chat_request.message,
                session_id=chat_request.session_id,
                language=chat_request.language,
                document_context=document_data,
                request_id=request_id
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'done', 'request_id': request_id})}\n\n"
        
        return StreamingResponse(
            generate_streaming_response(),
            media_type="text/event-stream",
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Request-ID': request_id
            }
        )
        
    except Exception as e:
        logger.error(f"Error in streaming chat: {str(e)}", extra={'request_id': request_id})
        
        async def error_stream():
            error_data = {
                'error': 'STREAMING_ERROR',
                'message': 'Failed to start streaming response',
                'request_id': request_id
            }
            yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(
            error_stream(),
            media_type="text/plain",
            status_code=500
        )


@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_id: str,
    limit: int = 50,
    offset: int = 0,
    include_metadata: bool = False,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Get conversation history for a chat session with enhanced filtering
    """
    
    try:
        # Validate session access
        if not await chat_handler.user_has_session_access(session_id, current_user):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this chat session"
            )
        
        history_data = await chat_handler.get_session_history(
            session_id=session_id,
            limit=min(limit, 100),  # Cap at 100
            offset=max(offset, 0),
            include_metadata=include_metadata,
            user_context=current_user
        )
        
        return ChatHistoryResponse(
            session_id=session_id,
            messages=history_data['messages'],
            total_messages=history_data['total_count'],
            has_more=history_data.get('has_more', False),
            session_info=history_data.get('session_info', {}),
            pagination={
                'limit': limit,
                'offset': offset,
                'total': history_data['total_count']
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get chat history"
        )


@router.get("/suggested-questions/{document_id}", response_model=SuggestedQuestionsResponse)
async def get_suggested_questions(
    document_id: str,
    language: str = "en",
    category: Optional[str] = None,
    difficulty: Optional[str] = "mixed",
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Get AI-generated suggested questions for a document with categorization
    """
    
    try:
        document_data = await _get_and_validate_document(document_id)
        
        # Check user access to document
        if not await _check_document_access(document_id, current_user):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this document"
            )
        
        # Generate suggested questions with enhanced context
        questions_data = await gemini_analyzer.generate_suggested_questions(
            document_type=document_data.get('document_type', 'general'),
            document_summary=document_data.get('summary', ''),
            risk_level=document_data.get('risk_level', 'medium'),
            key_topics=document_data.get('key_topics', []),
            language=language,
            category_filter=category,
            difficulty=difficulty,
            user_context=current_user
        )
        
        return SuggestedQuestionsResponse(
            questions=questions_data['questions'],
            document_type=document_data.get('document_type', 'unknown'),
            categories=questions_data.get('categories', {}),
            total_questions=len(questions_data['questions']),
            metadata={
                'generation_method': questions_data.get('method', 'ai'),
                'language': language,
                'difficulty': difficulty,
                'personalized': bool(current_user)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating suggested questions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate suggested questions"
        )


@router.post("/session", response_model=SessionResponse)
async def create_chat_session(
    session_request: ChatSessionRequest,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Create a new chat session for a document with enhanced configuration
    """
    
    try:
        document_data = await _get_and_validate_document(session_request.document_id)
        
        # Check document access
        if not await _check_document_access(session_request.document_id, current_user):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this document"
            )
        
        # Create session with user context
        session_data = await chat_handler.initialize_chat_session(
            document_id=session_request.document_id,
            document_data=document_data,
            language=session_request.language,
            user_context=current_user,
            session_config=session_request.config
        )
        
        return SessionResponse(
            session_id=session_data['session_id'],
            user_id=current_user.get('user_id', 'anonymous'),
            document_id=session_request.document_id,
            created_at=session_data.get('created_at'),
            expires_at=session_data.get('session_expires_at'),
            preferences={
                "language": session_request.language,
                "document_id": session_request.document_id,
                **session_request.config
            },
            capabilities=session_data.get('capabilities', []),
            welcome_message=session_data.get('welcome_message')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating chat session: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create chat session"
        )


@router.delete("/session/{session_id}")
async def end_chat_session(
    session_id: str,
    current_user: Optional[Dict] = Depends(get_current_user),
    export_history: bool = False
):
    """
    End a chat session with optional history export
    """
    
    try:
        # Validate session access
        if not await chat_handler.user_has_session_access(session_id, current_user):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this chat session"
            )
        
        # End session and get summary
        session_summary = await chat_handler.end_chat_session(
            session_id=session_id,
            user_context=current_user,
            export_history=export_history
        )
        
        response_data = {
            "success": True,
            "message": "Chat session ended successfully",
            "session_summary": {
                'message_count': session_summary.get('message_count', 0),
                'duration': session_summary.get('duration_minutes', 0),
                'topics_discussed': session_summary.get('topics', []),
                'ended_at': datetime.now(timezone.utc).isoformat()
            }
        }
        
        if export_history and session_summary.get('exported_history'):
            response_data['exported_history_url'] = session_summary['exported_history']
        
        return response_data
        
    except HTTPException:
        raise
    except ChatError as e:
        logger.error(f"Chat error ending session: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error ending chat session: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to end chat session"
        )


@router.put("/session/{session_id}/preferences")
async def update_session_preferences(
    session_id: str,
    preferences: Dict[str, Any],
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Update chat session preferences with validation
    """
    
    try:
        # Validate session access
        if not await chat_handler.user_has_session_access(session_id, current_user):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this chat session"
            )
        
        # Validate preferences
        valid_preferences = await _validate_preferences(preferences)
        
        success = await chat_handler.update_session_preferences(
            session_id=session_id,
            preferences=valid_preferences,
            user_context=current_user
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Session not found or cannot be updated"
            )
        
        return {
            "success": True,
            "message": "Preferences updated successfully",
            "updated_preferences": valid_preferences
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating session preferences: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to update preferences"
        )


@router.get("/sessions")
async def get_user_chat_sessions(
    current_user: Optional[Dict] = Depends(get_current_user),
    active_only: bool = False,
    limit: int = 20
):
    """
    Get user's chat sessions with filtering options
    """
    
    try:
        user_id = current_user.get('user_id', 'anonymous') if current_user else 'anonymous'
        
        sessions = await chat_handler.get_user_sessions(
            user_id=user_id,
            active_only=active_only,
            limit=min(limit, 50)  # Cap at 50
        )
        
        return {
            "sessions": sessions,
            "total_count": len(sessions),
            "active_count": len([s for s in sessions if s.get('status') == 'active'])
        }
        
    except Exception as e:
        logger.error(f"Error getting user sessions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get chat sessions"
        )


@router.get("/stats")
async def get_chat_statistics(
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Get chat system statistics (admin/user endpoint)
    """
    
    try:
        if current_user and current_user.get('role') == 'admin':
            # Admin gets full stats
            stats = await chat_handler.get_system_stats()
        else:
            # Users get limited stats
            user_id = current_user.get('user_id') if current_user else 'anonymous'
            stats = await chat_handler.get_user_stats(user_id)
        
        return {
            "success": True,
            "statistics": stats,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting chat statistics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get chat statistics"
        )


@router.post("/feedback/{message_id}")
async def submit_message_feedback(
    message_id: str,
    feedback: Dict[str, Any],
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Submit feedback for a chat message (thumbs up/down, quality rating)
    """
    
    try:
        # Validate feedback
        if 'rating' not in feedback or feedback['rating'] not in [-1, 0, 1]:
            raise HTTPException(
                status_code=400,
                detail="Invalid feedback rating. Must be -1, 0, or 1"
            )
        
        success = await chat_handler.submit_feedback(
            message_id=message_id,
            feedback=feedback,
            user_context=current_user
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Message not found or feedback already submitted"
            )
        
        return {
            "success": True,
            "message": "Feedback submitted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to submit feedback"
        )


# Admin endpoints
@router.post("/admin/cleanup-expired-sessions")
async def cleanup_expired_sessions(
    current_user: Dict = Depends(get_current_user)
):
    """
    Cleanup expired chat sessions (admin endpoint)
    """
    
    try:
        # Check admin permission
        if not current_user or current_user.get('role') != 'admin':
            raise HTTPException(
                status_code=403,
                detail="Admin access required"
            )
        
        cleanup_result = await chat_handler.cleanup_expired_sessions()
        
        return {
            "success": True,
            "message": f"Cleaned up {cleanup_result['cleaned_count']} expired sessions",
            "details": cleanup_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cleaning up sessions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to cleanup sessions"
        )


# Helper functions
async def _validate_chat_request(chat_request: ChatRequest) -> Dict[str, Any]:
    """Validate chat request with comprehensive checks"""
    
    # Message length validation
    if not validate_message_length(chat_request.message, min_length=1, max_length=2000):
        return {'valid': False, 'error': "Message must be between 1 and 2000 characters"}
    
    # Session ID validation
    if not validate_session_id(chat_request.session_id):
        return {'valid': False, 'error': "Invalid session ID format"}
    
    # Language validation
    supported_languages = settings.voice.supported_languages
    if chat_request.language not in supported_languages:
        return {'valid': False, 'error': f"Language '{chat_request.language}' not supported"}
    
    # Content filtering (basic)
    if await _contains_inappropriate_content(chat_request.message):
        return {'valid': False, 'error': "Message contains inappropriate content"}
    
    return {'valid': True}


async def _get_and_validate_document(document_id: str) -> Dict[str, Any]:
    """Get and validate document exists and is ready"""
    
    if document_id not in documents_db:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )
    
    document_data = documents_db[document_id]
    
    if document_data['status'] != 'completed':
        raise HTTPException(
            status_code=400,
            detail=f"Document is not ready for chat. Status: {document_data['status']}"
        )
    
    return document_data


async def _check_document_access(document_id: str, user_context: Optional[Dict]) -> bool:
    """Check if user has access to document"""
    # In production, implement proper access control
    return True


async def _contains_inappropriate_content(message: str) -> bool:
    """Basic content filtering"""
    # Implement content filtering logic
    inappropriate_words = ['spam', 'scam']  # Basic example
    return any(word in message.lower() for word in inappropriate_words)


async def _validate_preferences(preferences: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize preferences"""
    
    valid_keys = {
        'language', 'response_style', 'detail_level', 'include_citations',
        'suggest_follow_ups', 'max_response_length'
    }
    
    # Filter out invalid keys
    valid_preferences = {k: v for k, v in preferences.items() if k in valid_keys}
    
    # Validate specific preferences
    if 'language' in valid_preferences:
        if valid_preferences['language'] not in settings.voice.supported_languages:
            valid_preferences['language'] = 'en'
    
    if 'detail_level' in valid_preferences:
        if valid_preferences['detail_level'] not in ['brief', 'moderate', 'detailed']:
            valid_preferences['detail_level'] = 'moderate'
    
    return valid_preferences
