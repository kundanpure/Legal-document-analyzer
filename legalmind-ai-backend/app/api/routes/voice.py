"""
Enhanced Voice summary generation API routes
Production-ready with advanced features, caching, and multi-language support
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional, List, Dict, Any
import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

from config.logging import get_logger, log_api_request
from config.settings import get_settings
from app.models.requests import VoiceRequest, MultilingualVoiceRequest
from app.models.responses import (
    VoiceGenerationResponse, VoiceSummaryResponse, VoiceLibraryResponse,
    VoiceProgressResponse, ErrorResponse
)
from app.services.voice_generator import VoiceGenerator
from app.services.translation_service import TranslationService
from app.services.storage_manager import StorageManager
from app.utils.validators import validate_voice_request, validate_text_length
from app.utils.helpers import generate_request_id, sanitize_filename
from app.utils.formatters import format_file_size, format_duration
from app.core.exceptions import VoiceGenerationError, ValidationError
from app.core.dependencies import get_current_user, rate_limit

settings = get_settings()
logger = get_logger(__name__)
router = APIRouter(prefix="/voice", tags=["voice"])

# Service instances
voice_generator = VoiceGenerator()
translation_service = TranslationService()
storage_manager = StorageManager()

# External reference to documents
from app.api.routes.documents import documents_db

# Enhanced voice storage with metadata
voice_summaries_db = {}
voice_generation_status = {}
voice_analytics = {'total_generated': 0, 'total_duration': 0, 'languages_used': {}}


@router.post("/generate", response_model=VoiceGenerationResponse)
async def generate_voice_summary(
    background_tasks: BackgroundTasks,
    voice_request: VoiceRequest,
    current_user: Optional[Dict] = Depends(get_current_user),
    _rate_limit: None = Depends(rate_limit("voice_generation"))
):
    """
    Generate voice summary of document analysis with advanced features
    
    - **document_id**: Document identifier
    - **language**: Voice language (en, hi, ta, te, etc.)
    - **voice_type**: Voice type (male, female, neutral)
    - **speed**: Speech speed (0.5 - 2.0)
    - **content_type**: Type of content to vocalize (summary, risks, recommendations, full)
    """
    
    request_id = generate_request_id()
    start_time = datetime.now(timezone.utc)
    
    try:
        logger.info(
            f"Voice generation request",
            extra={
                'request_id': request_id,
                'document_id': voice_request.document_id,
                'language': voice_request.language,
                'voice_type': voice_request.voice_type,
                'user_id': current_user.get('user_id') if current_user else None
            }
        )
        
        # Enhanced validation
        validation_result = await _validate_voice_request_enhanced(voice_request, current_user)
        if not validation_result['valid']:
            raise HTTPException(
                status_code=400,
                detail={
                    'error': 'VALIDATION_FAILED',
                    'message': validation_result['error'],
                    'suggestions': validation_result.get('suggestions', []),
                    'request_id': request_id
                }
            )
        
        # Check document availability and access
        document_data = await _get_and_validate_document_for_voice(voice_request.document_id, current_user)
        
        # Check voice generation limits
        user_id = current_user.get('user_id') if current_user else 'anonymous'
        limit_check = await _check_voice_generation_limits(user_id)
        if not limit_check['allowed']:
            raise HTTPException(
                status_code=429,
                detail={
                    'error': 'VOICE_LIMIT_EXCEEDED',
                    'message': limit_check['message'],
                    'limit_info': limit_check['limit_info'],
                    'request_id': request_id
                }
            )
        
        # Generate unique voice ID
        voice_id = str(uuid.uuid4())
        
        # Prepare content for voice generation
        content_to_vocalize = await _prepare_voice_content(
            document_data, 
            voice_request.content_type,
            voice_request.max_length
        )
        
        # Estimate generation time and cost
        estimated_time = _estimate_voice_generation_time(
            len(content_to_vocalize.split()),
            voice_request.language,
            voice_request.voice_type
        )
        
        # Store voice info with enhanced metadata
        voice_info = {
            'voice_id': voice_id,
            'document_id': voice_request.document_id,
            'language': voice_request.language,
            'voice_type': voice_request.voice_type,
            'speed': voice_request.speed,
            'content_type': voice_request.content_type,
            'status': 'initializing',
            'created_at': datetime.now(timezone.utc),
            'progress': 0,
            'user_id': user_id,
            'request_id': request_id,
            'content_length': len(content_to_vocalize),
            'estimated_duration': estimated_time,
            'priority': _calculate_voice_priority(current_user),
            'metadata': {
                'document_title': document_data.get('title', document_data.get('filename')),
                'document_type': document_data.get('document_type'),
                'generation_method': 'ai',
                'version': '2.0'
            }
        }
        
        voice_summaries_db[voice_id] = voice_info
        voice_generation_status[voice_id] = {
            'stage': 'initializing', 
            'progress': 0,
            'started_at': datetime.now(timezone.utc)
        }
        
        # Start background voice generation
        background_tasks.add_task(
            generate_voice_workflow,
            voice_id,
            document_data,
            voice_request,
            content_to_vocalize,
            request_id
        )
        
        # Log API request
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        background_tasks.add_task(
            log_api_request,
            'POST',
            '/voice/generate',
            200,
            processing_time,
            user_id,
            voice_request.document_id,
            request_id
        )
        
        return VoiceGenerationResponse(
            voice_id=voice_id,
            status="initializing",
            estimated_time=format_duration(estimated_time),
            estimated_duration=f"{estimated_time // 60}:{estimated_time % 60:02d}",
            content_info={
                'type': voice_request.content_type,
                'length': len(content_to_vocalize),
                'words': len(content_to_vocalize.split())
            },
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except VoiceGenerationError as e:
        logger.error(f"Voice generation error: {str(e)}", extra={'request_id': request_id})
        raise HTTPException(
            status_code=400,
            detail={
                'error': 'VOICE_GENERATION_ERROR',
                'message': str(e),
                'request_id': request_id
            }
        )
    except Exception as e:
        logger.error(f"Error initiating voice generation: {str(e)}", extra={'request_id': request_id})
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'INTERNAL_ERROR',
                'message': "Failed to initiate voice generation",
                'request_id': request_id
            }
        )


@router.get("/{voice_id}/status", response_model=VoiceProgressResponse)
async def get_voice_generation_status(
    voice_id: str,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Get detailed voice generation status with real-time progress
    """
    
    try:
        if voice_id not in voice_summaries_db:
            raise HTTPException(
                status_code=404,
                detail="Voice generation not found"
            )
        
        voice_info = voice_summaries_db[voice_id]
        status_info = voice_generation_status.get(voice_id, {})
        
        # Check user access
        if not await _check_voice_access(voice_id, current_user):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this voice generation"
            )
        
        # Calculate processing time
        processing_time = None
        if status_info.get('started_at'):
            if voice_info['status'] == 'completed':
                processing_time = (voice_info.get('completed_at', datetime.now(timezone.utc)) - status_info['started_at']).total_seconds()
            else:
                processing_time = (datetime.now(timezone.utc) - status_info['started_at']).total_seconds()
        
        return VoiceProgressResponse(
            voice_id=voice_id,
            status=voice_info['status'],
            progress=status_info.get('progress', 0),
            current_stage=status_info.get('stage'),
            stage_description=status_info.get('description', ''),
            processing_time=format_duration(processing_time) if processing_time else None,
            estimated_remaining=status_info.get('estimated_remaining'),
            error_message=voice_info.get('error_message'),
            audio_preview_available=voice_info.get('status') == 'completed',
            metadata=voice_info.get('metadata', {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting voice status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get voice generation status"
        )


@router.get("/{voice_id}", response_model=VoiceSummaryResponse)
async def get_voice_summary(
    voice_id: str,
    include_transcript: bool = Query(True, description="Include full transcript"),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Get voice summary details and download information
    """
    
    try:
        if voice_id not in voice_summaries_db:
            raise HTTPException(
                status_code=404,
                detail="Voice summary not found"
            )
        
        # Check access
        if not await _check_voice_access(voice_id, current_user):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this voice summary"
            )
        
        voice_info = voice_summaries_db[voice_id]
        
        # Build response
        response_data = {
            'voice_id': voice_id,
            'status': voice_info['status'],
            'audio_url': voice_info.get('audio_url'),
            'download_url': voice_info.get('download_url'),
            'duration': voice_info.get('duration'),
            'language': voice_info['language'],
            'voice_type': voice_info['voice_type'],
            'content_type': voice_info['content_type'],
            'file_size': voice_info.get('file_size'),
            'file_size_formatted': format_file_size(voice_info.get('file_size', 0)),
            'created_at': voice_info['created_at'].isoformat(),
            'completed_at': voice_info.get('completed_at'),
            'expires_at': voice_info.get('expires_at'),
            'metadata': voice_info.get('metadata', {})
        }
        
        # Add transcript if requested and available
        if include_transcript and voice_info.get('transcript'):
            response_data['transcript'] = voice_info['transcript']
        
        return VoiceSummaryResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting voice summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get voice summary"
        )


@router.get("/{voice_id}/download")
async def download_voice_file(
    voice_id: str,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Download voice audio file with streaming support
    """
    
    try:
        if voice_id not in voice_summaries_db:
            raise HTTPException(
                status_code=404,
                detail="Voice summary not found"
            )
        
        # Check access
        if not await _check_voice_access(voice_id, current_user):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this voice file"
            )
        
        voice_info = voice_summaries_db[voice_id]
        
        if voice_info['status'] != 'completed':
            raise HTTPException(
                status_code=400,
                detail=f"Voice file not ready. Status: {voice_info['status']}"
            )
        
        # Get audio file from storage
        audio_content = await voice_generator.get_audio_file(voice_id)
        
        if not audio_content:
            raise HTTPException(
                status_code=404,
                detail="Audio file not found in storage"
            )
        
        # Generate safe filename
        safe_filename = sanitize_filename(
            f"{voice_info['metadata'].get('document_title', 'document')}_{voice_info['language']}_voice.mp3"
        )
        
        # Return streaming response
        def generate_audio_stream():
            chunk_size = 8192
            for i in range(0, len(audio_content), chunk_size):
                yield audio_content[i:i + chunk_size]
        
        return StreamingResponse(
            generate_audio_stream(),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"attachment; filename={safe_filename}",
                "Content-Length": str(len(audio_content))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading voice file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to download voice file"
        )


@router.get("/", response_model=VoiceLibraryResponse)
async def get_user_voice_summaries(
    current_user: Optional[Dict] = Depends(get_current_user),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=50, description="Items per page"),
    document_id: Optional[str] = Query(None, description="Filter by document"),
    language: Optional[str] = Query(None, description="Filter by language"),
    status: Optional[str] = Query(None, description="Filter by status"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order")
):
    """
    Get user's voice summaries with advanced filtering and pagination
    """
    
    try:
        user_id = current_user.get('user_id', 'anonymous') if current_user else 'anonymous'
        
        # Filter voice summaries
        filtered_voices = []
        
        for voice_id, voice_info in voice_summaries_db.items():
            # User access check
            if not await _check_voice_access(voice_id, current_user):
                continue
            
            # Apply filters
            if document_id and voice_info.get('document_id') != document_id:
                continue
                
            if language and voice_info.get('language') != language:
                continue
                
            if status and voice_info.get('status') != status:
                continue
            
            # Create voice summary
            voice_summary = {
                'voice_id': voice_id,
                'document_id': voice_info.get('document_id'),
                'document_title': voice_info.get('metadata', {}).get('document_title', ''),
                'language': voice_info.get('language'),
                'voice_type': voice_info.get('voice_type'),
                'content_type': voice_info.get('content_type'),
                'status': voice_info.get('status'),
                'created_at': voice_info.get('created_at'),
                'completed_at': voice_info.get('completed_at'),
                'duration': voice_info.get('duration'),
                'file_size': voice_info.get('file_size'),
                'file_size_formatted': format_file_size(voice_info.get('file_size', 0)),
                'download_available': voice_info.get('status') == 'completed',
                'expires_at': voice_info.get('expires_at')
            }
            
            filtered_voices.append(voice_summary)
        
        # Sort voices
        reverse_sort = sort_order.lower() == 'desc'
        
        if sort_by == 'created_at':
            filtered_voices.sort(key=lambda x: x['created_at'] or datetime.min, reverse=reverse_sort)
        elif sort_by == 'duration':
            filtered_voices.sort(key=lambda x: x['duration'] or 0, reverse=reverse_sort)
        elif sort_by == 'language':
            filtered_voices.sort(key=lambda x: x['language'], reverse=reverse_sort)
        
        # Apply pagination
        total_count = len(filtered_voices)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_voices = filtered_voices[start_idx:end_idx]
        
        # Calculate statistics
        stats = {
            'total_voices': total_count,
            'by_language': {},
            'by_status': {},
            'total_duration': 0,
            'total_size': 0
        }
        
        for voice in filtered_voices:
            lang = voice['language']
            status = voice['status']
            stats['by_language'][lang] = stats['by_language'].get(lang, 0) + 1
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
            stats['total_duration'] += voice['duration'] or 0
            stats['total_size'] += voice['file_size'] or 0
        
        return VoiceLibraryResponse(
            voice_summaries=paginated_voices,
            total_count=total_count,
            page=page,
            per_page=per_page,
            total_pages=(total_count + per_page - 1) // per_page,
            has_next=end_idx < total_count,
            has_prev=page > 1,
            statistics=stats
        )
        
    except Exception as e:
        logger.error(f"Error getting voice summaries: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get voice summaries"
        )


@router.post("/multilingual", response_model=Dict[str, Any])
async def generate_multilingual_summary(
    background_tasks: BackgroundTasks,
    multilingual_request: MultilingualVoiceRequest,
    current_user: Optional[Dict] = Depends(get_current_user),
    _rate_limit: None = Depends(rate_limit("voice_generation"))
):
    """
    Generate voice summaries in multiple languages simultaneously
    """
    
    request_id = generate_request_id()
    
    try:
        # Validate document exists
        if multilingual_request.document_id not in documents_db:
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )
        
        document_data = documents_db[multilingual_request.document_id]
        
        if document_data['status'] != 'completed':
            raise HTTPException(
                status_code=400,
                detail="Document not ready for voice generation"
            )
        
        # Check access
        if not await _check_document_access_for_voice(multilingual_request.document_id, current_user):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this document"
            )
        
        # Validate languages
        supported_languages = voice_generator.get_supported_languages()
        invalid_languages = [lang for lang in multilingual_request.languages if lang not in supported_languages]
        
        if invalid_languages:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported languages: {', '.join(invalid_languages)}"
            )
        
        # Start multilingual generation
        background_tasks.add_task(
            generate_multilingual_workflow,
            multilingual_request.document_id,
            document_data,
            multilingual_request.languages,
            multilingual_request.voice_type,
            multilingual_request.content_type,
            current_user,
            request_id
        )
        
        return {
            "success": True,
            "message": f"Multilingual voice generation started for {len(multilingual_request.languages)} languages",
            "languages": multilingual_request.languages,
            "estimated_time": format_duration(len(multilingual_request.languages) * 45),
            "request_id": request_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting multilingual generation: {str(e)}", extra={'request_id': request_id})
        raise HTTPException(
            status_code=500,
            detail="Failed to start multilingual generation"
        )


@router.delete("/{voice_id}")
async def delete_voice_summary(
    voice_id: str,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Delete a voice summary and its audio file
    """
    
    try:
        if voice_id not in voice_summaries_db:
            raise HTTPException(
                status_code=404,
                detail="Voice summary not found"
            )
        
        # Check access
        if not await _check_voice_access(voice_id, current_user):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this voice summary"
            )
        
        voice_info = voice_summaries_db[voice_id]
        
        # Delete audio file from storage
        try:
            await voice_generator.delete_audio_file(voice_id)
        except Exception as e:
            logger.warning(f"Failed to delete audio file: {str(e)}")
        
        # Remove from local storage
        del voice_summaries_db[voice_id]
        if voice_id in voice_generation_status:
            del voice_generation_status[voice_id]
        
        logger.info(f"Voice summary deleted", extra={'voice_id': voice_id})
        
        return {
            "success": True,
            "message": "Voice summary deleted successfully",
            "deleted_at": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting voice summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete voice summary"
        )


@router.get("/supported/languages")
async def get_supported_voice_languages():
    """
    Get supported languages for voice generation with voice options
    """
    
    try:
        languages_info = await voice_generator.get_supported_languages_detailed()
        
        return {
            "success": True,
            "supported_languages": languages_info,
            "total_languages": len(languages_info),
            "features": {
                "neural_voices": True,
                "custom_speed": True,
                "multiple_voices": True,
                "audio_formats": ["mp3", "wav"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting supported languages: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get supported languages"
        )


@router.get("/analytics/overview")
async def get_voice_analytics(
    current_user: Optional[Dict] = Depends(get_current_user),
    time_range: str = Query("7d", description="Time range (1d, 7d, 30d)")
):
    """
    Get voice generation analytics
    """
    
    try:
        user_id = current_user.get('user_id') if current_user else 'anonymous'
        
        # Calculate time range
        now = datetime.now(timezone.utc)
        if time_range == "1d":
            start_date = now - timedelta(days=1)
        elif time_range == "7d":
            start_date = now - timedelta(days=7)
        elif time_range == "30d":
            start_date = now - timedelta(days=30)
        else:
            start_date = now - timedelta(days=7)
        
        # Filter voices by time range and user
        user_voices = []
        for voice_id, voice_info in voice_summaries_db.items():
            if await _check_voice_access(voice_id, current_user):
                if voice_info.get('created_at', datetime.min.replace(tzinfo=timezone.utc)) >= start_date:
                    user_voices.append(voice_info)
        
        # Calculate analytics
        analytics = {
            'time_range': time_range,
            'total_voices': len(user_voices),
            'by_language': {},
            'by_status': {},
            'by_content_type': {},
            'generation_stats': {
                'total_duration': 0,
                'avg_duration': 0,
                'total_size': 0,
                'success_rate': 0
            }
        }
        
        total_duration = 0
        total_size = 0
        completed_count = 0
        
        for voice in user_voices:
            # Language distribution
            lang = voice.get('language', 'unknown')
            analytics['by_language'][lang] = analytics['by_language'].get(lang, 0) + 1
            
            # Status distribution
            status = voice.get('status', 'unknown')
            analytics['by_status'][status] = analytics['by_status'].get(status, 0) + 1
            
            # Content type distribution
            content_type = voice.get('content_type', 'summary')
            analytics['by_content_type'][content_type] = analytics['by_content_type'].get(content_type, 0) + 1
            
            # Duration and size stats
            if status == 'completed':
                completed_count += 1
                duration = voice.get('duration', 0)
                size = voice.get('file_size', 0)
                total_duration += duration
                total_size += size
        
        # Calculate generation stats
        if user_voices:
            analytics['generation_stats'] = {
                'total_duration': total_duration,
                'avg_duration': total_duration / completed_count if completed_count > 0 else 0,
                'total_size': total_size,
                'total_size_formatted': format_file_size(total_size),
                'success_rate': (completed_count / len(user_voices)) * 100
            }
        
        return {
            'success': True,
            'analytics': analytics,
            'generated_at': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting voice analytics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get voice analytics"
        )


# Background workflows

async def generate_voice_workflow(
    voice_id: str,
    document_data: Dict[str, Any],
    voice_request: VoiceRequest,
    content_to_vocalize: str,
    request_id: str
):
    """
    Enhanced voice generation workflow with detailed tracking
    """
    
    start_time = datetime.now(timezone.utc)
    
    try:
        logger.info(f"Starting voice generation workflow", extra={
            'voice_id': voice_id,
            'request_id': request_id,
            'language': voice_request.language
        })
        
        # Stage 1: Content Preparation
        await _update_voice_status(voice_id, 'preparing', 20, 'Preparing content for voice synthesis')
        
        # Translate content if needed
        if voice_request.language != 'en':
            await _update_voice_status(voice_id, 'translating', 40, 'Translating content to target language')
            
            translation_result = await translation_service.translate_text(
                content_to_vocalize,
                target_language=voice_request.language,
                source_language='en'
            )
            content_to_vocalize = translation_result.get('translated_text', content_to_vocalize)
        
        # Stage 2: Voice Synthesis
        await _update_voice_status(voice_id, 'synthesizing', 70, 'Generating voice audio')
        
        voice_result = await voice_generator.generate_voice_summary(
            text_content=content_to_vocalize,
            language=voice_request.language,
            voice_type=voice_request.voice_type,
            speed=voice_request.speed,
            voice_id=voice_id
        )
        
        # Stage 3: Post-processing and Storage
        await _update_voice_status(voice_id, 'finalizing', 90, 'Finalizing audio file')
        
        # Calculate expiration date
        expires_at = datetime.now(timezone.utc) + timedelta(days=30)  # 30 day retention
        
        # Update voice info with results
        voice_summaries_db[voice_id].update({
            'status': 'completed',
            'progress': 100,
            'audio_url': voice_result['audio_url'],
            'download_url': voice_result.get('download_url'),
            'duration': voice_result['duration'],
            'transcript': voice_result.get('transcript', content_to_vocalize),
            'file_size': voice_result['file_size'],
            'audio_format': voice_result.get('format', 'mp3'),
            'completed_at': datetime.now(timezone.utc),
            'expires_at': expires_at,
            'generation_time': (datetime.now(timezone.utc) - start_time).total_seconds()
        })
        
        voice_generation_status[voice_id] = {
            'stage': 'completed',
            'progress': 100,
            'completed_at': datetime.now(timezone.utc)
        }
        
        # Update analytics
        voice_analytics['total_generated'] += 1
        voice_analytics['total_duration'] += voice_result['duration']
        
        lang = voice_request.language
        voice_analytics['languages_used'][lang] = voice_analytics['languages_used'].get(lang, 0) + 1
        
        logger.info(f"Voice generation completed successfully", extra={
            'voice_id': voice_id,
            'duration': voice_result['duration'],
            'file_size': voice_result['file_size'],
            'generation_time': (datetime.now(timezone.utc) - start_time).total_seconds()
        })
        
    except Exception as e:
        logger.error(f"Error in voice generation workflow: {str(e)}", extra={
            'voice_id': voice_id,
            'request_id': request_id
        })
        
        voice_summaries_db[voice_id].update({
            'status': 'failed',
            'error_message': str(e),
            'failed_at': datetime.now(timezone.utc)
        })
        
        voice_generation_status[voice_id] = {
            'stage': 'failed',
            'progress': 0,
            'error': str(e)
        }


async def generate_multilingual_workflow(
    document_id: str,
    document_data: Dict[str, Any],
    languages: List[str],
    voice_type: str,
    content_type: str,
    user_context: Optional[Dict],
    request_id: str
):
    """
    Generate voice summaries in multiple languages
    """
    
    try:
        logger.info(f"Starting multilingual voice generation", extra={
            'document_id': document_id,
            'languages': languages,
            'request_id': request_id
        })
        
        # Prepare content once
        content_to_vocalize = await _prepare_voice_content(
            document_data,
            content_type,
            max_length=2000
        )
        
        # Generate voices for each language
        generated_voices = []
        
        for language in languages:
            try:
                # Create voice request
                voice_request = VoiceRequest(
                    document_id=document_id,
                    language=language,
                    voice_type=voice_type,
                    content_type=content_type
                )
                
                voice_id = str(uuid.uuid4())
                
                # Initialize voice info
                voice_info = {
                    'voice_id': voice_id,
                    'document_id': document_id,
                    'language': language,
                    'voice_type': voice_type,
                    'content_type': content_type,
                    'status': 'initializing',
                    'created_at': datetime.now(timezone.utc),
                    'user_id': user_context.get('user_id') if user_context else 'anonymous',
                    'request_id': request_id,
                    'batch_id': request_id,
                    'metadata': {
                        'document_title': document_data.get('title', document_data.get('filename')),
                        'batch_generation': True
                    }
                }
                
                voice_summaries_db[voice_id] = voice_info
                
                # Generate voice
                await generate_voice_workflow(
                    voice_id,
                    document_data,
                    voice_request,
                    content_to_vocalize,
                    request_id
                )
                
                generated_voices.append({
                    'language': language,
                    'voice_id': voice_id,
                    'status': voice_summaries_db[voice_id].get('status')
                })
                
            except Exception as e:
                logger.error(f"Failed to generate voice for {language}: {str(e)}")
                generated_voices.append({
                    'language': language,
                    'voice_id': None,
                    'status': 'failed',
                    'error': str(e)
                })
        
        logger.info(f"Multilingual voice generation completed", extra={
            'document_id': document_id,
            'total_languages': len(languages),
            'successful': len([v for v in generated_voices if v['status'] == 'completed']),
            'request_id': request_id
        })
        
    except Exception as e:
        logger.error(f"Error in multilingual voice generation: {str(e)}", extra={'request_id': request_id})


# Helper functions

async def _validate_voice_request_enhanced(voice_request: VoiceRequest, user_context: Optional[Dict]) -> Dict[str, Any]:
    """Enhanced voice request validation"""
    
    # Basic validation
    basic_validation = validate_voice_request(voice_request)
    if not basic_validation['valid']:
        return basic_validation
    
    # Check voice model availability
    available_voices = await voice_generator.get_available_voices(voice_request.language)
    if not available_voices:
        return {
            'valid': False,
            'error': f'No voices available for language: {voice_request.language}',
            'suggestions': ['Try a different language', 'Check supported languages endpoint']
        }
    
    # Validate content type
    valid_content_types = ['summary', 'risks', 'recommendations', 'full', 'executive']
    if voice_request.content_type not in valid_content_types:
        return {
            'valid': False,
            'error': f'Invalid content type: {voice_request.content_type}',
            'suggestions': [f'Valid options: {", ".join(valid_content_types)}']
        }
    
    return {'valid': True}


async def _get_and_validate_document_for_voice(document_id: str, user_context: Optional[Dict]) -> Dict[str, Any]:
    """Get and validate document for voice generation"""
    
    if document_id not in documents_db:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )
    
    document_data = documents_db[document_id]
    
    if document_data['status'] != 'completed':
        raise HTTPException(
            status_code=400,
            detail=f"Document not ready for voice generation. Status: {document_data['status']}"
        )
    
    # Check access
    if not await _check_document_access_for_voice(document_id, user_context):
        raise HTTPException(
            status_code=403,
            detail="Access denied to this document"
        )
    
    return document_data


async def _check_voice_generation_limits(user_id: str) -> Dict[str, Any]:
    """Check voice generation limits for user"""
    
    # Count user's voices in the last 24 hours
    now = datetime.now(timezone.utc)
    yesterday = now - timedelta(days=1)
    
    user_voices_today = [
        v for v in voice_summaries_db.values()
        if v.get('user_id') == user_id and v.get('created_at', datetime.min.replace(tzinfo=timezone.utc)) >= yesterday
    ]
    
    daily_limit = 10  # Configurable
    
    if len(user_voices_today) >= daily_limit:
        return {
            'allowed': False,
            'message': f'Daily voice generation limit of {daily_limit} exceeded',
            'limit_info': {
                'daily_limit': daily_limit,
                'used_today': len(user_voices_today),
                'resets_at': now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            }
        }
    
    return {'allowed': True}


async def _prepare_voice_content(document_data: Dict[str, Any], content_type: str, max_length: int = 3000) -> str:
    """Prepare content for voice generation based on type"""
    
    if content_type == 'summary':
        content = document_data.get('summary', '')
    elif content_type == 'executive':
        content = document_data.get('executive_summary', document_data.get('summary', ''))
    elif content_type == 'risks':
        risks = document_data.get('key_risks', [])
        content = f"Key risks identified: {'. '.join(risks[:5])}"  # Top 5 risks
    elif content_type == 'recommendations':
        recommendations = document_data.get('recommendations', [])
        content = f"Recommendations: {'. '.join(recommendations[:5])}"  # Top 5 recommendations
    elif content_type == 'full':
        content = document_data.get('extracted_text', '')
    else:
        content = document_data.get('summary', '')
    
    # Limit content length
    if len(content) > max_length:
        content = content[:max_length] + "..."
    
    return content


async def _check_voice_access(voice_id: str, user_context: Optional[Dict]) -> bool:
    """Check if user has access to voice"""
    
    if voice_id not in voice_summaries_db:
        return False
    
    voice_info = voice_summaries_db[voice_id]
    
    # Public access for demo
    if not user_context:
        return True
    
    # Owner access
    user_id = user_context.get('user_id', 'anonymous')
    if voice_info.get('user_id') == user_id:
        return True
    
    # Admin access
    if user_context.get('role') == 'admin':
        return True
    
    return False


async def _check_document_access_for_voice(document_id: str, user_context: Optional[Dict]) -> bool:
    """Check if user has access to document for voice generation"""
    
    # Import function from documents module
    from app.api.routes.documents import _check_document_access
    return await _check_document_access(document_id, user_context)


async def _update_voice_status(voice_id: str, stage: str, progress: int, description: str):
    """Update voice generation status"""
    
    voice_generation_status[voice_id] = {
        'stage': stage,
        'progress': progress,
        'description': description,
        'last_updated': datetime.now(timezone.utc)
    }
    
    voice_summaries_db[voice_id]['status'] = stage
    voice_summaries_db[voice_id]['progress'] = progress


def _calculate_voice_priority(user_context: Optional[Dict]) -> int:
    """Calculate voice generation priority"""
    
    base_priority = 5
    
    if not user_context:
        return base_priority
    
    # Premium users get higher priority
    if user_context.get('subscription_tier') == 'premium':
        base_priority += 3
    elif user_context.get('subscription_tier') == 'pro':
        base_priority += 2
    
    return min(10, base_priority)


def _estimate_voice_generation_time(word_count: int, language: str, voice_type: str) -> int:
    """Estimate voice generation time"""
    
    # Base time: ~150 words per minute of audio, plus processing overhead
    base_time = max(20, (word_count / 150) * 60 * 0.4)  # 40% of actual audio time
    
    # Language complexity factor
    complex_languages = ['hi', 'ta', 'te', 'bn', 'gu']
    if language in complex_languages:
        base_time *= 1.3
    
    # Voice type factor
    if voice_type == 'neural':
        base_time *= 1.2
    
    return int(base_time)
