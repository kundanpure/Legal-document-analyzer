"""
Enhanced Document management API routes
Production-ready with comprehensive features, validation, and monitoring
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional, List, Dict, Any
import asyncio
import uuid
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

from config.logging import get_logger, log_api_request, log_document_processing
from config.settings import get_settings
from app.models.requests import DocumentUploadRequest, DocumentAnalysisRequest
from app.models.responses import (
    DocumentUploadResponse, DocumentStatusResponse, DocumentLibraryResponse,
    DocumentSummaryResponse, DocumentAnalysisResponse, ErrorResponse
)
from app.services.document_processor import DocumentProcessor
from app.services.gemini_analyzer import GeminiAnalyzer
from app.services.storage_manager import StorageManager
from app.services.chat_handler import ChatHandler
from app.utils.validators import validate_file_upload, validate_query_length, validate_document_id
from app.utils.helpers import generate_document_id, generate_request_id, sanitize_filename
from app.utils.formatters import format_file_size, format_duration
from app.core.exceptions import DocumentProcessingError, ValidationError
from app.core.dependencies import get_current_user, rate_limit

settings = get_settings()
logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

# Service instances
document_processor = DocumentProcessor()
gemini_analyzer = GeminiAnalyzer()
storage_manager = StorageManager()
chat_handler = ChatHandler()

# Enhanced in-memory storage with metadata
documents_db = {}
processing_status = {}
document_analytics = {'total_uploaded': 0, 'total_processed': 0, 'total_failed': 0}


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    document: UploadFile = File(..., description="Legal document to analyze (PDF, DOCX, DOC, TXT)"),
    query: str = Form("Analyze this legal document for risks and key terms", description="Analysis query"),
    language: str = Form("en", description="Analysis language"),
    priority: str = Form("normal", description="Processing priority (low/normal/high)"),
    tags: Optional[str] = Form(None, description="Comma-separated tags"),
    client_info: Optional[str] = Form(None, description="Client metadata"),
    current_user: Optional[Dict] = Depends(get_current_user),
    _rate_limit: None = Depends(rate_limit("document_upload"))
):
    """
    Upload and process a legal document with enhanced features
    """
    
    request_id = generate_request_id()
    start_time = datetime.now(timezone.utc)
    
    try:
        logger.info(
            f"Document upload request",
            extra={
                'request_id': request_id,
                'filename': document.filename,
                'file_size': getattr(document, 'size', 0),
                'user_id': current_user.get('user_id') if current_user else None,
                'priority': priority
            }
        )
        
        # Enhanced file validation
        validation_result = await validate_file_upload(document)
        if not validation_result['valid']:
            raise HTTPException(
                status_code=400,
                detail={
                    'error': 'FILE_VALIDATION_FAILED',
                    'message': validation_result['error'],
                    'errors': validation_result.get('errors', []),
                    'recommendations': validation_result.get('recommendations', []),
                    'request_id': request_id
                }
            )
        
        # Query validation
        if not validate_query_length(query, min_length=5, max_length=1000):
            raise HTTPException(
                status_code=400,
                detail={
                    'error': 'INVALID_QUERY',
                    'message': "Query must be between 5 and 1000 characters",
                    'request_id': request_id
                }
            )
        
        # Check user upload limits
        user_id = current_user.get('user_id') if current_user else 'anonymous'
        upload_limit_check = await _check_user_upload_limits(user_id)
        if not upload_limit_check['allowed']:
            raise HTTPException(
                status_code=429,
                detail={
                    'error': 'UPLOAD_LIMIT_EXCEEDED',
                    'message': upload_limit_check['message'],
                    'limit_info': upload_limit_check['limit_info'],
                    'request_id': request_id
                }
            )
        
        # Generate unique IDs
        document_id = generate_document_id("doc")
        session_id = generate_document_id("session")
        
        # Parse tags and client info
        parsed_tags = [tag.strip() for tag in tags.split(',')] if tags else []
        parsed_client_info = json.loads(client_info) if client_info else {}
        
        # Sanitize filename
        safe_filename = sanitize_filename(document.filename)
        
        # Store initial document info with enhanced metadata
        document_info = {
            'document_id': document_id,
            'session_id': session_id,
            'filename': safe_filename,
            'original_filename': document.filename,
            'file_size': getattr(document, 'size', 0),
            'file_size_formatted': format_file_size(getattr(document, 'size', 0)),
            'content_type': document.content_type,
            'upload_date': datetime.now(timezone.utc),
            'status': 'uploaded',
            'progress': 0,
            'query': query,
            'language': language,
            'priority': priority,
            'tags': parsed_tags,
            'user_id': user_id,
            'client_info': parsed_client_info,
            'request_id': request_id,
            'processing_stages': {},
            'error_message': None,
            'retry_count': 0,
            'version': '2.0'
        }
        
        documents_db[document_id] = document_info
        processing_status[document_id] = {
            'stage': 'uploaded', 
            'progress': 0,
            'started_at': datetime.now(timezone.utc)
        }
        
        # Update analytics
        document_analytics['total_uploaded'] += 1
        
        # Calculate processing priority score
        priority_score = _calculate_priority_score(priority, user_id, getattr(document, 'size', 0))
        
        # Start background processing
        background_tasks.add_task(
            process_document_workflow,
            document_id,
            document,
            query,
            language,
            priority_score,
            request_id
        )
        
        # Estimate processing time
        estimated_time = await document_processor.estimate_processing_time(
            getattr(document, 'size', 0),
            document_type=_detect_document_type_from_filename(document.filename)
        )
        
        # Log API request
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        background_tasks.add_task(
            log_api_request,
            'POST',
            '/documents/upload',
            200,
            processing_time,
            user_id,
            session_id,
            request_id
        )
        
        response = DocumentUploadResponse(
            success=True,
            document_id=document_id,
            session_id=session_id,
            status="processing",
            estimated_time=format_duration(estimated_time),
            priority=priority,
            file_info={
                'filename': safe_filename,
                'size': format_file_size(getattr(document, 'size', 0)),
                'type': document.content_type
            },
            request_id=request_id
        )
        
        logger.info(f"Document upload initiated successfully", extra={'request_id': request_id, 'document_id': document_id})
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in document upload: {str(e)}", extra={'request_id': request_id})
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'UPLOAD_FAILED',
                'message': f"Document upload failed: {str(e)}",
                'request_id': request_id
            }
        )


@router.get("/status/{document_id}", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: str,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Get detailed processing status for a document with real-time updates
    """
    
    try:
        # Validate document ID
        if not validate_document_id(document_id):
            raise HTTPException(
                status_code=400,
                detail="Invalid document ID format"
            )
        
        if document_id not in documents_db:
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )
        
        doc_info = documents_db[document_id]
        status_info = processing_status.get(document_id, {})
        
        # Check user access
        if not await _check_document_access(document_id, current_user):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this document"
            )
        
        # Calculate processing time
        processing_time = None
        if status_info.get('started_at'):
            if doc_info['status'] == 'completed':
                processing_time = (doc_info.get('completed_at', datetime.now(timezone.utc)) - status_info['started_at']).total_seconds()
            else:
                processing_time = (datetime.now(timezone.utc) - status_info['started_at']).total_seconds()
        
        # Get stage details
        stage_details = doc_info.get('processing_stages', {})
        current_stage_info = stage_details.get(status_info.get('stage'), {})
        
        return DocumentStatusResponse(
            document_id=document_id,
            status=doc_info['status'],
            progress=status_info.get('progress', 0),
            current_stage=status_info.get('stage'),
            stage_description=current_stage_info.get('description', ''),
            processing_time=format_duration(processing_time) if processing_time else None,
            estimated_remaining=current_stage_info.get('estimated_remaining'),
            error_message=doc_info.get('error_message'),
            retry_count=doc_info.get('retry_count', 0),
            priority=doc_info.get('priority', 'normal'),
            last_updated=status_info.get('last_updated', datetime.now(timezone.utc)).isoformat(),
            can_retry=doc_info.get('status') == 'failed' and doc_info.get('retry_count', 0) < 3
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get document status"
        )


@router.get("/", response_model=DocumentLibraryResponse)
async def get_user_documents(
    current_user: Optional[Dict] = Depends(get_current_user),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=50, description="Items per page"),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    document_type: Optional[str] = Query(None, description="Filter by document type"),
    language: Optional[str] = Query(None, description="Filter by language"),
    sort_by: str = Query("upload_date", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
    search: Optional[str] = Query(None, description="Search in filename/title"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)")
):
    """
    Get user's document library with advanced filtering and search
    """
    
    try:
        user_id = current_user.get('user_id', 'anonymous') if current_user else 'anonymous'
        
        # Parse tags filter
        tag_filters = [tag.strip() for tag in tags.split(',')] if tags else []
        
        # Filter documents
        filtered_docs = []
        
        for doc_id, doc_info in documents_db.items():
            # User access check
            if not await _check_document_access(doc_id, current_user):
                continue
            
            # Apply filters
            if status_filter and doc_info.get('status') != status_filter:
                continue
                
            if document_type and doc_info.get('document_type') != document_type:
                continue
                
            if language and doc_info.get('language') != language:
                continue
            
            # Search filter
            if search:
                searchable_text = f"{doc_info.get('filename', '')} {doc_info.get('title', '')}".lower()
                if search.lower() not in searchable_text:
                    continue
            
            # Tags filter
            if tag_filters:
                doc_tags = doc_info.get('tags', [])
                if not any(tag in doc_tags for tag in tag_filters):
                    continue
            
            # Create document summary
            doc_summary = {
                'document_id': doc_id,
                'filename': doc_info.get('filename', ''),
                'title': doc_info.get('title', doc_info.get('filename', '')),
                'upload_date': doc_info.get('upload_date'),
                'document_type': doc_info.get('document_type', 'unknown'),
                'status': doc_info.get('status'),
                'progress': processing_status.get(doc_id, {}).get('progress', 0),
                'risk_level': doc_info.get('risk_level', 'unknown'),
                'risk_score': doc_info.get('overall_risk_score', 0),
                'file_size': doc_info.get('file_size', 0),
                'file_size_formatted': doc_info.get('file_size_formatted', ''),
                'language': doc_info.get('language', 'en'),
                'priority': doc_info.get('priority', 'normal'),
                'tags': doc_info.get('tags', []),
                'has_chat': bool(doc_info.get('chat_session_id')),
                'processing_time': doc_info.get('processing_time'),
                'completed_at': doc_info.get('completed_at'),
                'page_count': doc_info.get('page_count', 0),
                'word_count': doc_info.get('word_count', 0)
            }
            
            filtered_docs.append(doc_summary)
        
        # Sort documents
        reverse_sort = sort_order.lower() == 'desc'
        
        if sort_by == 'upload_date':
            filtered_docs.sort(key=lambda x: x['upload_date'] or datetime.min, reverse=reverse_sort)
        elif sort_by == 'filename':
            filtered_docs.sort(key=lambda x: x['filename'].lower(), reverse=reverse_sort)
        elif sort_by == 'risk_score':
            filtered_docs.sort(key=lambda x: x['risk_score'], reverse=reverse_sort)
        elif sort_by == 'status':
            filtered_docs.sort(key=lambda x: x['status'], reverse=reverse_sort)
        
        # Apply pagination
        total_count = len(filtered_docs)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_docs = filtered_docs[start_idx:end_idx]
        
        # Calculate statistics
        status_counts = {}
        type_counts = {}
        for doc in filtered_docs:
            status = doc['status']
            doc_type = doc['document_type']
            status_counts[status] = status_counts.get(status, 0) + 1
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        return DocumentLibraryResponse(
            documents=paginated_docs,
            total_count=total_count,
            page=page,
            per_page=per_page,
            total_pages=(total_count + per_page - 1) // per_page,
            has_next=end_idx < total_count,
            has_prev=page > 1,
            filters_applied={
                'status': status_filter,
                'document_type': document_type,
                'language': language,
                'search': search,
                'tags': tag_filters
            },
            statistics={
                'status_counts': status_counts,
                'type_counts': type_counts,
                'total_size': sum(doc['file_size'] for doc in filtered_docs)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document library: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get document library"
        )


@router.get("/{document_id}/summary", response_model=DocumentSummaryResponse)
async def get_document_summary(
    document_id: str,
    include_full_text: bool = Query(False, description="Include full extracted text"),
    include_analysis_details: bool = Query(True, description="Include detailed analysis"),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Get comprehensive summary of a processed document
    """
    
    try:
        if not validate_document_id(document_id):
            raise HTTPException(
                status_code=400,
                detail="Invalid document ID format"
            )
        
        if document_id not in documents_db:
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )
        
        # Check access
        if not await _check_document_access(document_id, current_user):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this document"
            )
        
        doc_info = documents_db[document_id]
        
        if doc_info['status'] != 'completed':
            raise HTTPException(
                status_code=202,
                detail={
                    'message': 'Document still processing',
                    'status': doc_info['status'],
                    'progress': processing_status.get(document_id, {}).get('progress', 0)
                }
            )
        
        # Build comprehensive summary
        summary_data = {
            'document_id': document_id,
            'document_type': doc_info.get('document_type', 'unknown'),
            'title': doc_info.get('title', doc_info.get('filename', '')),
            'filename': doc_info.get('filename', ''),
            'summary': doc_info.get('summary', 'Summary not available'),
            'executive_summary': doc_info.get('executive_summary', ''),
            'key_risks': doc_info.get('key_risks', []),
            'recommendations': doc_info.get('recommendations', []),
            'risk_score': doc_info.get('overall_risk_score', 0),
            'risk_level': doc_info.get('risk_level', 'unknown'),
            'risk_categories': doc_info.get('risk_categories', {}),
            'confidence_score': doc_info.get('confidence_score', 0.8),
            'processing_metadata': {
                'processing_time': doc_info.get('processing_time', 'Unknown'),
                'completed_at': doc_info.get('completed_at'),
                'language': doc_info.get('language', 'en'),
                'version': doc_info.get('version', '1.0'),
                'model_used': doc_info.get('model_used', settings.gemini.model)
            },
            'document_statistics': {
                'page_count': doc_info.get('page_count', 0),
                'word_count': doc_info.get('word_count', 0),
                'paragraph_count': doc_info.get('paragraph_count', 0),
                'file_size': doc_info.get('file_size_formatted', ''),
                'reading_time': doc_info.get('estimated_reading_time', '')
            }
        }
        
        # Add detailed analysis if requested
        if include_analysis_details:
            summary_data.update({
                'flagged_clauses': doc_info.get('flagged_clauses', []),
                'user_obligations': doc_info.get('user_obligations', []),
                'user_rights': doc_info.get('user_rights', []),
                'financial_implications': doc_info.get('financial_implications', {}),
                'fairness_score': doc_info.get('fairness_score', 5.0),
                'key_topics': doc_info.get('key_topics', []),
                'entities_extracted': doc_info.get('entities_extracted', {}),
                'legal_precedents': doc_info.get('legal_precedents', []),
                'compliance_requirements': doc_info.get('compliance_requirements', [])
            })
        
        # Add full text if requested and user has permission
        if include_full_text and await _check_full_text_access(document_id, current_user):
            summary_data['full_text'] = doc_info.get('extracted_text', '')
        
        return DocumentSummaryResponse(**summary_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get document summary"
        )


@router.post("/{document_id}/reanalyze")
async def reanalyze_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    analysis_request: DocumentAnalysisRequest,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Re-analyze a document with different parameters
    """
    
    try:
        if document_id not in documents_db:
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )
        
        # Check access and permissions
        if not await _check_document_access(document_id, current_user):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this document"
            )
        
        doc_info = documents_db[document_id]
        
        if doc_info['status'] != 'completed':
            raise HTTPException(
                status_code=400,
                detail="Document must be completed before re-analysis"
            )
        
        # Update document with new analysis parameters
        doc_info.update({
            'query': analysis_request.query,
            'language': analysis_request.language,
            'analysis_focus': analysis_request.focus_areas,
            'status': 'reanalyzing',
            'retry_count': doc_info.get('retry_count', 0) + 1
        })
        
        processing_status[document_id] = {
            'stage': 'reanalyzing',
            'progress': 0,
            'started_at': datetime.now(timezone.utc)
        }
        
        # Start reanalysis
        background_tasks.add_task(
            reanalyze_document_workflow,
            document_id,
            analysis_request,
            current_user
        )
        
        return {
            'success': True,
            'message': 'Document re-analysis started',
            'document_id': document_id,
            'estimated_time': '30-60 seconds'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting document re-analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to start document re-analysis"
        )


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Delete a document and all associated data
    """
    
    try:
        if document_id not in documents_db:
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )
        
        # Check access
        if not await _check_document_access(document_id, current_user):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this document"
            )
        
        doc_info = documents_db[document_id]
        
        # Delete from cloud storage
        try:
            await storage_manager.delete_document(document_id)
        except Exception as e:
            logger.warning(f"Failed to delete from storage: {str(e)}")
        
        # Delete associated chat sessions
        try:
            if doc_info.get('chat_session_id'):
                await chat_handler.end_chat_session(doc_info['chat_session_id'])
        except Exception as e:
            logger.warning(f"Failed to delete chat session: {str(e)}")
        
        # Remove from local storage
        del documents_db[document_id]
        if document_id in processing_status:
            del processing_status[document_id]
        
        logger.info(f"Document deleted successfully", extra={'document_id': document_id})
        
        return {
            'success': True,
            'message': 'Document deleted successfully',
            'deleted_at': datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete document"
        )


@router.get("/{document_id}/download")
async def download_document(
    document_id: str,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Download original document file with access control
    """
    
    try:
        if document_id not in documents_db:
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )
        
        # Check access
        if not await _check_document_access(document_id, current_user):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this document"
            )
        
        doc_info = documents_db[document_id]
        
        # Get document from storage
        document_content = await storage_manager.get_document(document_id)
        
        if not document_content:
            raise HTTPException(
                status_code=404,
                detail="Document file not found in storage"
            )
        
        # Return streaming response for large files
        def generate_file_stream():
            chunk_size = 8192
            for i in range(0, len(document_content), chunk_size):
                yield document_content[i:i + chunk_size]
        
        return StreamingResponse(
            generate_file_stream(),
            media_type=doc_info.get('content_type', 'application/octet-stream'),
            headers={
                "Content-Disposition": f"attachment; filename={doc_info.get('filename', 'document')}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to download document"
        )


@router.get("/analytics/overview")
async def get_document_analytics(
    current_user: Optional[Dict] = Depends(get_current_user),
    time_range: str = Query("7d", description="Time range (1d, 7d, 30d)")
):
    """
    Get document processing analytics
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
        
        # Filter documents by time range and user
        user_docs = []
        for doc_id, doc_info in documents_db.items():
            if await _check_document_access(doc_id, current_user):
                if doc_info.get('upload_date', datetime.min.replace(tzinfo=timezone.utc)) >= start_date:
                    user_docs.append(doc_info)
        
        # Calculate analytics
        analytics = {
            'time_range': time_range,
            'total_documents': len(user_docs),
            'by_status': {},
            'by_type': {},
            'by_risk_level': {},
            'processing_stats': {
                'avg_processing_time': 0,
                'total_pages': 0,
                'total_words': 0
            },
            'upload_trends': []
        }
        
        # Status distribution
        for doc in user_docs:
            status = doc.get('status', 'unknown')
            analytics['by_status'][status] = analytics['by_status'].get(status, 0) + 1
        
        # Document type distribution
        for doc in user_docs:
            doc_type = doc.get('document_type', 'unknown')
            analytics['by_type'][doc_type] = analytics['by_type'].get(doc_type, 0) + 1
        
        # Risk level distribution
        for doc in user_docs:
            risk_level = doc.get('risk_level', 'unknown')
            analytics['by_risk_level'][risk_level] = analytics['by_risk_level'].get(risk_level, 0) + 1
        
        # Processing statistics
        completed_docs = [d for d in user_docs if d.get('status') == 'completed']
        if completed_docs:
            total_pages = sum(d.get('page_count', 0) for d in completed_docs)
            total_words = sum(d.get('word_count', 0) for d in completed_docs)
            
            analytics['processing_stats'] = {
                'avg_processing_time': sum(
                    float(d.get('processing_time', '0').replace('s', '')) for d in completed_docs
                ) / len(completed_docs),
                'total_pages': total_pages,
                'total_words': total_words,
                'avg_pages_per_doc': total_pages / len(completed_docs),
                'avg_words_per_doc': total_words / len(completed_docs)
            }
        
        return {
            'success': True,
            'analytics': analytics,
            'generated_at': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting document analytics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get document analytics"
        )


# Background processing workflows

async def process_document_workflow(
    document_id: str,
    document: UploadFile,
    query: str,
    language: str,
    priority_score: int,
    request_id: str
):
    """
    Enhanced document processing workflow with detailed tracking
    """
    
    start_time = datetime.now(timezone.utc)
    
    try:
        logger.info(f"Starting enhanced document processing", extra={'document_id': document_id, 'request_id': request_id})
        
        # Stage 1: File Upload and Validation
        await _update_processing_status(document_id, 'uploading', 10, 'Uploading file to secure storage')
        
        file_content = await document.read()
        
        # Upload to cloud storage with metadata
        gcs_uri = await storage_manager.upload_document(
            file_content, 
            document.filename, 
            document_id, 
            document.content_type,
            metadata={
                'request_id': request_id,
                'priority': priority_score,
                'language': language
            }
        )
        
        documents_db[document_id]['gcs_uri'] = gcs_uri
        
        # Stage 2: Text Extraction
        await _update_processing_status(document_id, 'extracting_text', 30, 'Extracting text content from document')
        
        extraction_result = await document_processor.extract_text_from_upload(
            file_content, 
            document.filename,
            enhanced_ocr=True
        )
        
        documents_db[document_id].update({
            'extracted_text': extraction_result['full_text'],
            'page_count': extraction_result['statistics']['total_pages'],
            'word_count': extraction_result['statistics']['total_words'],
            'paragraph_count': extraction_result['statistics'].get('total_paragraphs', 0),
            'extraction_confidence': extraction_result.get('confidence', 0.9),
            'estimated_reading_time': _calculate_reading_time(extraction_result['statistics']['total_words'])
        })
        
        # Stage 3: Document Classification
        await _update_processing_status(document_id, 'classifying', 50, 'Identifying document type and structure')
        
        classification_result = await gemini_analyzer.classify_document(
            extraction_result['full_text'],
            document.filename
        )
        
        documents_db[document_id].update({
            'document_type': classification_result.get('document_type', 'unknown'),
            'document_subtype': classification_result.get('document_subtype'),
            'classification_confidence': classification_result.get('confidence', 0.8)
        })
        
        # Stage 4: AI Analysis
        await _update_processing_status(document_id, 'analyzing', 70, 'Performing comprehensive AI analysis')
        
        analysis_result = await gemini_analyzer.analyze_document_comprehensive(
            text=extraction_result['full_text'],
            query=query,
            language=language,
            filename=document.filename,
            document_type=classification_result.get('document_type'),
            priority=priority_score
        )
        
        # Stage 5: Post-processing and Finalization
        await _update_processing_status(document_id, 'finalizing', 90, 'Finalizing analysis and preparing results')
        
        # Calculate additional metrics
        risk_level = _determine_risk_level(analysis_result.get('overall_risk_score', 0))
        confidence_score = _calculate_overall_confidence(analysis_result, extraction_result)
        
        # Update document with comprehensive results
        documents_db[document_id].update({
            'status': 'completed',
            'progress': 100,
            'title': analysis_result.get('title', document.filename),
            'summary': analysis_result.get('summary'),
            'executive_summary': analysis_result.get('executive_summary'),
            'key_risks': analysis_result.get('key_risks', []),
            'recommendations': analysis_result.get('recommendations', []),
            'overall_risk_score': analysis_result.get('overall_risk_score', 0),
            'risk_categories': analysis_result.get('risk_categories', {}),
            'flagged_clauses': analysis_result.get('flagged_clauses', []),
            'user_obligations': analysis_result.get('user_obligations', []),
            'user_rights': analysis_result.get('user_rights', []),
            'financial_implications': analysis_result.get('financial_implications', {}),
            'fairness_score': analysis_result.get('fairness_score', 5.0),
            'key_topics': analysis_result.get('key_topics', []),
            'entities_extracted': analysis_result.get('entities_extracted', {}),
            'legal_precedents': analysis_result.get('legal_precedents', []),
            'compliance_requirements': analysis_result.get('compliance_requirements', []),
            'risk_level': risk_level,
            'confidence_score': confidence_score,
            'processing_time': format_duration((datetime.now(timezone.utc) - start_time).total_seconds()),
            'completed_at': datetime.now(timezone.utc),
            'model_used': settings.gemini.model,
            'analysis_version': '2.0'
        })
        
        processing_status[document_id] = {
            'stage': 'completed', 
            'progress': 100,
            'completed_at': datetime.now(timezone.utc)
        }
        
        # Initialize chat session
        try:
            chat_session = await chat_handler.initialize_chat_session(
                document_id, 
                documents_db[document_id], 
                language
            )
            documents_db[document_id]['chat_session_id'] = chat_session['session_id']
        except Exception as e:
            logger.warning(f"Failed to initialize chat session: {str(e)}")
        
        # Update analytics
        document_analytics['total_processed'] += 1
        
        # Log completion
        processing_duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        log_document_processing(
            document_id=document_id,
            operation="full_processing",
            status="success",
            processing_time=processing_duration,
            file_size=documents_db[document_id].get('file_size'),
            document_type=documents_db[document_id].get('document_type'),
            user_id=documents_db[document_id].get('user_id'),
            session_id=documents_db[document_id].get('session_id')
        )
        
        logger.info(f"Document processing completed successfully", extra={
            'document_id': document_id,
            'processing_time': processing_duration,
            'request_id': request_id
        })
        
    except Exception as e:
        logger.error(f"Error in document processing workflow: {str(e)}", extra={
            'document_id': document_id,
            'request_id': request_id
        })
        
        # Update status to failed
        documents_db[document_id].update({
            'status': 'failed',
            'error_message': str(e),
            'failed_at': datetime.now(timezone.utc),
            'retry_available': True
        })
        
        processing_status[document_id] = {
            'stage': 'failed', 
            'progress': 0,
            'error': str(e)
        }
        
        # Update analytics
        document_analytics['total_failed'] += 1
        
        # Log failure
        log_document_processing(
            document_id=document_id,
            operation="full_processing", 
            status="error",
            processing_time=(datetime.now(timezone.utc) - start_time).total_seconds(),
            error=str(e),
            user_id=documents_db[document_id].get('user_id')
        )


async def reanalyze_document_workflow(
    document_id: str,
    analysis_request: DocumentAnalysisRequest,
    user_context: Optional[Dict]
):
    """
    Re-analyze document with new parameters
    """
    
    try:
        doc_info = documents_db[document_id]
        
        # Perform new analysis
        analysis_result = await gemini_analyzer.analyze_document_comprehensive(
            text=doc_info['extracted_text'],
            query=analysis_request.query,
            language=analysis_request.language,
            filename=doc_info['filename'],
            document_type=doc_info.get('document_type'),
            focus_areas=analysis_request.focus_areas
        )
        
        # Update document with new analysis
        doc_info.update({
            'status': 'completed',
            'query': analysis_request.query,
            'language': analysis_request.language,
            'analysis_focus': analysis_request.focus_areas,
            'summary': analysis_result.get('summary'),
            'key_risks': analysis_result.get('key_risks', []),
            'recommendations': analysis_result.get('recommendations', []),
            'overall_risk_score': analysis_result.get('overall_risk_score', 0),
            'risk_level': _determine_risk_level(analysis_result.get('overall_risk_score', 0)),
            'reanalyzed_at': datetime.now(timezone.utc),
            'analysis_version': '2.0'
        })
        
        processing_status[document_id] = {
            'stage': 'completed',
            'progress': 100,
            'reanalyzed_at': datetime.now(timezone.utc)
        }
        
        logger.info(f"Document re-analysis completed", extra={'document_id': document_id})
        
    except Exception as e:
        logger.error(f"Error in document re-analysis: {str(e)}")
        
        doc_info.update({
            'status': 'failed',
            'error_message': f"Re-analysis failed: {str(e)}",
            'failed_at': datetime.now(timezone.utc)
        })


# Helper functions

async def _update_processing_status(document_id: str, stage: str, progress: int, description: str):
    """Update processing status with detailed information"""
    
    processing_status[document_id] = {
        'stage': stage,
        'progress': progress,
        'last_updated': datetime.now(timezone.utc)
    }
    
    documents_db[document_id]['processing_stages'][stage] = {
        'description': description,
        'started_at': datetime.now(timezone.utc),
        'progress': progress
    }
    
    documents_db[document_id]['status'] = stage


async def _check_user_upload_limits(user_id: str) -> Dict[str, Any]:
    """Check if user has exceeded upload limits"""
    
    # Basic rate limiting - in production, use Redis or database
    user_docs = [d for d in documents_db.values() if d.get('user_id') == user_id]
    
    # Check daily limit
    today_docs = [
        d for d in user_docs 
        if d.get('upload_date', datetime.min.replace(tzinfo=timezone.utc)).date() == datetime.now(timezone.utc).date()
    ]
    
    daily_limit = 20  # Configurable limit
    
    if len(today_docs) >= daily_limit:
        return {
            'allowed': False,
            'message': f'Daily upload limit of {daily_limit} documents exceeded',
            'limit_info': {
                'daily_limit': daily_limit,
                'uploaded_today': len(today_docs),
                'resets_at': datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            }
        }
    
    return {'allowed': True}


async def _check_document_access(document_id: str, user_context: Optional[Dict]) -> bool:
    """Check if user has access to document"""
    
    if document_id not in documents_db:
        return False
    
    doc_info = documents_db[document_id]
    
    # Public access for demo
    if not user_context:
        return True
    
    # Owner access
    user_id = user_context.get('user_id', 'anonymous')
    if doc_info.get('user_id') == user_id:
        return True
    
    # Admin access
    if user_context.get('role') == 'admin':
        return True
    
    # Shared access (implement if needed)
    return False


async def _check_full_text_access(document_id: str, user_context: Optional[Dict]) -> bool:
    """Check if user has access to full document text"""
    
    # Basic access check + additional permissions for full text
    if not await _check_document_access(document_id, user_context):
        return False
    
    # Additional checks for premium features could be added here
    return True


def _calculate_priority_score(priority: str, user_id: str, file_size: int) -> int:
    """Calculate processing priority score"""
    
    base_score = {'low': 1, 'normal': 5, 'high': 9}.get(priority, 5)
    
    # Adjust for file size (smaller files get slight priority)
    if file_size < 1024 * 1024:  # < 1MB
        base_score += 1
    elif file_size > 10 * 1024 * 1024:  # > 10MB
        base_score -= 1
    
    # Premium users get higher priority
    if user_id != 'anonymous':
        base_score += 2
    
    return max(1, min(10, base_score))


def _detect_document_type_from_filename(filename: str) -> Optional[str]:
    """Detect document type from filename patterns"""
    
    filename_lower = filename.lower()
    
    type_patterns = {
        'lease': ['lease', 'rental', 'rent'],
        'employment': ['employment', 'job', 'offer', 'contract', 'employee'],
        'nda': ['nda', 'confidentiality', 'non-disclosure'],
        'loan': ['loan', 'mortgage', 'credit'],
        'purchase': ['purchase', 'sale', 'buy', 'acquisition']
    }
    
    for doc_type, patterns in type_patterns.items():
        if any(pattern in filename_lower for pattern in patterns):
            return doc_type
    
    return None


def _calculate_reading_time(word_count: int, words_per_minute: int = 200) -> str:
    """Calculate estimated reading time"""
    
    if word_count == 0:
        return "0 minutes"
    
    minutes = max(1, word_count // words_per_minute)
    
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    else:
        hours = minutes // 60
        remaining_minutes = minutes % 60
        return f"{hours}h {remaining_minutes}m"


def _determine_risk_level(risk_score: float) -> str:
    """Enhanced risk level determination"""
    
    if risk_score >= 8.0:
        return 'critical'
    elif risk_score >= 6.5:
        return 'high'
    elif risk_score >= 4.0:
        return 'medium'
    elif risk_score >= 2.0:
        return 'low'
    else:
        return 'minimal'


def _calculate_overall_confidence(analysis_result: Dict, extraction_result: Dict) -> float:
    """Calculate overall confidence score"""
    
    analysis_confidence = analysis_result.get('confidence_score', 0.8)
    extraction_confidence = extraction_result.get('confidence', 0.9)
    
    # Weighted average
    return (analysis_confidence * 0.7) + (extraction_confidence * 0.3)
