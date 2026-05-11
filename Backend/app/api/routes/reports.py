"""
Enhanced PDF Report generation API routes
Production-ready with advanced templates, customization, and analytics
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional, List, Dict, Any
import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json

from config.logging import get_logger, log_api_request
from config.settings import get_settings
from app.models.requests import ReportRequest, CustomReportRequest, BulkReportRequest
from app.models.responses import (
    ReportGenerationResponse, ReportStatusResponse, ReportLibraryResponse,
    ReportTemplateResponse, ReportAnalyticsResponse, ErrorResponse
)
from app.services.report_generator import ReportGenerator
from app.services.template_manager import TemplateManager
from app.services.storage_manager import StorageManager
from app.utils.validators import validate_report_request, validate_template_config
from app.utils.helpers import generate_request_id, sanitize_filename
from app.utils.formatters import format_file_size, format_duration
from app.core.exceptions import ReportGenerationError, ValidationError, TemplateError
from app.core.dependencies import get_current_user, rate_limit

settings = get_settings()
logger = get_logger(__name__)
router = APIRouter(prefix="/reports", tags=["reports"])

# Service instances
report_generator = ReportGenerator()
template_manager = TemplateManager()
storage_manager = StorageManager()

# External reference to documents
from app.api.routes.documents import documents_db

# Enhanced report storage with metadata
reports_db = {}
report_generation_status = {}
report_analytics = {'total_generated': 0, 'total_size': 0, 'templates_used': {}}


@router.post("/generate", response_model=ReportGenerationResponse)
async def generate_report(
    background_tasks: BackgroundTasks,
    report_request: ReportRequest,
    current_user: Optional[Dict] = Depends(get_current_user),
    _rate_limit: None = Depends(rate_limit("report_generation"))
):
    """
    Generate comprehensive PDF report with advanced customization options
    
    - **document_id**: Document identifier
    - **template**: Report template (executive, detailed, compliance, comparison)
    - **language**: Report language
    - **include_sections**: Sections to include in report
    - **custom_branding**: Custom branding options
    - **export_format**: Output format (pdf, docx, html)
    """
    
    request_id = generate_request_id()
    start_time = datetime.now(timezone.utc)
    
    try:
        logger.info(
            f"Report generation request",
            extra={
                'request_id': request_id,
                'document_id': report_request.document_id,
                'template': report_request.template,
                'language': report_request.language,
                'user_id': current_user.get('user_id') if current_user else None
            }
        )
        
        # Enhanced validation
        validation_result = await _validate_report_request_enhanced(report_request, current_user)
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
        document_data = await _get_and_validate_document_for_report(report_request.document_id, current_user)
        
        # Check report generation limits
        user_id = current_user.get('user_id') if current_user else 'anonymous'
        limit_check = await _check_report_generation_limits(user_id)
        if not limit_check['allowed']:
            raise HTTPException(
                status_code=429,
                detail={
                    'error': 'REPORT_LIMIT_EXCEEDED',
                    'message': limit_check['message'],
                    'limit_info': limit_check['limit_info'],
                    'request_id': request_id
                }
            )
        
        # Generate unique report ID
        report_id = str(uuid.uuid4())
        
        # Validate template and get configuration
        template_config = await template_manager.get_template_config(
            report_request.template,
            report_request.language
        )
        
        if not template_config:
            raise HTTPException(
                status_code=400,
                detail={
                    'error': 'TEMPLATE_NOT_FOUND',
                    'message': f'Template "{report_request.template}" not found for language "{report_request.language}"',
                    'available_templates': await template_manager.get_available_templates(report_request.language),
                    'request_id': request_id
                }
            )
        
        # Estimate generation time and cost
        estimated_time = _estimate_report_generation_time(
            document_data,
            report_request.template,
            len(report_request.include_sections)
        )
        
        # Store report info with enhanced metadata
        report_info = {
            'report_id': report_id,
            'document_id': report_request.document_id,
            'template': report_request.template,
            'language': report_request.language,
            'export_format': report_request.export_format,
            'include_sections': report_request.include_sections,
            'custom_branding': report_request.custom_branding,
            'status': 'initializing',
            'created_at': datetime.now(timezone.utc),
            'progress': 0,
            'user_id': user_id,
            'request_id': request_id,
            'estimated_time': estimated_time,
            'priority': _calculate_report_priority(current_user),
            'template_config': template_config,
            'metadata': {
                'document_title': document_data.get('title', document_data.get('filename')),
                'document_type': document_data.get('document_type'),
                'generation_method': 'ai_enhanced',
                'version': '2.0',
                'total_sections': len(report_request.include_sections)
            }
        }
        
        reports_db[report_id] = report_info
        report_generation_status[report_id] = {
            'stage': 'initializing',
            'progress': 0,
            'started_at': datetime.now(timezone.utc)
        }
        
        # Start background report generation
        background_tasks.add_task(
            generate_report_workflow,
            report_id,
            document_data,
            report_request,
            template_config,
            request_id
        )
        
        # Log API request
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        background_tasks.add_task(
            log_api_request,
            'POST',
            '/reports/generate',
            200,
            processing_time,
            user_id,
            report_request.document_id,
            request_id
        )
        
        return ReportGenerationResponse(
            report_id=report_id,
            status="initializing",
            estimated_time=format_duration(estimated_time),
            template=report_request.template,
            language=report_request.language,
            export_format=report_request.export_format,
            sections_included=len(report_request.include_sections),
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except ReportGenerationError as e:
        logger.error(f"Report generation error: {str(e)}", extra={'request_id': request_id})
        raise HTTPException(
            status_code=400,
            detail={
                'error': 'REPORT_GENERATION_ERROR',
                'message': str(e),
                'request_id': request_id
            }
        )
    except Exception as e:
        logger.error(f"Error initiating report generation: {str(e)}", extra={'request_id': request_id})
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'INTERNAL_ERROR',
                'message': "Failed to initiate report generation",
                'request_id': request_id
            }
        )


@router.get("/{report_id}/status", response_model=ReportStatusResponse)
async def get_report_status(
    report_id: str,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Get detailed report generation status with real-time progress
    """
    
    try:
        if report_id not in reports_db:
            raise HTTPException(
                status_code=404,
                detail="Report not found"
            )
        
        report_info = reports_db[report_id]
        status_info = report_generation_status.get(report_id, {})
        
        # Check user access
        if not await _check_report_access(report_id, current_user):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this report"
            )
        
        # Calculate processing time
        processing_time = None
        if status_info.get('started_at'):
            if report_info['status'] == 'completed':
                processing_time = (report_info.get('completed_at', datetime.now(timezone.utc)) - status_info['started_at']).total_seconds()
            else:
                processing_time = (datetime.now(timezone.utc) - status_info['started_at']).total_seconds()
        
        # Get stage details
        stage_details = {
            'initializing': 'Setting up report generation environment',
            'analyzing': 'Analyzing document content and structure',
            'templating': 'Applying template and formatting content',
            'rendering': 'Rendering final report document',
            'finalizing': 'Finalizing and preparing download',
            'completed': 'Report generation completed successfully',
            'failed': 'Report generation failed'
        }
        
        current_stage = status_info.get('stage', 'unknown')
        
        return ReportStatusResponse(
            report_id=report_id,
            status=report_info['status'],
            progress=status_info.get('progress', 0),
            current_stage=current_stage,
            stage_description=stage_details.get(current_stage, 'Processing...'),
            processing_time=format_duration(processing_time) if processing_time else None,
            estimated_remaining=status_info.get('estimated_remaining'),
            error_message=report_info.get('error_message'),
            file_size=report_info.get('file_size'),
            file_size_formatted=format_file_size(report_info.get('file_size', 0)),
            download_available=report_info.get('status') == 'completed',
            expires_at=report_info.get('expires_at')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting report status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get report status"
        )


@router.get("/{report_id}", response_model=Dict[str, Any])
async def get_report_details(
    report_id: str,
    include_metadata: bool = Query(True, description="Include generation metadata"),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Get comprehensive report details and information
    """
    
    try:
        if report_id not in reports_db:
            raise HTTPException(
                status_code=404,
                detail="Report not found"
            )
        
        # Check access
        if not await _check_report_access(report_id, current_user):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this report"
            )
        
        report_info = reports_db[report_id]
        
        response_data = {
            'report_id': report_id,
            'document_id': report_info['document_id'],
            'template': report_info['template'],
            'language': report_info['language'],
            'export_format': report_info['export_format'],
            'status': report_info['status'],
            'created_at': report_info['created_at'].isoformat(),
            'completed_at': report_info.get('completed_at'),
            'download_url': report_info.get('download_url'),
            'file_size': report_info.get('file_size'),
            'file_size_formatted': format_file_size(report_info.get('file_size', 0)),
            'expires_at': report_info.get('expires_at'),
            'sections_included': report_info.get('include_sections', []),
            'page_count': report_info.get('page_count'),
            'generation_time': report_info.get('generation_time')
        }
        
        if include_metadata and report_info.get('metadata'):
            response_data['metadata'] = report_info['metadata']
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting report details: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get report details"
        )


@router.get("/{report_id}/download")
async def download_report(
    report_id: str,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Download generated report file with streaming support
    """
    
    try:
        if report_id not in reports_db:
            raise HTTPException(
                status_code=404,
                detail="Report not found"
            )
        
        # Check access
        if not await _check_report_access(report_id, current_user):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this report"
            )
        
        report_info = reports_db[report_id]
        
        if report_info['status'] != 'completed':
            raise HTTPException(
                status_code=400,
                detail=f"Report not ready for download. Status: {report_info['status']}"
            )
        
        # Check if report has expired
        if report_info.get('expires_at') and datetime.now(timezone.utc) > report_info['expires_at']:
            raise HTTPException(
                status_code=410,
                detail="Report has expired and is no longer available for download"
            )
        
        # Get report file from storage
        report_content = await report_generator.get_report_file(report_id)
        
        if not report_content:
            raise HTTPException(
                status_code=404,
                detail="Report file not found in storage"
            )
        
        # Generate safe filename
        document_title = report_info.get('metadata', {}).get('document_title', 'document')
        safe_filename = sanitize_filename(
            f"{document_title}_{report_info['template']}_report.{report_info['export_format']}"
        )
        
        # Determine media type
        media_types = {
            'pdf': 'application/pdf',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'html': 'text/html'
        }
        media_type = media_types.get(report_info['export_format'], 'application/octet-stream')
        
        # Return streaming response
        def generate_report_stream():
            chunk_size = 8192
            for i in range(0, len(report_content), chunk_size):
                yield report_content[i:i + chunk_size]
        
        return StreamingResponse(
            generate_report_stream(),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={safe_filename}",
                "Content-Length": str(len(report_content))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading report: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to download report"
        )


@router.get("/", response_model=ReportLibraryResponse)
async def get_user_reports(
    current_user: Optional[Dict] = Depends(get_current_user),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=50, description="Items per page"),
    document_id: Optional[str] = Query(None, description="Filter by document"),
    template: Optional[str] = Query(None, description="Filter by template"),
    status: Optional[str] = Query(None, description="Filter by status"),
    language: Optional[str] = Query(None, description="Filter by language"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order")
):
    """
    Get user's report library with advanced filtering and pagination
    """
    
    try:
        user_id = current_user.get('user_id', 'anonymous') if current_user else 'anonymous'
        
        # Filter reports
        filtered_reports = []
        
        for report_id, report_info in reports_db.items():
            # User access check
            if not await _check_report_access(report_id, current_user):
                continue
            
            # Apply filters
            if document_id and report_info.get('document_id') != document_id:
                continue
            if template and report_info.get('template') != template:
                continue
            if status and report_info.get('status') != status:
                continue
            if language and report_info.get('language') != language:
                continue
            
            # Create report summary
            report_summary = {
                'report_id': report_id,
                'document_id': report_info.get('document_id'),
                'document_title': report_info.get('metadata', {}).get('document_title', ''),
                'template': report_info.get('template'),
                'language': report_info.get('language'),
                'export_format': report_info.get('export_format'),
                'status': report_info.get('status'),
                'created_at': report_info.get('created_at'),
                'completed_at': report_info.get('completed_at'),
                'file_size': report_info.get('file_size'),
                'file_size_formatted': format_file_size(report_info.get('file_size', 0)),
                'page_count': report_info.get('page_count'),
                'download_available': report_info.get('status') == 'completed',
                'expires_at': report_info.get('expires_at'),
                'generation_time': report_info.get('generation_time')
            }
            
            filtered_reports.append(report_summary)
        
        # Sort reports
        reverse_sort = sort_order.lower() == 'desc'
        
        if sort_by == 'created_at':
            filtered_reports.sort(key=lambda x: x['created_at'] or datetime.min, reverse=reverse_sort)
        elif sort_by == 'file_size':
            filtered_reports.sort(key=lambda x: x['file_size'] or 0, reverse=reverse_sort)
        elif sort_by == 'template':
            filtered_reports.sort(key=lambda x: x['template'], reverse=reverse_sort)
        
        # Apply pagination
        total_count = len(filtered_reports)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_reports = filtered_reports[start_idx:end_idx]
        
        # Calculate statistics
        stats = {
            'total_reports': total_count,
            'by_template': {},
            'by_status': {},
            'by_language': {},
            'total_size': 0,
            'total_pages': 0
        }
        
        for report in filtered_reports:
            template = report['template']
            status = report['status']
            language = report['language']
            
            stats['by_template'][template] = stats['by_template'].get(template, 0) + 1
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
            stats['by_language'][language] = stats['by_language'].get(language, 0) + 1
            stats['total_size'] += report['file_size'] or 0
            stats['total_pages'] += report['page_count'] or 0
        
        return ReportLibraryResponse(
            reports=paginated_reports,
            total_count=total_count,
            page=page,
            per_page=per_page,
            total_pages=(total_count + per_page - 1) // per_page,
            has_next=end_idx < total_count,
            has_prev=page > 1,
            statistics=stats
        )
        
    except Exception as e:
        logger.error(f"Error getting report library: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get report library"
        )


@router.post("/custom", response_model=ReportGenerationResponse)
async def generate_custom_report(
    background_tasks: BackgroundTasks,
    custom_request: CustomReportRequest,
    current_user: Optional[Dict] = Depends(get_current_user),
    _rate_limit: None = Depends(rate_limit("report_generation"))
):
    """
    Generate custom report with user-defined template and sections
    """
    
    request_id = generate_request_id()
    
    try:
        # Validate custom template configuration
        template_validation = await template_manager.validate_custom_template(custom_request.template_config)
        if not template_validation['valid']:
            raise HTTPException(
                status_code=400,
                detail={
                    'error': 'INVALID_TEMPLATE',
                    'message': template_validation['error'],
                    'suggestions': template_validation.get('suggestions', [])
                }
            )
        
        # Check document access
        document_data = await _get_and_validate_document_for_report(custom_request.document_id, current_user)
        
        # Generate report with custom template
        report_id = str(uuid.uuid4())
        
        # Convert custom request to standard format
        report_request = ReportRequest(
            document_id=custom_request.document_id,
            template="custom",
            language=custom_request.language,
            export_format=custom_request.export_format,
            include_sections=custom_request.sections,
            custom_branding=custom_request.branding
        )
        
        # Store custom template configuration
        reports_db[report_id] = {
            'report_id': report_id,
            'document_id': custom_request.document_id,
            'template': 'custom',
            'language': custom_request.language,
            'export_format': custom_request.export_format,
            'status': 'initializing',
            'created_at': datetime.now(timezone.utc),
            'user_id': current_user.get('user_id') if current_user else 'anonymous',
            'request_id': request_id,
            'custom_template': custom_request.template_config,
            'include_sections': custom_request.sections,
            'custom_branding': custom_request.branding
        }
        
        # Start generation
        background_tasks.add_task(
            generate_custom_report_workflow,
            report_id,
            document_data,
            custom_request,
            request_id
        )
        
        return ReportGenerationResponse(
            report_id=report_id,
            status="initializing",
            estimated_time="2-5 minutes",
            template="custom",
            language=custom_request.language,
            export_format=custom_request.export_format,
            sections_included=len(custom_request.sections),
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating custom report: {str(e)}", extra={'request_id': request_id})
        raise HTTPException(
            status_code=500,
            detail="Failed to generate custom report"
        )


@router.post("/bulk", response_model=Dict[str, Any])
async def generate_bulk_reports(
    background_tasks: BackgroundTasks,
    bulk_request: BulkReportRequest,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Generate reports for multiple documents in batch
    """
    
    request_id = generate_request_id()
    
    try:
        # Validate document access for all documents
        valid_documents = []
        
        for document_id in bulk_request.document_ids:
            try:
                document_data = await _get_and_validate_document_for_report(document_id, current_user)
                valid_documents.append((document_id, document_data))
            except HTTPException as e:
                logger.warning(f"Skipping document {document_id}: {e.detail}")
        
        if not valid_documents:
            raise HTTPException(
                status_code=400,
                detail="No valid documents found for report generation"
            )
        
        # Start bulk generation
        background_tasks.add_task(
            generate_bulk_reports_workflow,
            valid_documents,
            bulk_request,
            current_user,
            request_id
        )
        
        return {
            "success": True,
            "message": f"Bulk report generation started for {len(valid_documents)} documents",
            "batch_id": request_id,
            "documents_count": len(valid_documents),
            "template": bulk_request.template,
            "estimated_time": format_duration(len(valid_documents) * 60)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting bulk report generation: {str(e)}", extra={'request_id': request_id})
        raise HTTPException(
            status_code=500,
            detail="Failed to start bulk report generation"
        )


@router.get("/templates/available", response_model=ReportTemplateResponse)
async def get_available_templates(
    language: str = Query("en", description="Language for templates"),
    category: Optional[str] = Query(None, description="Template category filter")
):
    """
    Get available report templates with detailed information
    """
    
    try:
        templates = await template_manager.get_available_templates_detailed(language, category)
        
        return ReportTemplateResponse(
            templates=templates,
            total_count=len(templates),
            supported_languages=await template_manager.get_supported_languages(),
            categories=await template_manager.get_template_categories()
        )
        
    except Exception as e:
        logger.error(f"Error getting available templates: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get available templates"
        )


@router.delete("/{report_id}")
async def delete_report(
    report_id: str,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Delete a generated report and its file
    """
    
    try:
        if report_id not in reports_db:
            raise HTTPException(
                status_code=404,
                detail="Report not found"
            )
        
        # Check access
        if not await _check_report_access(report_id, current_user):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this report"
            )
        
        report_info = reports_db[report_id]
        
        # Delete report file from storage
        try:
            await report_generator.delete_report_file(report_id)
        except Exception as e:
            logger.warning(f"Failed to delete report file: {str(e)}")
        
        # Remove from local storage
        del reports_db[report_id]
        if report_id in report_generation_status:
            del report_generation_status[report_id]
        
        logger.info(f"Report deleted successfully", extra={'report_id': report_id})
        
        return {
            "success": True,
            "message": "Report deleted successfully",
            "deleted_at": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting report: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete report"
        )


@router.get("/analytics/overview")
async def get_report_analytics(
    current_user: Optional[Dict] = Depends(get_current_user),
    time_range: str = Query("7d", description="Time range (1d, 7d, 30d)")
):
    """
    Get comprehensive report analytics and statistics
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
        
        # Filter reports by time range and user
        user_reports = []
        for report_id, report_info in reports_db.items():
            if await _check_report_access(report_id, current_user):
                if report_info.get('created_at', datetime.min.replace(tzinfo=timezone.utc)) >= start_date:
                    user_reports.append(report_info)
        
        # Calculate analytics
        analytics = {
            'time_range': time_range,
            'total_reports': len(user_reports),
            'by_template': {},
            'by_status': {},
            'by_language': {},
            'by_format': {},
            'generation_stats': {
                'total_size': 0,
                'total_pages': 0,
                'avg_generation_time': 0,
                'success_rate': 0
            },
            'usage_trends': []
        }
        
        total_size = 0
        total_pages = 0
        total_generation_time = 0
        completed_count = 0
        
        for report in user_reports:
            # Template distribution
            template = report.get('template', 'unknown')
            analytics['by_template'][template] = analytics['by_template'].get(template, 0) + 1
            
            # Status distribution
            status = report.get('status', 'unknown')
            analytics['by_status'][status] = analytics['by_status'].get(status, 0) + 1
            
            # Language distribution
            language = report.get('language', 'en')
            analytics['by_language'][language] = analytics['by_language'].get(language, 0) + 1
            
            # Format distribution
            export_format = report.get('export_format', 'pdf')
            analytics['by_format'][export_format] = analytics['by_format'].get(export_format, 0) + 1
            
            # Generation stats
            if status == 'completed':
                completed_count += 1
                size = report.get('file_size', 0)
                pages = report.get('page_count', 0)
                gen_time = report.get('generation_time', 0)
                
                total_size += size
                total_pages += pages
                total_generation_time += gen_time
        
        # Calculate generation statistics
        if user_reports:
            analytics['generation_stats'] = {
                'total_size': total_size,
                'total_size_formatted': format_file_size(total_size),
                'total_pages': total_pages,
                'avg_generation_time': total_generation_time / completed_count if completed_count > 0 else 0,
                'avg_pages_per_report': total_pages / completed_count if completed_count > 0 else 0,
                'success_rate': (completed_count / len(user_reports)) * 100
            }
        
        return ReportAnalyticsResponse(
            analytics=analytics,
            generated_at=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error getting report analytics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get report analytics"
        )


# Background workflows

async def generate_report_workflow(
    report_id: str,
    document_data: Dict[str, Any],
    report_request: ReportRequest,
    template_config: Dict[str, Any],
    request_id: str
):
    """
    Enhanced report generation workflow with detailed tracking
    """
    
    start_time = datetime.now(timezone.utc)
    
    try:
        logger.info(f"Starting report generation workflow", extra={
            'report_id': report_id,
            'request_id': request_id,
            'template': report_request.template
        })
        
        # Stage 1: Content Analysis and Preparation
        await _update_report_status(report_id, 'analyzing', 20, 'Analyzing document content')
        
        # Prepare report content based on included sections
        report_content = await report_generator.prepare_report_content(
            document_data,
            report_request.include_sections,
            report_request.language
        )
        
        # Stage 2: Template Application
        await _update_report_status(report_id, 'templating', 50, 'Applying template and formatting')
        
        # Apply template with custom branding if provided
        formatted_content = await report_generator.apply_template(
            report_content,
            template_config,
            report_request.custom_branding
        )
        
        # Stage 3: Report Rendering
        await _update_report_status(report_id, 'rendering', 80, 'Rendering final report')
        
        # Generate the actual report file
        report_result = await report_generator.render_report(
            formatted_content,
            report_request.export_format,
            report_id
        )
        
        # Stage 4: Finalization
        await _update_report_status(report_id, 'finalizing', 95, 'Finalizing report')
        
        # Calculate expiration date
        expires_at = datetime.now(timezone.utc) + timedelta(days=7)  # 7 day retention
        
        # Update report info with results
        reports_db[report_id].update({
            'status': 'completed',
            'progress': 100,
            'download_url': report_result['download_url'],
            'file_size': report_result['file_size'],
            'page_count': report_result.get('page_count', 0),
            'completed_at': datetime.now(timezone.utc),
            'expires_at': expires_at,
            'generation_time': (datetime.now(timezone.utc) - start_time).total_seconds()
        })
        
        report_generation_status[report_id] = {
            'stage': 'completed',
            'progress': 100,
            'completed_at': datetime.now(timezone.utc)
        }
        
        # Update analytics
        report_analytics['total_generated'] += 1
        report_analytics['total_size'] += report_result['file_size']
        
        template = report_request.template
        report_analytics['templates_used'][template] = report_analytics['templates_used'].get(template, 0) + 1
        
        logger.info(f"Report generation completed successfully", extra={
            'report_id': report_id,
            'file_size': report_result['file_size'],
            'page_count': report_result.get('page_count', 0),
            'generation_time': (datetime.now(timezone.utc) - start_time).total_seconds()
        })
        
    except Exception as e:
        logger.error(f"Error in report generation workflow: {str(e)}", extra={
            'report_id': report_id,
            'request_id': request_id
        })
        
        reports_db[report_id].update({
            'status': 'failed',
            'error_message': str(e),
            'failed_at': datetime.now(timezone.utc)
        })
        
        report_generation_status[report_id] = {
            'stage': 'failed',
            'progress': 0,
            'error': str(e)
        }


async def generate_custom_report_workflow(
    report_id: str,
    document_data: Dict[str, Any],
    custom_request: CustomReportRequest,
    request_id: str
):
    """
    Generate report with custom template configuration
    """
    
    try:
        logger.info(f"Starting custom report generation", extra={
            'report_id': report_id,
            'request_id': request_id
        })
        
        # Use custom template manager
        custom_template_config = await template_manager.create_custom_template(
            custom_request.template_config
        )
        
        # Create standard report request
        report_request = ReportRequest(
            document_id=custom_request.document_id,
            template="custom",
            language=custom_request.language,
            export_format=custom_request.export_format,
            include_sections=custom_request.sections,
            custom_branding=custom_request.branding
        )
        
        # Use standard workflow with custom template
        await generate_report_workflow(
            report_id,
            document_data,
            report_request,
            custom_template_config,
            request_id
        )
        
    except Exception as e:
        logger.error(f"Error in custom report generation: {str(e)}")
        reports_db[report_id].update({
            'status': 'failed',
            'error_message': f"Custom report generation failed: {str(e)}"
        })


async def generate_bulk_reports_workflow(
    documents: List[Tuple[str, Dict[str, Any]]],
    bulk_request: BulkReportRequest,
    user_context: Optional[Dict],
    batch_id: str
):
    """
    Generate reports for multiple documents
    """
    
    try:
        logger.info(f"Starting bulk report generation", extra={
            'batch_id': batch_id,
            'document_count': len(documents)
        })
        
        generated_reports = []
        
        for document_id, document_data in documents:
            try:
                # Create individual report request
                report_request = ReportRequest(
                    document_id=document_id,
                    template=bulk_request.template,
                    language=bulk_request.language,
                    export_format=bulk_request.export_format,
                    include_sections=bulk_request.include_sections or ['summary', 'risks', 'recommendations']
                )
                
                report_id = str(uuid.uuid4())
                
                # Initialize report info
                report_info = {
                    'report_id': report_id,
                    'document_id': document_id,
                    'template': bulk_request.template,
                    'language': bulk_request.language,
                    'export_format': bulk_request.export_format,
                    'status': 'initializing',
                    'created_at': datetime.now(timezone.utc),
                    'user_id': user_context.get('user_id') if user_context else 'anonymous',
                    'batch_id': batch_id,
                    'metadata': {
                        'document_title': document_data.get('title', document_data.get('filename')),
                        'batch_generation': True
                    }
                }
                
                reports_db[report_id] = report_info
                
                # Get template config
                template_config = await template_manager.get_template_config(
                    bulk_request.template,
                    bulk_request.language
                )
                
                # Generate report
                await generate_report_workflow(
                    report_id,
                    document_data,
                    report_request,
                    template_config,
                    batch_id
                )
                
                generated_reports.append({
                    'document_id': document_id,
                    'report_id': report_id,
                    'status': reports_db[report_id].get('status')
                })
                
            except Exception as e:
                logger.error(f"Failed to generate report for document {document_id}: {str(e)}")
                generated_reports.append({
                    'document_id': document_id,
                    'report_id': None,
                    'status': 'failed',
                    'error': str(e)
                })
        
        logger.info(f"Bulk report generation completed", extra={
            'batch_id': batch_id,
            'total_documents': len(documents),
            'successful': len([r for r in generated_reports if r['status'] == 'completed'])
        })
        
    except Exception as e:
        logger.error(f"Error in bulk report generation: {str(e)}", extra={'batch_id': batch_id})


# Helper functions

async def _validate_report_request_enhanced(report_request: ReportRequest, user_context: Optional[Dict]) -> Dict[str, Any]:
    """Enhanced report request validation"""
    
    # Basic validation
    basic_validation = validate_report_request(report_request)
    if not basic_validation['valid']:
        return basic_validation
    
    # Check template availability
    available_templates = await template_manager.get_available_templates(report_request.language)
    if report_request.template not in available_templates:
        return {
            'valid': False,
            'error': f'Template "{report_request.template}" not available for language "{report_request.language}"',
            'suggestions': [f'Available templates: {", ".join(available_templates)}']
        }
    
    # Validate sections
    valid_sections = ['summary', 'risks', 'recommendations', 'obligations', 'rights', 'financial', 'compliance']
    invalid_sections = [s for s in report_request.include_sections if s not in valid_sections]
    
    if invalid_sections:
        return {
            'valid': False,
            'error': f'Invalid sections: {", ".join(invalid_sections)}',
            'suggestions': [f'Valid sections: {", ".join(valid_sections)}']
        }
    
    return {'valid': True}


async def _get_and_validate_document_for_report(document_id: str, user_context: Optional[Dict]) -> Dict[str, Any]:
    """Get and validate document for report generation"""
    
    if document_id not in documents_db:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )
    
    document_data = documents_db[document_id]
    
    if document_data['status'] != 'completed':
        raise HTTPException(
            status_code=400,
            detail=f"Document not ready for report generation. Status: {document_data['status']}"
        )
    
    # Check access
    from app.api.routes.documents import _check_document_access
    if not await _check_document_access(document_id, user_context):
        raise HTTPException(
            status_code=403,
            detail="Access denied to this document"
        )
    
    return document_data


async def _check_report_generation_limits(user_id: str) -> Dict[str, Any]:
    """Check report generation limits for user"""
    
    # Count user's reports in the last 24 hours
    now = datetime.now(timezone.utc)
    yesterday = now - timedelta(days=1)
    
    user_reports_today = [
        r for r in reports_db.values()
        if r.get('user_id') == user_id and r.get('created_at', datetime.min.replace(tzinfo=timezone.utc)) >= yesterday
    ]
    
    daily_limit = 15  # Configurable
    
    if len(user_reports_today) >= daily_limit:
        return {
            'allowed': False,
            'message': f'Daily report generation limit of {daily_limit} exceeded',
            'limit_info': {
                'daily_limit': daily_limit,
                'used_today': len(user_reports_today),
                'resets_at': now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            }
        }
    
    return {'allowed': True}


async def _check_report_access(report_id: str, user_context: Optional[Dict]) -> bool:
    """Check if user has access to report"""
    
    if report_id not in reports_db:
        return False
    
    report_info = reports_db[report_id]
    
    # Public access for demo
    if not user_context:
        return True
    
    # Owner access
    user_id = user_context.get('user_id', 'anonymous')
    if report_info.get('user_id') == user_id:
        return True
    
    # Admin access
    if user_context.get('role') == 'admin':
        return True
    
    return False


async def _update_report_status(report_id: str, stage: str, progress: int, description: str):
    """Update report generation status"""
    
    report_generation_status[report_id] = {
        'stage': stage,
        'progress': progress,
        'description': description,
        'last_updated': datetime.now(timezone.utc)
    }
    
    reports_db[report_id]['status'] = stage
    reports_db[report_id]['progress'] = progress


def _calculate_report_priority(user_context: Optional[Dict]) -> int:
    """Calculate report generation priority"""
    
    base_priority = 5
    
    if not user_context:
        return base_priority
    
    # Premium users get higher priority
    if user_context.get('subscription_tier') == 'premium':
        base_priority += 3
    elif user_context.get('subscription_tier') == 'pro':
        base_priority += 2
    
    return min(10, base_priority)


def _estimate_report_generation_time(document_data: Dict[str, Any], template: str, section_count: int) -> int:
    """Estimate report generation time"""
    
    # Base time based on document size and complexity
    word_count = document_data.get('word_count', 1000)
    base_time = max(30, (word_count / 1000) * 20)  # 20 seconds per 1000 words
    
    # Template complexity factor
    template_multipliers = {
        'executive': 1.0,
        'detailed': 1.5,
        'compliance': 1.8,
        'comparison': 2.0,
        'custom': 2.5
    }
    base_time *= template_multipliers.get(template, 1.2)
    
    # Section count factor
    base_time += section_count * 5  # 5 seconds per section
    
    return int(base_time)
