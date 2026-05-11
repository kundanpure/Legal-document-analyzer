"""
Enhanced database model utilities and helpers for LegalMind AI
Provides database model validation, serialization, and utility functions
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union
from enum import Enum

from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from app.core.database import (
    Document, User, ChatSession, ChatMessage, Report, 
    VoiceSummary, DocumentProcessingLog, SystemMetric, AuditLog
)
from config.logging import get_logger

logger = get_logger(__name__)


class DatabaseOperationResult(BaseModel):
    """Result of database operation"""
    success: bool
    operation: str
    affected_rows: int = 0
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ModelSerializer:
    """Utility class for serializing database models to dictionaries"""
    
    @staticmethod
    def serialize_document(document: Document) -> Dict[str, Any]:
        """Serialize Document model to dictionary"""
        return {
            'id': document.id,
            'filename': document.filename,
            'original_filename': document.original_filename,
            'file_size': document.file_size,
            'file_size_formatted': document.file_size_formatted,
            'content_type': document.content_type,
            'status': document.status,
            'progress': document.progress,
            'upload_date': document.upload_date.isoformat() if document.upload_date else None,
            'completed_date': document.completed_date.isoformat() if document.completed_date else None,
            'document_type': document.document_type,
            'title': document.title,
            'summary': document.summary,
            'executive_summary': document.executive_summary,
            'risk_score': document.risk_score,
            'risk_level': document.risk_level,
            'risk_categories': document.risk_categories,
            'confidence_score': document.confidence_score,
            'page_count': document.page_count,
            'word_count': document.word_count,
            'estimated_reading_time': document.estimated_reading_time,
            'key_risks': document.key_risks,
            'recommendations': document.recommendations,
            'user_obligations': document.user_obligations,
            'user_rights': document.user_rights,
            'financial_implications': document.financial_implications,
            'fairness_score': document.fairness_score,
            'language': document.language,
            'tags': document.tags,
            'user_id': document.user_id
        }
    
    @staticmethod
    def serialize_chat_session(session: ChatSession) -> Dict[str, Any]:
        """Serialize ChatSession model to dictionary"""
        return {
            'id': session.id,
            'document_id': session.document_id,
            'user_id': session.user_id,
            'language': session.language,
            'is_active': session.is_active,
            'created_at': session.created_at.isoformat() if session.created_at else None,
            'last_activity': session.last_activity.isoformat() if session.last_activity else None,
            'message_count': session.message_count,
            'preferences': session.preferences,
            'capabilities': session.capabilities
        }
    
    @staticmethod
    def serialize_chat_message(message: ChatMessage) -> Dict[str, Any]:
        """Serialize ChatMessage model to dictionary"""
        return {
            'id': message.id,
            'session_id': message.session_id,
            'message_type': message.message_type,
            'content': message.content,
            'language': message.language,
            'timestamp': message.timestamp.isoformat() if message.timestamp else None,
            'confidence': message.confidence,
            'processing_time': message.processing_time,
            'citations': message.citations,
            'suggestions': message.suggestions,
            'user_rating': message.user_rating
        }
    
    @staticmethod
    def serialize_report(report: Report) -> Dict[str, Any]:
        """Serialize Report model to dictionary"""
        return {
            'id': report.id,
            'document_id': report.document_id,
            'user_id': report.user_id,
            'template': report.template,
            'language': report.language,
            'export_format': report.export_format,
            'status': report.status,
            'created_at': report.created_at.isoformat() if report.created_at else None,
            'completed_at': report.completed_at.isoformat() if report.completed_at else None,
            'download_url': report.download_url,
            'file_size': report.file_size,
            'file_size_formatted': report.file_size_formatted,
            'page_count': report.page_count,
            'generation_time': report.generation_time,
            'include_sections': report.include_sections
        }
    
    @staticmethod
    def serialize_voice_summary(voice: VoiceSummary) -> Dict[str, Any]:
        """Serialize VoiceSummary model to dictionary"""
        return {
            'id': voice.id,
            'document_id': voice.document_id,
            'user_id': voice.user_id,
            'language': voice.language,
            'voice_type': voice.voice_type,
            'status': voice.status,
            'created_at': voice.created_at.isoformat() if voice.created_at else None,
            'completed_at': voice.completed_at.isoformat() if voice.completed_at else None,
            'audio_url': voice.audio_url,
            'duration': voice.duration,
            'file_size': voice.file_size,
            'file_size_formatted': voice.file_size_formatted,
            'content_type': voice.content_type,
            'transcript': voice.transcript
        }
    
    @staticmethod
    def serialize_user(user: User) -> Dict[str, Any]:
        """Serialize User model to dictionary (excluding sensitive data)"""
        return {
            'id': user.id,
            'email': user.email,
            'username': user.username,
            'full_name': user.full_name,
            'organization': user.organization,
            'role': user.role,
            'subscription_tier': user.subscription_tier,
            'is_active': user.is_active,
            'is_verified': user.is_verified,
            'created_at': user.created_at.isoformat() if user.created_at else None,
            'last_login': user.last_login.isoformat() if user.last_login else None,
            'documents_uploaded': user.documents_uploaded,
            'reports_generated': user.reports_generated,
            'voice_summaries_created': user.voice_summaries_created,
            'preferences': user.preferences
        }


class DatabaseQueryHelper:
    """Helper class for common database queries"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def get_user_documents(self, user_id: str, status: Optional[str] = None, 
                          limit: int = 10, offset: int = 0) -> List[Document]:
        """Get documents for a user with optional filtering"""
        query = self.session.query(Document).filter(Document.user_id == user_id)
        
        if status:
            query = query.filter(Document.status == status)
        
        return query.order_by(Document.upload_date.desc()).offset(offset).limit(limit).all()
    
    def get_document_with_relations(self, document_id: str, user_id: str) -> Optional[Document]:
        """Get document with related data"""
        return self.session.query(Document)\
            .filter(Document.id == document_id)\
            .filter(Document.user_id == user_id)\
            .first()
    
    def get_active_chat_sessions(self, user_id: str) -> List[ChatSession]:
        """Get active chat sessions for user"""
        return self.session.query(ChatSession)\
            .filter(ChatSession.user_id == user_id)\
            .filter(ChatSession.is_active == True)\
            .order_by(ChatSession.last_activity.desc())\
            .all()
    
    def get_chat_history(self, session_id: str, limit: int = 50, offset: int = 0) -> List[ChatMessage]:
        """Get chat history for a session"""
        return self.session.query(ChatMessage)\
            .filter(ChatMessage.session_id == session_id)\
            .order_by(ChatMessage.timestamp.desc())\
            .offset(offset)\
            .limit(limit)\
            .all()
    
    def get_user_reports(self, user_id: str, status: Optional[str] = None, 
                        limit: int = 10, offset: int = 0) -> List[Report]:
        """Get reports for a user"""
        query = self.session.query(Report).filter(Report.user_id == user_id)
        
        if status:
            query = query.filter(Report.status == status)
        
        return query.order_by(Report.created_at.desc()).offset(offset).limit(limit).all()
    
    def get_user_voice_summaries(self, user_id: str, limit: int = 10, 
                                offset: int = 0) -> List[VoiceSummary]:
        """Get voice summaries for a user"""
        return self.session.query(VoiceSummary)\
            .filter(VoiceSummary.user_id == user_id)\
            .order_by(VoiceSummary.created_at.desc())\
            .offset(offset)\
            .limit(limit)\
            .all()
    
    def get_processing_logs(self, document_id: str, limit: int = 50) -> List[DocumentProcessingLog]:
        """Get processing logs for a document"""
        return self.session.query(DocumentProcessingLog)\
            .filter(DocumentProcessingLog.document_id == document_id)\
            .order_by(DocumentProcessingLog.timestamp.desc())\
            .limit(limit)\
            .all()
    
    def get_system_metrics(self, metric_name: Optional[str] = None, 
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          limit: int = 100) -> List[SystemMetric]:
        """Get system metrics with optional filtering"""
        query = self.session.query(SystemMetric)
        
        if metric_name:
            query = query.filter(SystemMetric.metric_name == metric_name)
        
        if start_time:
            query = query.filter(SystemMetric.timestamp >= start_time)
        
        if end_time:
            query = query.filter(SystemMetric.timestamp <= end_time)
        
        return query.order_by(SystemMetric.timestamp.desc()).limit(limit).all()
    
    def get_audit_logs(self, user_id: Optional[str] = None,
                      action: Optional[str] = None,
                      resource_type: Optional[str] = None,
                      start_time: Optional[datetime] = None,
                      limit: int = 100) -> List[AuditLog]:
        """Get audit logs with filtering"""
        query = self.session.query(AuditLog)
        
        if user_id:
            query = query.filter(AuditLog.user_id == user_id)
        
        if action:
            query = query.filter(AuditLog.action == action)
        
        if resource_type:
            query = query.filter(AuditLog.resource_type == resource_type)
        
        if start_time:
            query = query.filter(AuditLog.timestamp >= start_time)
        
        return query.order_by(AuditLog.timestamp.desc()).limit(limit).all()


class DatabaseStatistics:
    """Utility class for generating database statistics"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a user"""
        stats = {}
        
        # Document statistics
        doc_stats = self.session.query(
            func.count(Document.id).label('total'),
            func.sum(func.case([(Document.status == 'completed', 1)], else_=0)).label('completed'),
            func.sum(func.case([(Document.status == 'processing', 1)], else_=0)).label('processing'),
            func.sum(func.case([(Document.status == 'failed', 1)], else_=0)).label('failed'),
            func.avg(Document.risk_score).label('avg_risk_score'),
            func.sum(Document.page_count).label('total_pages'),
            func.sum(Document.word_count).label('total_words')
        ).filter(Document.user_id == user_id).first()
        
        stats['documents'] = {
            'total': doc_stats.total or 0,
            'completed': doc_stats.completed or 0,
            'processing': doc_stats.processing or 0,
            'failed': doc_stats.failed or 0,
            'avg_risk_score': float(doc_stats.avg_risk_score or 0),
            'total_pages': doc_stats.total_pages or 0,
            'total_words': doc_stats.total_words or 0
        }
        
        # Chat statistics
        chat_stats = self.session.query(
            func.count(ChatSession.id).label('total_sessions'),
            func.sum(ChatSession.message_count).label('total_messages'),
            func.sum(func.case([(ChatSession.is_active == True, 1)], else_=0)).label('active_sessions')
        ).filter(ChatSession.user_id == user_id).first()
        
        stats['chat'] = {
            'total_sessions': chat_stats.total_sessions or 0,
            'total_messages': chat_stats.total_messages or 0,
            'active_sessions': chat_stats.active_sessions or 0
        }
        
        # Report statistics
        report_stats = self.session.query(
            func.count(Report.id).label('total'),
            func.sum(func.case([(Report.status == 'completed', 1)], else_=0)).label('completed'),
            func.sum(Report.file_size).label('total_size')
        ).filter(Report.user_id == user_id).first()
        
        stats['reports'] = {
            'total': report_stats.total or 0,
            'completed': report_stats.completed or 0,
            'total_size': report_stats.total_size or 0
        }
        
        # Voice summary statistics
        voice_stats = self.session.query(
            func.count(VoiceSummary.id).label('total'),
            func.sum(func.case([(VoiceSummary.status == 'completed', 1)], else_=0)).label('completed'),
            func.sum(VoiceSummary.duration_seconds).label('total_duration')
        ).filter(VoiceSummary.user_id == user_id).first()
        
        stats['voice_summaries'] = {
            'total': voice_stats.total or 0,
            'completed': voice_stats.completed or 0,
            'total_duration': voice_stats.total_duration or 0
        }
        
        return stats
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        stats = {}
        
        # User statistics
        user_stats = self.session.query(
            func.count(User.id).label('total'),
            func.sum(func.case([(User.is_active == True, 1)], else_=0)).label('active'),
            func.sum(func.case([(User.role == 'premium', 1)], else_=0)).label('premium')
        ).first()
        
        stats['users'] = {
            'total': user_stats.total or 0,
            'active': user_stats.active or 0,
            'premium': user_stats.premium or 0
        }
        
        # Document statistics
        doc_stats = self.session.query(
            func.count(Document.id).label('total'),
            func.sum(func.case([(Document.status == 'completed', 1)], else_=0)).label('completed'),
            func.avg(Document.risk_score).label('avg_risk_score'),
            func.sum(Document.file_size).label('total_size')
        ).first()
        
        stats['documents'] = {
            'total': doc_stats.total or 0,
            'completed': doc_stats.completed or 0,
            'avg_risk_score': float(doc_stats.avg_risk_score or 0),
            'total_size': doc_stats.total_size or 0
        }
        
        return stats
    
    def get_usage_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get usage trends over specified period"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Daily document uploads
        daily_uploads = self.session.query(
            func.date(Document.upload_date).label('date'),
            func.count(Document.id).label('count')
        ).filter(Document.upload_date >= cutoff_date)\
         .group_by(func.date(Document.upload_date))\
         .order_by(func.date(Document.upload_date))\
         .all()
        
        # Daily chat messages
        daily_messages = self.session.query(
            func.date(ChatMessage.timestamp).label('date'),
            func.count(ChatMessage.id).label('count')
        ).filter(ChatMessage.timestamp >= cutoff_date)\
         .group_by(func.date(ChatMessage.timestamp))\
         .order_by(func.date(ChatMessage.timestamp))\
         .all()
        
        return {
            'daily_uploads': [{'date': str(d.date), 'count': d.count} for d in daily_uploads],
            'daily_messages': [{'date': str(d.date), 'count': d.count} for d in daily_messages],
            'period_days': days
        }


class ModelValidator:
    """Utility class for validating model data"""
    
    @staticmethod
    def validate_document_data(data: Dict[str, Any]) -> DatabaseOperationResult:
        """Validate document data before database insertion"""
        errors = []
        
        # Required fields
        required_fields = ['filename', 'user_id', 'status']
        for field in required_fields:
            if not data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Validate status
        valid_statuses = ['uploaded', 'processing', 'extracting_text', 'analyzing', 'completed', 'failed']
        if data.get('status') not in valid_statuses:
            errors.append(f"Invalid status. Valid options: {valid_statuses}")
        
        # Validate risk score
        if data.get('risk_score') is not None:
            if not isinstance(data['risk_score'], (int, float)) or data['risk_score'] < 0 or data['risk_score'] > 10:
                errors.append("Risk score must be a number between 0 and 10")
        
        # Validate file size
        if data.get('file_size') is not None:
            if not isinstance(data['file_size'], int) or data['file_size'] < 0:
                errors.append("File size must be a positive integer")
        
        if errors:
            return DatabaseOperationResult(
                success=False,
                operation="validate_document",
                error="; ".join(errors)
            )
        
        return DatabaseOperationResult(
            success=True,
            operation="validate_document",
            data=data
        )
    
    @staticmethod
    def validate_user_data(data: Dict[str, Any]) -> DatabaseOperationResult:
        """Validate user data before database insertion"""
        errors = []
        
        # Required fields
        required_fields = ['email', 'username', 'hashed_password']
        for field in required_fields:
            if not data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Validate email format
        if data.get('email'):
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, data['email']):
                errors.append("Invalid email format")
        
        # Validate role
        valid_roles = ['guest', 'user', 'pro', 'premium', 'admin', 'super_admin']
        if data.get('role') and data['role'] not in valid_roles:
            errors.append(f"Invalid role. Valid options: {valid_roles}")
        
        if errors:
            return DatabaseOperationResult(
                success=False,
                operation="validate_user",
                error="; ".join(errors)
            )
        
        return DatabaseOperationResult(
            success=True,
            operation="validate_user",
            data=data
        )


# Export commonly used classes
__all__ = [
    "DatabaseOperationResult",
    "ModelSerializer", 
    "DatabaseQueryHelper",
    "DatabaseStatistics",
    "ModelValidator"
]
