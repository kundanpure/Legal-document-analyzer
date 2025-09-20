"""
Enhanced Database configuration and models for LegalMind AI
Production-ready with comprehensive models, migrations, and connection management
Supports SQLite for demo/development and PostgreSQL for production
"""

import os
import asyncio
import uuid
import json
from typing import AsyncGenerator, Optional, Dict, Any, List
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta

from sqlalchemy import (
    create_engine, Column, String, DateTime, Integer, Float, Text, Boolean, 
    JSON, ForeignKey, Index, UniqueConstraint, CheckConstraint, event
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine, AsyncEngine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext

from config.settings import get_settings
from config.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Database base
Base = declarative_base()

# Custom UUID type that works with both SQLite and PostgreSQL
def generate_uuid():
    return str(uuid.uuid4())


# Enhanced Database Models

class User(Base):
    """User model for authentication and authorization"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, nullable=False, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    
    # Profile information
    organization = Column(String)
    department = Column(String)
    role = Column(String, default="user")  # user, admin, premium
    subscription_tier = Column(String, default="free")  # free, pro, premium
    
    # Account status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    email_verified_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    last_login = Column(DateTime)
    
    # Usage tracking
    documents_uploaded = Column(Integer, default=0)
    reports_generated = Column(Integer, default=0)
    voice_summaries_created = Column(Integer, default=0)
    
    # Preferences
    preferences = Column(JSON, default={})
    
    # Relationships
    documents = relationship("Document", back_populates="user")
    chat_sessions = relationship("ChatSession", back_populates="user")
    reports = relationship("Report", back_populates="user")
    voice_summaries = relationship("VoiceSummary", back_populates="user")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('role IN ("user", "admin", "premium")', name='check_valid_role'),
        CheckConstraint('subscription_tier IN ("free", "pro", "premium")', name='check_valid_subscription'),
    )


class Document(Base):
    """Enhanced document model with comprehensive metadata"""
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    
    # File information
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_size = Column(Integer)
    file_size_formatted = Column(String)
    content_type = Column(String)
    file_hash = Column(String, unique=True)  # For deduplication
    
    # Processing status
    status = Column(String, default="uploaded", index=True)
    progress = Column(Integer, default=0)
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    
    # Timestamps
    upload_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    processing_started_at = Column(DateTime)
    completed_date = Column(DateTime)
    last_accessed = Column(DateTime)
    
    # Document analysis results
    document_type = Column(String, index=True)
    document_subtype = Column(String)
    title = Column(String)
    summary = Column(Text)
    executive_summary = Column(Text)
    
    # Risk assessment
    risk_score = Column(Float)
    risk_level = Column(String, index=True)
    risk_categories = Column(JSON, default={})
    confidence_score = Column(Float)
    
    # Document statistics
    page_count = Column(Integer)
    word_count = Column(Integer)
    paragraph_count = Column(Integer)
    estimated_reading_time = Column(String)
    
    # Analysis results (JSON fields for flexibility)
    key_risks = Column(JSON, default=[])
    recommendations = Column(JSON, default=[])
    flagged_clauses = Column(JSON, default=[])
    user_obligations = Column(JSON, default=[])
    user_rights = Column(JSON, default=[])
    financial_implications = Column(JSON, default={})
    key_topics = Column(JSON, default=[])
    entities_extracted = Column(JSON, default={})
    legal_precedents = Column(JSON, default=[])
    compliance_requirements = Column(JSON, default=[])
    
    # Processing metadata
    fairness_score = Column(Float)
    extraction_confidence = Column(Float)
    classification_confidence = Column(Float)
    model_used = Column(String)
    analysis_version = Column(String, default="2.0")
    
    # Storage and access
    gcs_uri = Column(String)
    extracted_text = Column(Text)
    
    # User and organization
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    
    # Query and language
    query = Column(Text)
    language = Column(String, default="en")
    
    # Processing priority and tags
    priority = Column(Integer, default=5)
    tags = Column(JSON, default=[])
    
    # Client and request metadata
    client_info = Column(JSON, default={})
    request_id = Column(String, index=True)
    
    # Relationships
    user = relationship("User", back_populates="documents")
    chat_sessions = relationship("ChatSession", back_populates="document")
    reports = relationship("Report", back_populates="document")
    voice_summaries = relationship("VoiceSummary", back_populates="document")
    processing_logs = relationship("DocumentProcessingLog", back_populates="document")
    
    # Indexes for better performance
    __table_args__ = (
        Index('idx_user_status', 'user_id', 'status'),
        Index('idx_document_type_risk', 'document_type', 'risk_level'),
        Index('idx_upload_date_desc', 'upload_date'),
        CheckConstraint('status IN ("uploaded", "processing", "extracting_text", "analyzing", "completed", "failed")', name='check_valid_status'),
        CheckConstraint('risk_level IN ("minimal", "low", "medium", "high", "critical")', name='check_valid_risk_level'),
    )


class ChatSession(Base):
    """Enhanced chat session model"""
    __tablename__ = "chat_sessions"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    
    # References
    document_id = Column(String, ForeignKey("documents.id"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    
    # Session configuration
    language = Column(String, default="en")
    session_config = Column(JSON, default={})
    
    # Session status
    is_active = Column(Boolean, default=True, index=True)
    session_type = Column(String, default="document_chat")
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_activity = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ended_at = Column(DateTime)
    expires_at = Column(DateTime)
    
    # Session statistics
    message_count = Column(Integer, default=0)
    total_tokens_used = Column(Integer, default=0)
    avg_response_time = Column(Float)
    
    # Session metadata
    capabilities = Column(JSON, default=[])
    welcome_message = Column(Text)
    context_summary = Column(Text)
    
    # User preferences for this session
    preferences = Column(JSON, default={})
    
    # Relationships
    document = relationship("Document", back_populates="chat_sessions")
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", order_by="ChatMessage.timestamp")
    
    # Constraints and indexes
    __table_args__ = (
        Index('idx_user_active_sessions', 'user_id', 'is_active'),
        Index('idx_document_sessions', 'document_id', 'created_at'),
    )


class ChatMessage(Base):
    """Enhanced chat message model"""
    __tablename__ = "chat_messages"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    
    # References
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False, index=True)
    
    # Message content
    message_type = Column(String, nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    language = Column(String, default="en")
    
    # Message metadata
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    confidence = Column(Float)
    processing_time = Column(Float)
    tokens_used = Column(Integer)
    
    # AI response details
    citations = Column(JSON, default=[])
    suggestions = Column(JSON, default=[])
    follow_up_questions = Column(JSON, default=[])
    
    # Context and analysis
    context_used = Column(Text)
    model_used = Column(String)
    temperature = Column(Float)
    
    # User feedback
    user_rating = Column(Integer)  # -1, 0, 1 for thumbs down, neutral, thumbs up
    user_feedback = Column(Text)
    
    # Request tracking
    request_id = Column(String)
    
    # Relationships
    session = relationship("ChatSession", back_populates="messages")
    
    # Constraints and indexes
    __table_args__ = (
        Index('idx_session_timestamp', 'session_id', 'timestamp'),
        CheckConstraint('message_type IN ("user", "assistant", "system")', name='check_valid_message_type'),
        CheckConstraint('user_rating IS NULL OR user_rating IN (-1, 0, 1)', name='check_valid_rating'),
    )


class Report(Base):
    """Enhanced report model"""
    __tablename__ = "reports"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    
    # References
    document_id = Column(String, ForeignKey("documents.id"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    
    # Report configuration
    template = Column(String, nullable=False)
    language = Column(String, default="en")
    export_format = Column(String, default="pdf")
    
    # Report content configuration
    include_sections = Column(JSON, default=[])
    custom_branding = Column(JSON, default={})
    
    # Processing status
    status = Column(String, default="initializing", index=True)
    progress = Column(Integer, default=0)
    error_message = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime)
    expires_at = Column(DateTime)
    last_downloaded = Column(DateTime)
    
    # File information
    download_url = Column(String)
    file_size = Column(Integer)
    file_size_formatted = Column(String)
    page_count = Column(Integer)
    
    # Generation metadata
    generation_time = Column(Float)
    template_version = Column(String)
    priority = Column(Integer, default=5)
    
    # Request tracking
    request_id = Column(String)
    estimated_time = Column(Integer)
    
    # Batch processing
    batch_id = Column(String, index=True)
    is_batch_item = Column(Boolean, default=False)
    
    # Relationships
    document = relationship("Document", back_populates="reports")
    user = relationship("User", back_populates="reports")
    
    # Constraints and indexes
    __table_args__ = (
        Index('idx_user_reports', 'user_id', 'created_at'),
        Index('idx_document_reports', 'document_id', 'template'),
        Index('idx_batch_reports', 'batch_id', 'created_at'),
        CheckConstraint('status IN ("initializing", "analyzing", "templating", "rendering", "completed", "failed")', name='check_valid_report_status'),
        CheckConstraint('export_format IN ("pdf", "docx", "html")', name='check_valid_export_format'),
    )


class VoiceSummary(Base):
    """Enhanced voice summary model"""
    __tablename__ = "voice_summaries"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    
    # References
    document_id = Column(String, ForeignKey("documents.id"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    
    # Voice configuration
    language = Column(String, nullable=False)
    voice_type = Column(String, default="female")
    speed = Column(Float, default=1.0)
    content_type = Column(String, default="summary")
    
    # Processing status
    status = Column(String, default="initializing", index=True)
    progress = Column(Integer, default=0)
    error_message = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime)
    expires_at = Column(DateTime)
    last_played = Column(DateTime)
    
    # Audio file information
    audio_url = Column(String)
    download_url = Column(String)
    duration = Column(String)  # In MM:SS format
    duration_seconds = Column(Integer)
    file_size = Column(Integer)
    file_size_formatted = Column(String)
    audio_format = Column(String, default="mp3")
    
    # Content information
    transcript = Column(Text)
    content_length = Column(Integer)
    
    # Generation metadata
    generation_time = Column(Float)
    estimated_time = Column(Integer)
    priority = Column(Integer, default=5)
    
    # Request tracking
    request_id = Column(String)
    batch_id = Column(String, index=True)
    
    # Quality and metadata
    audio_quality = Column(String, default="standard")
    model_version = Column(String)
    
    # Relationships
    document = relationship("Document", back_populates="voice_summaries")
    user = relationship("User", back_populates="voice_summaries")
    
    # Constraints and indexes
    __table_args__ = (
        Index('idx_user_voice', 'user_id', 'created_at'),
        Index('idx_document_voice', 'document_id', 'language'),
        Index('idx_language_status', 'language', 'status'),
        CheckConstraint('status IN ("initializing", "preparing", "translating", "synthesizing", "completed", "failed")', name='check_valid_voice_status'),
        CheckConstraint('voice_type IN ("male", "female", "neutral")', name='check_valid_voice_type'),
        CheckConstraint('content_type IN ("summary", "risks", "recommendations", "full", "executive")', name='check_valid_content_type'),
    )


class DocumentProcessingLog(Base):
    """Processing log for detailed tracking"""
    __tablename__ = "document_processing_logs"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False, index=True)
    
    # Log details
    stage = Column(String, nullable=False)
    status = Column(String, nullable=False)  # started, completed, failed
    message = Column(Text)
    error_details = Column(Text)
    
    # Timing
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    duration_seconds = Column(Float)
    
    # Metadata
    metadata = Column(JSON, default={})
    
    # Relationships
    document = relationship("Document", back_populates="processing_logs")
    
    # Indexes
    __table_args__ = (
        Index('idx_document_stage_timestamp', 'document_id', 'stage', 'timestamp'),
    )


class SystemMetric(Base):
    """System metrics for monitoring"""
    __tablename__ = "system_metrics"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    
    # Metric information
    metric_name = Column(String, nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String)
    
    # Context
    service_name = Column(String)
    environment = Column(String, default=settings.ENVIRONMENT)
    
    # Timestamp
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Additional data
    labels = Column(JSON, default={})
    metadata = Column(JSON, default={})
    
    # Indexes
    __table_args__ = (
        Index('idx_metric_name_timestamp', 'metric_name', 'timestamp'),
        Index('idx_service_timestamp', 'service_name', 'timestamp'),
    )


class AuditLog(Base):
    """Audit log for security and compliance"""
    __tablename__ = "audit_logs"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    
    # Action details
    action = Column(String, nullable=False, index=True)
    resource_type = Column(String, nullable=False)
    resource_id = Column(String)
    
    # User and session
    user_id = Column(String, ForeignKey("users.id"), index=True)
    session_id = Column(String)
    ip_address = Column(String)
    user_agent = Column(Text)
    
    # Request details
    request_id = Column(String, index=True)
    endpoint = Column(String)
    method = Column(String)
    
    # Timestamp
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Result and details
    status = Column(String)  # success, failure, error
    details = Column(JSON, default={})
    error_message = Column(Text)
    
    # Relationships
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('idx_user_action_timestamp', 'user_id', 'action', 'timestamp'),
        Index('idx_resource_timestamp', 'resource_type', 'resource_id', 'timestamp'),
    )


# Enhanced Database Manager

class DatabaseManager:
    """Enhanced database manager with connection pooling and health monitoring"""
    
    def __init__(self):
        self.settings = settings
        self.logger = logger
        self.engine: Optional[AsyncEngine] = None
        self.sync_engine = None
        self.AsyncSessionLocal = None
        self.SyncSessionLocal = None
        self._health_status = {"healthy": False, "last_check": None}
        
    async def initialize(self):
        """Initialize database connections with enhanced configuration"""
        try:
            await self._setup_engines()
            await self._create_tables()
            await self._run_migrations()
            await self._setup_event_listeners()
            
            # Verify connection health
            await self.health_check()
            
            self.logger.info("Database initialized successfully", extra={
                "database_type": "postgresql" if "postgresql" in str(self.settings.database.url) else "sqlite",
                "pool_size": self.settings.database.pool_size
            })
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            raise
    
    async def _setup_engines(self):
        """Setup database engines with appropriate configuration"""
        database_url = self.settings.database.url
        
        if not database_url:
            # Default to SQLite for demo
            database_url = "sqlite+aiosqlite:///./legalmind.db"
            self.logger.warning("No database URL configured, using default SQLite")
        
        # Async engine for main operations
        if "sqlite" in database_url:
            self.engine = create_async_engine(
                database_url,
                echo=self.settings.DEBUG,
                connect_args={"check_same_thread": False}
            )
            # Sync engine for migrations
            sync_url = database_url.replace("sqlite+aiosqlite://", "sqlite:///")
            self.sync_engine = create_engine(sync_url, echo=self.settings.DEBUG)
        else:
            # PostgreSQL configuration
            self.engine = create_async_engine(
                database_url,
                echo=self.settings.DEBUG,
                pool_size=self.settings.database.pool_size,
                max_overflow=self.settings.database.max_overflow,
                pool_timeout=self.settings.database.pool_timeout,
                pool_recycle=self.settings.database.pool_recycle,
                pool_pre_ping=True  # Verify connections before use
            )
            # Sync engine for migrations
            sync_url = database_url.replace("postgresql+asyncpg://", "postgresql://")
            self.sync_engine = create_engine(sync_url, echo=self.settings.DEBUG)
        
        # Session factories
        self.AsyncSessionLocal = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        self.SyncSessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.sync_engine
        )
    
    async def _create_tables(self):
        """Create database tables"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            self.logger.info("Database tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Table creation failed: {str(e)}")
            raise
    
    async def _run_migrations(self):
        """Run database migrations using Alembic"""
        try:
            # In a real application, you would configure Alembic properly
            self.logger.info("Database migrations completed (skipped in demo)")
        except Exception as e:
            self.logger.warning(f"Migration check failed: {str(e)}")
    
    async def _setup_event_listeners(self):
        """Setup SQLAlchemy event listeners for automatic updates"""
        
        @event.listens_for(User, 'before_update')
        def update_user_timestamp(mapper, connection, target):
            target.updated_at = datetime.now(timezone.utc)
        
        @event.listens_for(ChatSession, 'before_update')
        def update_session_activity(mapper, connection, target):
            if hasattr(target, 'last_activity'):
                target.last_activity = datetime.now(timezone.utc)
        
        self.logger.info("Database event listeners configured")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session with proper error handling"""
        async with self.AsyncSessionLocal() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    def get_sync_session(self) -> Session:
        """Get synchronous database session"""
        return self.SyncSessionLocal()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        try:
            async with self.get_session() as session:
                # Simple query to check connectivity
                result = await session.execute(func.current_timestamp())
                timestamp = result.scalar()
                
                self._health_status = {
                    "healthy": True,
                    "last_check": datetime.now(timezone.utc),
                    "database_time": timestamp,
                    "connection_pool_size": self.engine.pool.size() if hasattr(self.engine.pool, 'size') else None
                }
                
                return self._health_status
                
        except Exception as e:
            self._health_status = {
                "healthy": False,
                "last_check": datetime.now(timezone.utc),
                "error": str(e)
            }
            
            self.logger.error(f"Database health check failed: {str(e)}")
            return self._health_status
    
    async def get_connection_info(self) -> Dict[str, Any]:
        """Get database connection information"""
        try:
            pool = self.engine.pool
            return {
                "pool_size": pool.size() if hasattr(pool, 'size') else None,
                "checked_in": pool.checkedin() if hasattr(pool, 'checkedin') else None,
                "checked_out": pool.checkedout() if hasattr(pool, 'checkedout') else None,
                "invalid": pool.invalid() if hasattr(pool, 'invalid') else None,
                "database_type": str(self.engine.url.drivername),
                "health_status": self._health_status
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def cleanup(self):
        """Cleanup database connections"""
        try:
            if self.engine:
                await self.engine.dispose()
            if self.sync_engine:
                self.sync_engine.dispose()
                
            self.logger.info("Database connections cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during database cleanup: {str(e)}")


# Global database manager instance
db_manager = DatabaseManager()


# FastAPI Dependencies

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for getting database session"""
    async with db_manager.get_session() as session:
        yield session


async def get_db_health() -> Dict[str, Any]:
    """FastAPI dependency for database health status"""
    return await db_manager.health_check()


# Database Operations

class DatabaseOperations:
    """High-level database operations"""
    
    @staticmethod
    async def save_document(session: AsyncSession, document_data: Dict[str, Any]) -> Document:
        """Save document to database with comprehensive data"""
        try:
            document = Document(
                id=document_data.get('document_id', generate_uuid()),
                filename=document_data['filename'],
                original_filename=document_data.get('original_filename', document_data['filename']),
                file_size=document_data.get('file_size'),
                file_size_formatted=document_data.get('file_size_formatted'),
                content_type=document_data.get('content_type'),
                file_hash=document_data.get('file_hash'),
                status=document_data.get('status', 'uploaded'),
                document_type=document_data.get('document_type'),
                document_subtype=document_data.get('document_subtype'),
                title=document_data.get('title'),
                summary=document_data.get('summary'),
                executive_summary=document_data.get('executive_summary'),
                risk_score=document_data.get('overall_risk_score'),
                risk_level=document_data.get('risk_level'),
                risk_categories=document_data.get('risk_categories', {}),
                confidence_score=document_data.get('confidence_score'),
                page_count=document_data.get('page_count'),
                word_count=document_data.get('word_count'),
                paragraph_count=document_data.get('paragraph_count'),
                estimated_reading_time=document_data.get('estimated_reading_time'),
                key_risks=document_data.get('key_risks', []),
                recommendations=document_data.get('recommendations', []),
                flagged_clauses=document_data.get('flagged_clauses', []),
                user_obligations=document_data.get('user_obligations', []),
                user_rights=document_data.get('user_rights', []),
                financial_implications=document_data.get('financial_implications', {}),
                key_topics=document_data.get('key_topics', []),
                entities_extracted=document_data.get('entities_extracted', {}),
                legal_precedents=document_data.get('legal_precedents', []),
                compliance_requirements=document_data.get('compliance_requirements', []),
                fairness_score=document_data.get('fairness_score'),
                extraction_confidence=document_data.get('extraction_confidence'),
                classification_confidence=document_data.get('classification_confidence'),
                model_used=document_data.get('model_used'),
                analysis_version=document_data.get('analysis_version', '2.0'),
                gcs_uri=document_data.get('gcs_uri'),
                extracted_text=document_data.get('extracted_text'),
                user_id=document_data.get('user_id', 'demo_user'),
                query=document_data.get('query'),
                language=document_data.get('language', 'en'),
                priority=document_data.get('priority', 5),
                tags=document_data.get('tags', []),
                client_info=document_data.get('client_info', {}),
                request_id=document_data.get('request_id')
            )
            
            session.add(document)
            await session.commit()
            await session.refresh(document)
            
            logger.info(f"Document saved to database: {document.id}")
            return document
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Error saving document to database: {str(e)}")
            raise
    
    @staticmethod
    async def save_chat_session(session: AsyncSession, session_data: Dict[str, Any]) -> ChatSession:
        """Save chat session to database"""
        try:
            chat_session = ChatSession(
                id=session_data.get('session_id', generate_uuid()),
                document_id=session_data['document_id'],
                user_id=session_data.get('user_id', 'demo_user'),
                language=session_data.get('language', 'en'),
                session_config=session_data.get('session_config', {}),
                capabilities=session_data.get('capabilities', []),
                welcome_message=session_data.get('welcome_message'),
                preferences=session_data.get('preferences', {})
            )
            
            session.add(chat_session)
            await session.commit()
            await session.refresh(chat_session)
            
            logger.info(f"Chat session saved to database: {chat_session.id}")
            return chat_session
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Error saving chat session to database: {str(e)}")
            raise
    
    @staticmethod
    async def save_chat_message(session: AsyncSession, message_data: Dict[str, Any]) -> ChatMessage:
        """Save chat message to database"""
        try:
            message = ChatMessage(
                id=message_data.get('message_id', generate_uuid()),
                session_id=message_data['session_id'],
                message_type=message_data['type'],
                content=message_data['content'],
                language=message_data.get('language', 'en'),
                confidence=message_data.get('confidence'),
                processing_time=message_data.get('processing_time'),
                tokens_used=message_data.get('tokens_used'),
                citations=message_data.get('citations', []),
                suggestions=message_data.get('suggestions', []),
                follow_up_questions=message_data.get('follow_up_questions', []),
                context_used=message_data.get('context_used'),
                model_used=message_data.get('model_used'),
                temperature=message_data.get('temperature'),
                request_id=message_data.get('request_id')
            )
            
            session.add(message)
            await session.commit()
            await session.refresh(message)
            
            return message
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Error saving chat message to database: {str(e)}")
            raise
    
    @staticmethod
    async def log_document_processing(session: AsyncSession, document_id: str, stage: str, status: str, message: str = None, error_details: str = None, duration: float = None, metadata: Dict[str, Any] = None) -> DocumentProcessingLog:
        """Log document processing stage"""
        try:
            log_entry = DocumentProcessingLog(
                document_id=document_id,
                stage=stage,
                status=status,
                message=message,
                error_details=error_details,
                duration_seconds=duration,
                metadata=metadata or {}
            )
            
            session.add(log_entry)
            await session.commit()
            
            return log_entry
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Error logging document processing: {str(e)}")
            raise


# Database Maintenance

class DatabaseMaintenance:
    """Database maintenance operations"""
    
    @staticmethod
    async def cleanup_expired_sessions():
        """Clean up expired chat sessions"""
        try:
            async with db_manager.get_session() as session:
                expired_time = datetime.now(timezone.utc) - timedelta(hours=2)
                
                # Update expired sessions
                from sqlalchemy import update
                result = await session.execute(
                    update(ChatSession)
                    .where(ChatSession.last_activity < expired_time)
                    .where(ChatSession.is_active == True)
                    .values(is_active=False, ended_at=func.now())
                )
                
                await session.commit()
                
                expired_count = result.rowcount
                logger.info(f"Cleaned up {expired_count} expired chat sessions")
                return expired_count
                
        except Exception as e:
            logger.error(f"Error cleaning up expired sessions: {str(e)}")
            raise
    
    @staticmethod
    async def cleanup_old_data(days: int = 30):
        """Clean up old data from database"""
        try:
            async with db_manager.get_session() as session:
                old_date = datetime.now(timezone.utc) - timedelta(days=days)
                
                # Count records to be deleted
                from sqlalchemy import select, delete
                
                # Old completed documents
                doc_count = await session.execute(
                    select(func.count()).select_from(Document)
                    .where(Document.completed_date < old_date)
                    .where(Document.status == "completed")
                )
                docs_to_delete = doc_count.scalar()
                
                # Delete old documents and related data
                await session.execute(
                    delete(Document)
                    .where(Document.completed_date < old_date)
                    .where(Document.status == "completed")
                )
                
                await session.commit()
                
                logger.info(f"Cleaned up {docs_to_delete} old documents and related data")
                return {"documents_deleted": docs_to_delete}
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
            raise
    
    @staticmethod
    async def optimize_database():
        """Optimize database performance"""
        try:
            async with db_manager.get_session() as session:
                # Database-specific optimization
                if "sqlite" in str(db_manager.engine.url):
                    await session.execute("VACUUM")
                    await session.execute("ANALYZE")
                elif "postgresql" in str(db_manager.engine.url):
                    await session.execute("VACUUM ANALYZE")
                
                await session.commit()
                
                logger.info("Database optimization completed")
                
        except Exception as e:
            logger.error(f"Error optimizing database: {str(e)}")
            raise


# Initialize database
async def init_database():
    """Initialize database on application startup"""
    await db_manager.initialize()


async def close_database():
    """Close database connections on application shutdown"""
    await db_manager.cleanup()


# Export commonly used items
__all__ = [
    "Base", "User", "Document", "ChatSession", "ChatMessage", "Report", 
    "VoiceSummary", "DocumentProcessingLog", "SystemMetric", "AuditLog",
    "DatabaseManager", "DatabaseOperations", "DatabaseMaintenance",
    "db_manager", "get_db", "get_db_health", "init_database", "close_database"
]
