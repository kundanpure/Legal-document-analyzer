"""
Enhanced settings configuration for LegalMind AI
Production-ready configuration with validation, environment support, and security
"""

import os
import secrets
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: Optional[str] = None
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600  # 1 hour


@dataclass
class RedisConfig:
    """Redis configuration"""
    url: Optional[str] = None
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    connection_pool_size: int = 10


@dataclass
class GoogleCloudConfig:
    """Google Cloud configuration"""
    project_id: str = "docanalyzer-470219"
    api_key: str = ""
    service_account_path: Optional[str] = None
    location: str = "us-central1"
    
    # Storage
    gcs_bucket_name: str = "docanalyzer-470219-storage"
    
    # Document AI
    document_ai_processor_id: str = ""
    document_ai_location: str = "us"
    
    # Text-to-Speech
    tts_service_account_path: Optional[str] = None
    tts_project_id: Optional[str] = None


@dataclass
class GeminiConfig:
    """Gemini AI configuration"""
    model: str = "gemini-2.0-flash-exp"
    temperature: float = 0.3
    max_tokens: int = 8192
    top_p: float = 0.8
    top_k: int = 40
    timeout: int = 120  # seconds
    max_retries: int = 3


@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    bcrypt_rounds: int = 12
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["*"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])
    trusted_hosts: List[str] = field(default_factory=list)
    rate_limit_enabled: bool = True
    max_upload_size: int = 50 * 1024 * 1024  # 50MB


@dataclass
class FileStorageConfig:
    """File storage configuration"""
    upload_dir: str = "uploads"
    static_dir: str = "static"
    temp_dir: str = "temp"
    report_template_dir: str = "templates"
    report_output_dir: str = "static/reports"
    voice_output_dir: str = "static/voice"
    max_file_age_days: int = 30
    cleanup_interval_hours: int = 24


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    log_dir: str = "logs"
    enable_console: bool = True
    enable_file: bool = True
    enable_structured: bool = False
    enable_access_log: bool = False
    max_log_size_mb: int = 50
    backup_count: int = 10


@dataclass
class PerformanceConfig:
    """Performance and monitoring configuration"""
    enable_profiling: bool = False
    enable_metrics: bool = False
    metrics_port: int = 9090
    health_check_interval: int = 30
    request_timeout: int = 300  # 5 minutes
    worker_timeout: int = 300
    max_concurrent_requests: int = 100


@dataclass
class VoiceConfig:
    """Voice generation configuration"""
    enabled: bool = True
    default_language: str = "en"
    default_voice_type: str = "female"
    default_speed: float = 1.0
    max_text_length: int = 5000
    cache_enabled: bool = True
    cache_ttl_hours: int = 24
    supported_languages: List[str] = field(default_factory=lambda: [
        "en", "hi", "ta", "te", "bn", "gu", "kn", "ml", "mr", "or", "pa", "ur"
    ])


@dataclass
class DocumentProcessingConfig:
    """Document processing configuration"""
    max_pages: int = 500
    max_text_length: int = 1000000  # 1M characters
    min_text_length: int = 100
    ocr_enabled: bool = True
    ocr_languages: List[str] = field(default_factory=lambda: ["eng", "hin", "tam", "tel", "ben"])
    parallel_processing: bool = True
    max_workers: int = 4
    processing_timeout: int = 300  # 5 minutes
    retry_attempts: int = 3
    cache_enabled: bool = True
    cache_ttl_hours: int = 2


class Settings:
    """Enhanced application settings with environment-aware configuration"""
    
    def __init__(self):
        # Environment
        self.ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
        self.DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
        self.TESTING: bool = os.getenv("TESTING", "false").lower() == "true"
        
        # Application info
        self.APP_NAME: str = os.getenv("APP_NAME", "LegalMind AI")
        self.APP_VERSION: str = os.getenv("APP_VERSION", "2.0.0")
        self.API_V1_PREFIX: str = "/api/v1"
        
        # Host and port
        self.HOST: str = os.getenv("HOST", "0.0.0.0")
        self.PORT: int = int(os.getenv("PORT", "8000"))
        
        # Configuration objects
        self.google_cloud = self._load_google_cloud_config()
        self.gemini = self._load_gemini_config()
        self.database = self._load_database_config()
        self.redis = self._load_redis_config()
        self.security = self._load_security_config()
        self.file_storage = self._load_file_storage_config()
        self.logging = self._load_logging_config()
        self.performance = self._load_performance_config()
        self.voice = self._load_voice_config()
        self.document_processing = self._load_document_processing_config()
        
        # Legacy compatibility properties
        self._setup_legacy_properties()
        
        # Validate configuration
        self._validate_configuration()
    
    def _load_google_cloud_config(self) -> GoogleCloudConfig:
        """Load Google Cloud configuration"""
        return GoogleCloudConfig(
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT", "docanalyzer-470219"),
            api_key=os.getenv("GOOGLE_API_KEY", ""),
            service_account_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
            gcs_bucket_name=os.getenv("GCS_BUCKET_NAME", "docanalyzer-470219-storage"),
            document_ai_processor_id=os.getenv("DOCUMENT_AI_PROCESSOR_ID", ""),
            document_ai_location=os.getenv("DOCUMENT_AI_LOCATION", "us"),
            tts_service_account_path=os.getenv("TTS_SERVICE_ACCOUNT_PATH"),
            tts_project_id=os.getenv("TTS_PROJECT_ID")
        )
    
    def _load_gemini_config(self) -> GeminiConfig:
        """Load Gemini AI configuration"""
        return GeminiConfig(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"),
            temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("GEMINI_MAX_TOKENS", "8192")),
            top_p=float(os.getenv("GEMINI_TOP_P", "0.8")),
            top_k=int(os.getenv("GEMINI_TOP_K", "40")),
            timeout=int(os.getenv("GEMINI_TIMEOUT", "120")),
            max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "3"))
        )
    
    def _load_database_config(self) -> DatabaseConfig:
        """Load database configuration"""
        return DatabaseConfig(
            url=os.getenv("DATABASE_URL"),
            pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
            pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
            pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "3600"))
        )
    
    def _load_redis_config(self) -> RedisConfig:
        """Load Redis configuration"""
        return RedisConfig(
            url=os.getenv("REDIS_URL"),
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
            ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
            connection_pool_size=int(os.getenv("REDIS_POOL_SIZE", "10"))
        )
    
    def _load_security_config(self) -> SecurityConfig:
        """Load security configuration"""
        secret_key = os.getenv("SECRET_KEY")
        if not secret_key:
            if self.ENVIRONMENT == "production":
                raise ValueError("SECRET_KEY must be set in production")
            secret_key = secrets.token_urlsafe(32)
        
        cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
        trusted_hosts = os.getenv("TRUSTED_HOSTS", "").split(",") if os.getenv("TRUSTED_HOSTS") else []
        
        return SecurityConfig(
            secret_key=secret_key,
            jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
            jwt_expiration_hours=int(os.getenv("JWT_EXPIRATION_HOURS", "24")),
            bcrypt_rounds=int(os.getenv("BCRYPT_ROUNDS", "12")),
            cors_origins=cors_origins,
            trusted_hosts=trusted_hosts,
            rate_limit_enabled=os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true",
            max_upload_size=int(os.getenv("MAX_UPLOAD_SIZE", str(50 * 1024 * 1024)))
        )
    
    def _load_file_storage_config(self) -> FileStorageConfig:
        """Load file storage configuration"""
        return FileStorageConfig(
            upload_dir=os.getenv("UPLOAD_DIR", "uploads"),
            static_dir=os.getenv("STATIC_DIR", "static"),
            temp_dir=os.getenv("TEMP_DIR", "temp"),
            report_template_dir=os.getenv("REPORT_TEMPLATE_DIR", "templates"),
            report_output_dir=os.getenv("REPORT_OUTPUT_DIR", "static/reports"),
            voice_output_dir=os.getenv("VOICE_OUTPUT_DIR", "static/voice"),
            max_file_age_days=int(os.getenv("MAX_FILE_AGE_DAYS", "30")),
            cleanup_interval_hours=int(os.getenv("CLEANUP_INTERVAL_HOURS", "24"))
        )
    
    def _load_logging_config(self) -> LoggingConfig:
        """Load logging configuration"""
        return LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            log_dir=os.getenv("LOG_DIR", "logs"),
            enable_console=os.getenv("LOG_CONSOLE", "true").lower() == "true",
            enable_file=os.getenv("LOG_FILE", "true").lower() == "true",
            enable_structured=os.getenv("STRUCTURED_LOGGING", "false").lower() == "true",
            enable_access_log=os.getenv("ACCESS_LOG", "false").lower() == "true",
            max_log_size_mb=int(os.getenv("MAX_LOG_SIZE_MB", "50")),
            backup_count=int(os.getenv("LOG_BACKUP_COUNT", "10"))
        )
    
    def _load_performance_config(self) -> PerformanceConfig:
        """Load performance configuration"""
        return PerformanceConfig(
            enable_profiling=os.getenv("ENABLE_PROFILING", "false").lower() == "true",
            enable_metrics=os.getenv("ENABLE_METRICS", "false").lower() == "true",
            metrics_port=int(os.getenv("METRICS_PORT", "9090")),
            health_check_interval=int(os.getenv("HEALTH_CHECK_INTERVAL", "30")),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "300")),
            worker_timeout=int(os.getenv("WORKER_TIMEOUT", "300")),
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "100"))
        )
    
    def _load_voice_config(self) -> VoiceConfig:
        """Load voice generation configuration"""
        supported_languages = os.getenv("VOICE_SUPPORTED_LANGUAGES", "").split(",") or [
            "en", "hi", "ta", "te", "bn", "gu", "kn", "ml", "mr", "or", "pa", "ur"
        ]
        
        return VoiceConfig(
            enabled=os.getenv("VOICE_ENABLED", "true").lower() == "true",
            default_language=os.getenv("VOICE_DEFAULT_LANGUAGE", "en"),
            default_voice_type=os.getenv("VOICE_DEFAULT_TYPE", "female"),
            default_speed=float(os.getenv("VOICE_DEFAULT_SPEED", "1.0")),
            max_text_length=int(os.getenv("VOICE_MAX_TEXT_LENGTH", "5000")),
            cache_enabled=os.getenv("VOICE_CACHE_ENABLED", "true").lower() == "true",
            cache_ttl_hours=int(os.getenv("VOICE_CACHE_TTL_HOURS", "24")),
            supported_languages=supported_languages
        )
    
    def _load_document_processing_config(self) -> DocumentProcessingConfig:
        """Load document processing configuration"""
        ocr_languages = os.getenv("OCR_LANGUAGES", "").split(",") or ["eng", "hin", "tam", "tel", "ben"]
        
        return DocumentProcessingConfig(
            max_pages=int(os.getenv("DOC_MAX_PAGES", "500")),
            max_text_length=int(os.getenv("DOC_MAX_TEXT_LENGTH", "1000000")),
            min_text_length=int(os.getenv("DOC_MIN_TEXT_LENGTH", "100")),
            ocr_enabled=os.getenv("OCR_ENABLED", "true").lower() == "true",
            ocr_languages=ocr_languages,
            parallel_processing=os.getenv("PARALLEL_PROCESSING", "true").lower() == "true",
            max_workers=int(os.getenv("MAX_WORKERS", "4")),
            processing_timeout=int(os.getenv("PROCESSING_TIMEOUT", "300")),
            retry_attempts=int(os.getenv("RETRY_ATTEMPTS", "3")),
            cache_enabled=os.getenv("DOC_CACHE_ENABLED", "true").lower() == "true",
            cache_ttl_hours=int(os.getenv("DOC_CACHE_TTL_HOURS", "2"))
        )
    
    def _setup_legacy_properties(self):
        """Setup legacy properties for backward compatibility"""
        # Google Cloud legacy properties
        self.GOOGLE_CLOUD_PROJECT = self.google_cloud.project_id
        self.GOOGLE_API_KEY = self.google_cloud.api_key
        self.GCS_BUCKET_NAME = self.google_cloud.gcs_bucket_name
        self.DOCUMENT_AI_PROCESSOR_ID = self.google_cloud.document_ai_processor_id
        
        # Gemini legacy properties
        self.GEMINI_MODEL = self.gemini.model
        self.GEMINI_TEMPERATURE = self.gemini.temperature
        self.GEMINI_MAX_TOKENS = self.gemini.max_tokens
        
        # Other legacy properties
        self.SECRET_KEY = self.security.secret_key
        self.DATABASE_URL = self.database.url
        self.REDIS_URL = self.redis.url
        self.UPLOAD_DIR = self.file_storage.upload_dir
        self.STATIC_DIR = self.file_storage.static_dir
        self.TTS_SERVICE_ACCOUNT_PATH = self.google_cloud.tts_service_account_path
        self.TTS_PROJECT_ID = self.google_cloud.tts_project_id or self.google_cloud.project_id
        self.REPORT_TEMPLATE_DIR = self.file_storage.report_template_dir
        self.REPORT_OUTPUT_DIR = self.file_storage.report_output_dir
        
        # Voice models for compatibility
        self.VOICE_MODELS = self._get_voice_models()
    
    def _get_voice_models(self) -> Dict[str, Dict[str, Any]]:
        """Get voice models configuration"""
        return {
            'en': {
                'language_code': 'en-US',
                'name': 'en-US-Wavenet-F',
                'gender': 'FEMALE'
            },
            'hi': {
                'language_code': 'hi-IN', 
                'name': 'hi-IN-Wavenet-A',
                'gender': 'FEMALE'
            },
            'ta': {
                'language_code': 'ta-IN',
                'name': 'ta-IN-Wavenet-A', 
                'gender': 'FEMALE'
            },
            'te': {
                'language_code': 'te-IN',
                'name': 'te-IN-Standard-A',
                'gender': 'FEMALE'
            },
            'bn': {
                'language_code': 'bn-IN',
                'name': 'bn-IN-Wavenet-A',
                'gender': 'FEMALE'
            },
            'gu': {
                'language_code': 'gu-IN',
                'name': 'gu-IN-Standard-A',
                'gender': 'FEMALE'
            },
            'kn': {
                'language_code': 'kn-IN',
                'name': 'kn-IN-Standard-A', 
                'gender': 'FEMALE'
            },
            'ml': {
                'language_code': 'ml-IN',
                'name': 'ml-IN-Standard-A',
                'gender': 'FEMALE'
            },
            'mr': {
                'language_code': 'mr-IN',
                'name': 'mr-IN-Standard-A',
                'gender': 'FEMALE'
            },
            'pa': {
                'language_code': 'pa-IN',
                'name': 'pa-IN-Standard-A',
                'gender': 'FEMALE'
            }
        }
    
    def _validate_configuration(self):
        """Validate critical configuration settings"""
        errors = []
        
        # Production-specific validations
        if self.ENVIRONMENT == "production":
            if not self.google_cloud.api_key:
                errors.append("GOOGLE_API_KEY must be set in production")
            
            if not self.google_cloud.service_account_path:
                errors.append("GOOGLE_APPLICATION_CREDENTIALS must be set in production")
            
            if self.security.secret_key == "default-secret-key":
                errors.append("SECRET_KEY must be changed from default in production")
        
        # Validate numeric ranges
        if not 0.0 <= self.gemini.temperature <= 2.0:
            errors.append("GEMINI_TEMPERATURE must be between 0.0 and 2.0")
        
        if self.gemini.max_tokens < 1:
            errors.append("GEMINI_MAX_TOKENS must be positive")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors)
            raise ValueError(error_msg)
    
    def create_directories(self):
        """Create required directories if they don't exist"""
        directories = [
            self.file_storage.upload_dir,
            self.file_storage.static_dir,
            self.file_storage.temp_dir,
            self.file_storage.report_template_dir,
            self.file_storage.report_output_dir,
            self.file_storage.voice_output_dir,
            self.logging.log_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for debugging"""
        return {
            "app_name": self.APP_NAME,
            "app_version": self.APP_VERSION,
            "environment": self.ENVIRONMENT,
            "debug": self.DEBUG,
            "host": self.HOST,
            "port": self.PORT,
            "google_cloud": {
                "project_id": self.google_cloud.project_id,
                "location": self.google_cloud.location,
                "has_api_key": bool(self.google_cloud.api_key),
                "has_service_account": bool(self.google_cloud.service_account_path)
            },
            "gemini": {
                "model": self.gemini.model,
                "temperature": self.gemini.temperature,
                "max_tokens": self.gemini.max_tokens
            },
            "features": {
                "voice_enabled": self.voice.enabled,
                "ocr_enabled": self.document_processing.ocr_enabled,
                "parallel_processing": self.document_processing.parallel_processing,
                "rate_limiting": self.security.rate_limit_enabled
            }
        }


def get_settings() -> Settings:
    """Get application settings instance"""
    return Settings()


# Create global settings instance
settings = get_settings()

# Create required directories
settings.create_directories()

# Auto-setup logging
from .logging import setup_logging
setup_logging(
    environment=settings.ENVIRONMENT,
    log_level=settings.logging.level,
    log_dir=settings.logging.log_dir,
    enable_console=settings.logging.enable_console,
    enable_file=settings.logging.enable_file,
    enable_structured=settings.logging.enable_structured
)
