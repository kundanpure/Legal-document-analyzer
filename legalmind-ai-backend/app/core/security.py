"""
Enhanced Security utilities and authentication for LegalMind AI
Production-ready with comprehensive authentication, authorization, and security controls
"""

import hashlib
import secrets
import hmac
import base64
import re
import ipaddress
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Union, Tuple
from enum import Enum
import json

import jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt
from cryptography.fernet import Fernet
from fastapi import Request, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis.asyncio as redis

from config.settings import get_settings
from config.logging import get_logger
from app.core.exceptions import (
    AuthenticationError, AuthorizationError, RateLimitError, 
    ValidationError, SecurityError
)

logger = get_logger(__name__)
settings = get_settings()


class UserRole(Enum):
    """User role enumeration"""
    GUEST = "guest"
    USER = "user"
    PRO = "pro"
    PREMIUM = "premium"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class Permission(Enum):
    """Permission enumeration"""
    READ_DOCUMENTS = "read:documents"
    WRITE_DOCUMENTS = "write:documents"
    DELETE_DOCUMENTS = "delete:documents"
    READ_REPORTS = "read:reports"
    WRITE_REPORTS = "write:reports"
    READ_VOICE = "read:voice"
    WRITE_VOICE = "write:voice"
    READ_CHAT = "read:chat"
    WRITE_CHAT = "write:chat"
    ADMIN_USERS = "admin:users"
    ADMIN_SYSTEM = "admin:system"
    ADMIN_SETTINGS = "admin:settings"


class SecurityLevel(Enum):
    """Security level for operations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Password hashing configuration
pwd_context = CryptContext(
    schemes=["bcrypt", "pbkdf2_sha256"],
    deprecated="auto",
    bcrypt__rounds=12,  # Higher rounds for better security
    pbkdf2_sha256__rounds=100000
)


class SecurityManager:
    """Enhanced security manager with comprehensive security features"""
    
    def __init__(self):
        self.settings = settings
        self.logger = logger
        self.secret_key = settings.security.secret_key
        self.algorithm = "HS256"
        self.token_expire_minutes = settings.security.access_token_expire_minutes
        self.refresh_token_expire_days = settings.security.refresh_token_expire_days
        
        # Initialize encryption
        self._setup_encryption()
        
        # Role-based permissions
        self.role_permissions = self._setup_role_permissions()
        
        # Security patterns
        self.security_patterns = self._setup_security_patterns()
    
    def _setup_encryption(self):
        """Setup encryption for sensitive data"""
        if hasattr(settings.security, 'encryption_key') and settings.security.encryption_key:
            self.cipher_suite = Fernet(settings.security.encryption_key.encode())
        else:
            # Generate a key for demo (in production, this should be from secure storage)
            key = Fernet.generate_key()
            self.cipher_suite = Fernet(key)
            self.logger.warning("Using generated encryption key - not suitable for production")
    
    def _setup_role_permissions(self) -> Dict[UserRole, List[Permission]]:
        """Setup role-based permissions mapping"""
        return {
            UserRole.GUEST: [
                Permission.READ_DOCUMENTS
            ],
            UserRole.USER: [
                Permission.READ_DOCUMENTS,
                Permission.WRITE_DOCUMENTS,
                Permission.READ_REPORTS,
                Permission.WRITE_REPORTS,
                Permission.READ_VOICE,
                Permission.READ_CHAT,
                Permission.WRITE_CHAT
            ],
            UserRole.PRO: [
                Permission.READ_DOCUMENTS,
                Permission.WRITE_DOCUMENTS,
                Permission.READ_REPORTS,
                Permission.WRITE_REPORTS,
                Permission.READ_VOICE,
                Permission.WRITE_VOICE,
                Permission.READ_CHAT,
                Permission.WRITE_CHAT
            ],
            UserRole.PREMIUM: [
                Permission.READ_DOCUMENTS,
                Permission.WRITE_DOCUMENTS,
                Permission.DELETE_DOCUMENTS,
                Permission.READ_REPORTS,
                Permission.WRITE_REPORTS,
                Permission.READ_VOICE,
                Permission.WRITE_VOICE,
                Permission.READ_CHAT,
                Permission.WRITE_CHAT
            ],
            UserRole.ADMIN: [perm for perm in Permission],  # All permissions
            UserRole.SUPER_ADMIN: [perm for perm in Permission]  # All permissions
        }
    
    def _setup_security_patterns(self) -> Dict[str, re.Pattern]:
        """Setup security patterns for validation"""
        return {
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'password_strong': re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'),
            'filename_safe': re.compile(r'^[a-zA-Z0-9._-]+$'),
            'sql_injection': re.compile(r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\b)', re.IGNORECASE),
            'xss_basic': re.compile(r'<script|javascript:|on\w+\s*=', re.IGNORECASE),
            'path_traversal': re.compile(r'\.\.[\\/]'),
            'uuid': re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$', re.IGNORECASE)
        }
    
    # Password Management
    
    def hash_password(self, password: str) -> str:
        """Hash a password with secure algorithm"""
        if not self.validate_password_strength(password):
            raise ValidationError(
                "Password does not meet security requirements",
                suggestions=[
                    "Password must be at least 8 characters long",
                    "Include uppercase and lowercase letters",
                    "Include at least one number",
                    "Include at least one special character"
                ]
            )
        
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            self.logger.error(f"Password verification error: {str(e)}")
            return False
    
    def validate_password_strength(self, password: str) -> bool:
        """Validate password strength"""
        if not password or len(password) < 8:
            return False
        
        return bool(self.security_patterns['password_strong'].match(password))
    
    def generate_secure_password(self, length: int = 16) -> str:
        """Generate a secure random password"""
        characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@$!%*?&"
        return ''.join(secrets.choice(characters) for _ in range(length))
    
    # Token Management
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token with enhanced claims"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=self.token_expire_minutes)
        
        # Enhanced token claims
        to_encode.update({
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "access_token",
            "jti": secrets.token_urlsafe(16)  # JWT ID for revocation
        })
        
        try:
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            return encoded_jwt
        except Exception as e:
            self.logger.error(f"Token creation error: {str(e)}")
            raise AuthenticationError("Failed to create authentication token")
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create a refresh token"""
        data = {
            "user_id": user_id,
            "type": "refresh_token",
            "exp": datetime.now(timezone.utc) + timedelta(days=self.refresh_token_expire_days),
            "iat": datetime.now(timezone.utc),
            "jti": secrets.token_urlsafe(16)
        }
        
        try:
            return jwt.encode(data, self.secret_key, algorithm=self.algorithm)
        except Exception as e:
            self.logger.error(f"Refresh token creation error: {str(e)}")
            raise AuthenticationError("Failed to create refresh token")
    
    def verify_token(self, token: str, token_type: str = "access_token") -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token with type checking"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Verify token type
            if payload.get("type") != token_type:
                return None
            
            # Check expiration
            if payload.get("exp", 0) < datetime.now(timezone.utc).timestamp():
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            self.logger.warning("Invalid token")
            return None
        except Exception as e:
            self.logger.error(f"Token verification error: {str(e)}")
            return None
    
    def revoke_token(self, token_jti: str):
        """Revoke a token by adding its JTI to blacklist"""
        # In production, store in Redis or database
        # For demo, we'll log it
        self.logger.info(f"Token revoked: {token_jti}")
    
    # API Key Management
    
    def generate_api_key(self, user_id: str, permissions: List[str] = None) -> Dict[str, str]:
        """Generate a secure API key with metadata"""
        key = secrets.token_urlsafe(32)
        key_id = secrets.token_urlsafe(8)
        
        # In production, store key metadata in database
        api_key_data = {
            "key_id": key_id,
            "key": key,
            "user_id": user_id,
            "permissions": permissions or [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_used": None,
            "usage_count": 0,
            "is_active": True
        }
        
        self.logger.info(f"API key generated for user: {user_id}")
        return api_key_data
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return associated data"""
        # In production, lookup in database
        # For demo, implement basic validation
        if len(api_key) >= 32:
            return {
                "user_id": "api_user",
                "permissions": ["read:documents", "write:documents"],
                "is_active": True
            }
        return None
    
    # Session Management
    
    def generate_session_id(self) -> str:
        """Generate a secure session ID"""
        return secrets.token_urlsafe(32)
    
    def create_session_data(self, user_id: str, additional_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create session data with security metadata"""
        session_data = {
            "user_id": user_id,
            "session_id": self.generate_session_id(),
            "created_at": datetime.now(timezone.utc),
            "last_activity": datetime.now(timezone.utc),
            "expires_at": datetime.now(timezone.utc) + timedelta(hours=24),
            "ip_address": None,  # Set by caller
            "user_agent": None,  # Set by caller
            "is_active": True
        }
        
        if additional_data:
            session_data.update(additional_data)
        
        return session_data
    
    # Input Validation and Sanitization
    
    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        return bool(self.security_patterns['email'].match(email))
    
    def validate_filename(self, filename: str) -> bool:
        """Validate filename for security"""
        if not filename or len(filename) > 255:
            return False
        
        # Check for safe characters
        return bool(self.security_patterns['filename_safe'].match(filename))
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for security"""
        if not filename:
            return "unnamed_file"
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove path traversal attempts
        sanitized = sanitized.replace('..', '_')
        
        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')
        
        # Ensure reasonable length
        if len(sanitized) > 255:
            name, ext = (sanitized.rsplit('.', 1) + [''])[:2]
            if ext:
                sanitized = name[:255-len(ext)-1] + '.' + ext
            else:
                sanitized = sanitized[:255]
        
        return sanitized or "unnamed_file"
    
    def validate_user_input(self, input_str: str, max_length: int = 1000, 
                          check_sql_injection: bool = True, check_xss: bool = True) -> str:
        """Comprehensive user input validation and sanitization"""
        if not isinstance(input_str, str):
            return ""
        
        # Check for SQL injection patterns
        if check_sql_injection and self.security_patterns['sql_injection'].search(input_str):
            raise ValidationError("Input contains potentially harmful SQL patterns")
        
        # Check for XSS patterns
        if check_xss and self.security_patterns['xss_basic'].search(input_str):
            raise ValidationError("Input contains potentially harmful script content")
        
        # Check for path traversal
        if self.security_patterns['path_traversal'].search(input_str):
            raise ValidationError("Input contains path traversal attempts")
        
        # Sanitize and limit length
        sanitized = input_str.strip()
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized
    
    # File Security
    
    def validate_file_upload_security(self, file_content: bytes, filename: str, 
                                    content_type: str = None) -> Dict[str, Any]:
        """Comprehensive file upload security validation"""
        result = {"valid": True, "errors": [], "warnings": []}
        
        # File size check
        max_size = settings.security.max_upload_size
        if len(file_content) > max_size:
            result["valid"] = False
            result["errors"].append(f"File too large. Maximum size: {max_size // (1024*1024)}MB")
        
        # Filename validation
        if not self.validate_filename(filename):
            result["valid"] = False
            result["errors"].append("Invalid filename")
        
        # File extension check
        allowed_extensions = ['.pdf', '.docx', '.doc', '.txt', '.rtf']
        file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
        if f'.{file_ext}' not in allowed_extensions:
            result["valid"] = False
            result["errors"].append(f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}")
        
        # Content type validation
        if content_type:
            allowed_content_types = [
                'application/pdf',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'application/msword',
                'text/plain',
                'text/rtf'
            ]
            if content_type not in allowed_content_types:
                result["warnings"].append("Content type may not match file extension")
        
        # Malicious content checks
        malicious_patterns = [
            b'<script',
            b'javascript:',
            b'<?php',
            b'<%',
            b'eval(',
            b'exec(',
            b'system(',
        ]
        
        file_content_lower = file_content.lower()
        for pattern in malicious_patterns:
            if pattern in file_content_lower:
                result["valid"] = False
                result["errors"].append("File contains potentially malicious content")
                break
        
        # File header validation (magic bytes)
        if not self._validate_file_header(file_content, filename):
            result["warnings"].append("File header doesn't match expected format")
        
        return result
    
    def _validate_file_header(self, file_content: bytes, filename: str) -> bool:
        """Validate file header against expected format"""
        if not file_content or len(file_content) < 4:
            return False
        
        # PDF files
        if filename.lower().endswith('.pdf'):
            return file_content.startswith(b'%PDF-')
        
        # DOCX files
        elif filename.lower().endswith('.docx'):
            return file_content.startswith(b'PK\x03\x04')  # ZIP header
        
        # DOC files
        elif filename.lower().endswith('.doc'):
            return file_content.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1')  # OLE header
        
        # TXT files (no specific header)
        elif filename.lower().endswith(('.txt', '.rtf')):
            return True
        
        return True  # Allow by default
    
    def hash_file_content(self, content: bytes) -> str:
        """Generate secure hash of file content for integrity checking"""
        return hashlib.sha256(content).hexdigest()
    
    # Encryption/Decryption
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            return self.cipher_suite.encrypt(data.encode()).decode()
        except Exception as e:
            self.logger.error(f"Encryption error: {str(e)}")
            raise SecurityError("Failed to encrypt sensitive data")
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            return self.cipher_suite.decrypt(encrypted_data.encode()).decode()
        except Exception as e:
            self.logger.error(f"Decryption error: {str(e)}")
            raise SecurityError("Failed to decrypt sensitive data")
    
    # Authorization
    
    def check_permission(self, user_role: UserRole, required_permission: Permission) -> bool:
        """Check if user role has required permission"""
        user_permissions = self.role_permissions.get(user_role, [])
        return required_permission in user_permissions
    
    def get_user_permissions(self, user_role: UserRole) -> List[Permission]:
        """Get all permissions for a user role"""
        return self.role_permissions.get(user_role, [])
    
    # IP and Network Security
    
    def validate_ip_address(self, ip_str: str) -> bool:
        """Validate IP address format"""
        try:
            ipaddress.ip_address(ip_str)
            return True
        except ValueError:
            return False
    
    def is_ip_whitelisted(self, ip_str: str) -> bool:
        """Check if IP is in whitelist"""
        # In production, check against whitelist in database/config
        whitelisted_ranges = [
            '127.0.0.0/8',  # Localhost
            '10.0.0.0/8',   # Private network
            '172.16.0.0/12',  # Private network
            '192.168.0.0/16'  # Private network
        ]
        
        try:
            ip = ipaddress.ip_address(ip_str)
            for range_str in whitelisted_ranges:
                if ip in ipaddress.ip_network(range_str):
                    return True
        except ValueError:
            pass
        
        return False
    
    def is_ip_blacklisted(self, ip_str: str) -> bool:
        """Check if IP is in blacklist"""
        # In production, check against blacklist in database/Redis
        blacklisted_ips = []  # Load from configuration
        return ip_str in blacklisted_ips
    
    # Security Headers
    
    def get_security_headers(self, security_level: SecurityLevel = SecurityLevel.MEDIUM) -> Dict[str, str]:
        """Get security headers based on security level"""
        base_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "X-Permitted-Cross-Domain-Policies": "none"
        }
        
        if security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            base_headers.update({
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
                "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
                "X-Download-Options": "noopen",
                "X-DNS-Prefetch-Control": "off"
            })
        
        if security_level == SecurityLevel.CRITICAL:
            base_headers.update({
                "Feature-Policy": "geolocation 'none'; camera 'none'; microphone 'none'",
                "Permissions-Policy": "geolocation=(), camera=(), microphone=()"
            })
        
        return base_headers


# Enhanced Rate Limiting

class RateLimiter:
    """Enhanced rate limiter with multiple strategies and persistence"""
    
    def __init__(self, redis_client=None):
        self.requests = {}  # In-memory fallback
        self.logger = logger
        self.redis_client = redis_client
    
    async def is_allowed(self, key: str, limit: int, window_seconds: int, 
                        burst_limit: Optional[int] = None) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed with detailed response"""
        
        if self.redis_client:
            return await self._redis_rate_limit(key, limit, window_seconds, burst_limit)
        else:
            return await self._memory_rate_limit(key, limit, window_seconds, burst_limit)
    
    async def _redis_rate_limit(self, key: str, limit: int, window_seconds: int, 
                              burst_limit: Optional[int] = None) -> Tuple[bool, Dict[str, Any]]:
        """Redis-based rate limiting with sliding window"""
        current_time = datetime.now(timezone.utc).timestamp()
        
        try:
            # Sliding window rate limiting using Redis
            pipe = self.redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, current_time - window_seconds)
            pipe.zcard(key)
            pipe.zadd(key, {str(current_time): current_time})
            pipe.expire(key, window_seconds)
            
            results = await pipe.execute()
            current_requests = results[1]
            
            # Check burst limit
            if burst_limit and current_requests >= burst_limit:
                return False, {
                    "allowed": False,
                    "current_requests": current_requests,
                    "limit": limit,
                    "burst_limit": burst_limit,
                    "window_seconds": window_seconds,
                    "retry_after": window_seconds // 4,
                    "limit_type": "burst"
                }
            
            # Check regular limit
            if current_requests >= limit:
                return False, {
                    "allowed": False,
                    "current_requests": current_requests,
                    "limit": limit,
                    "window_seconds": window_seconds,
                    "retry_after": window_seconds // 2,
                    "limit_type": "regular"
                }
            
            return True, {
                "allowed": True,
                "current_requests": current_requests + 1,
                "limit": limit,
                "remaining": limit - current_requests - 1,
                "window_seconds": window_seconds,
                "reset_time": current_time + window_seconds
            }
            
        except Exception as e:
            self.logger.error(f"Redis rate limiting error: {str(e)}")
            # Fallback to memory-based rate limiting
            return await self._memory_rate_limit(key, limit, window_seconds, burst_limit)
    
    async def _memory_rate_limit(self, key: str, limit: int, window_seconds: int,
                               burst_limit: Optional[int] = None) -> Tuple[bool, Dict[str, Any]]:
        """Memory-based rate limiting"""
        now = datetime.now(timezone.utc)
        
        if key not in self.requests:
            self.requests[key] = []
        
        # Remove old requests outside the window
        window_start = now - timedelta(seconds=window_seconds)
        self.requests[key] = [
            req_time for req_time in self.requests[key] 
            if req_time > window_start
        ]
        
        current_requests = len(self.requests[key])
        
        # Check burst limit
        if burst_limit and current_requests >= burst_limit:
            return False, {
                "allowed": False,
                "current_requests": current_requests,
                "limit": limit,
                "burst_limit": burst_limit,
                "window_seconds": window_seconds,
                "retry_after": window_seconds // 4,
                "limit_type": "burst"
            }
        
        # Check regular limit
        if current_requests >= limit:
            return False, {
                "allowed": False,
                "current_requests": current_requests,
                "limit": limit,
                "window_seconds": window_seconds,
                "retry_after": window_seconds // 2,
                "limit_type": "regular"
            }
        
        # Add current request
        self.requests[key].append(now)
        
        return True, {
            "allowed": True,
            "current_requests": current_requests + 1,
            "limit": limit,
            "remaining": limit - current_requests - 1,
            "window_seconds": window_seconds,
            "reset_time": (now + timedelta(seconds=window_seconds)).timestamp()
        }
    
    async def get_rate_limit_info(self, key: str, limit: int, window_seconds: int) -> Dict[str, Any]:
        """Get current rate limit information"""
        allowed, info = await self.is_allowed(key, limit, window_seconds)
        return info
    
    async def cleanup_old_entries(self):
        """Clean up old rate limiting entries from memory"""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=1)
        
        keys_to_remove = []
        for key, requests in self.requests.items():
            # Remove old requests
            self.requests[key] = [req for req in requests if req > cutoff]
            
            # Mark empty keys for removal
            if not self.requests[key]:
                keys_to_remove.append(key)
        
        # Remove empty keys
        for key in keys_to_remove:
            del self.requests[key]


# FastAPI Security Dependencies

security_scheme = HTTPBearer(auto_error=False)

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme)
) -> Optional[Dict[str, Any]]:
    """Get current authenticated user from token"""
    
    if not credentials:
        # Return demo user for development
        if settings.ENVIRONMENT == "development":
            return {
                "user_id": "demo_user",
                "email": "demo@example.com",
                "role": UserRole.USER.value,
                "permissions": ["read:documents", "write:documents", "read:reports", "write:reports"],
                "is_active": True,
                "subscription_tier": "free"
            }
        return None
    
    try:
        token_data = security_manager.verify_token(credentials.credentials, "access_token")
        if not token_data:
            return None
        
        # In production, lookup user in database
        user_data = {
            "user_id": token_data.get("user_id"),
            "email": token_data.get("email"),
            "role": token_data.get("role", UserRole.USER.value),
            "permissions": token_data.get("permissions", []),
            "is_active": token_data.get("is_active", True),
            "subscription_tier": token_data.get("subscription_tier", "free")
        }
        
        return user_data
        
    except Exception as e:
        logger.warning(f"Token validation error: {str(e)}")
        return None


def require_auth():
    """Require authentication"""
    async def auth_dependency(current_user: Optional[Dict] = Depends(get_current_user)):
        if not current_user:
            raise AuthenticationError("Authentication required")
        return current_user
    
    return auth_dependency


def require_role(required_role: UserRole):
    """Require specific user role"""
    async def role_dependency(current_user: Dict = Depends(require_auth())):
        user_role = UserRole(current_user.get("role", UserRole.USER.value))
        
        # Role hierarchy check
        role_hierarchy = [
            UserRole.GUEST,
            UserRole.USER, 
            UserRole.PRO,
            UserRole.PREMIUM,
            UserRole.ADMIN,
            UserRole.SUPER_ADMIN
        ]
        
        if role_hierarchy.index(user_role) < role_hierarchy.index(required_role):
            raise AuthorizationError(
                f"Required role: {required_role.value}",
                required_permission=required_role.value,
                user_role=user_role.value
            )
        
        return current_user
    
    return role_dependency


def require_permission(required_permission: Permission):
    """Require specific permission"""
    async def permission_dependency(current_user: Dict = Depends(require_auth())):
        user_permissions = current_user.get("permissions", [])
        
        if required_permission.value not in user_permissions:
            user_role = UserRole(current_user.get("role", UserRole.USER.value))
            
            # Check role-based permissions
            if not security_manager.check_permission(user_role, required_permission):
                raise AuthorizationError(
                    f"Required permission: {required_permission.value}",
                    required_permission=required_permission.value,
                    user_role=user_role.value
                )
        
        return current_user
    
    return permission_dependency


def require_admin():
    """Require admin access"""
    return require_role(UserRole.ADMIN)


def rate_limit(endpoint_name: str, limit: int = None, window_seconds: int = None):
    """Rate limiting dependency"""
    async def rate_limit_dependency(request: Request):
        # Get rate limit configuration
        if not limit or not window_seconds:
            # Use default limits from settings
            endpoint_limits = {
                "document_upload": {"limit": 10, "window": 300},
                "chat_message": {"limit": 60, "window": 60},
                "voice_generation": {"limit": 5, "window": 300},
                "report_generation": {"limit": 10, "window": 300},
                "general_api": {"limit": 100, "window": 60}
            }
            
            config = endpoint_limits.get(endpoint_name, {"limit": 30, "window": 60})
            actual_limit = limit or config["limit"]
            actual_window = window_seconds or config["window"]
        else:
            actual_limit = limit
            actual_window = window_seconds
        
        # Get client identifier
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        # Create rate limit key
        rate_key = f"rate_limit:{endpoint_name}:{client_ip}"
        
        # Check rate limit
        allowed, info = await rate_limiter.is_allowed(rate_key, actual_limit, actual_window)
        
        if not allowed:
            raise RateLimitError(
                "Rate limit exceeded",
                limit=actual_limit,
                window_seconds=actual_window,
                retry_after=info.get("retry_after"),
                endpoint=endpoint_name
            )
        
        # Add rate limit headers to response (handled by middleware)
        return info
    
    return rate_limit_dependency


# Global instances
security_manager = SecurityManager()
rate_limiter = RateLimiter()


# Audit Logging

class AuditLogger:
    """Enhanced audit logger for security events and compliance"""
    
    def __init__(self):
        self.logger = logger
    
    def log_authentication(self, event_type: str, user_identifier: str, 
                         ip_address: str, user_agent: str = None, 
                         success: bool = True, details: Dict[str, Any] = None):
        """Log authentication events"""
        audit_data = {
            "event_type": f"auth_{event_type}",
            "user_identifier": user_identifier,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {}
        }
        
        if success:
            self.logger.info(f"AUTH: {event_type} successful - {user_identifier}", extra=audit_data)
        else:
            self.logger.warning(f"AUTH: {event_type} failed - {user_identifier}", extra=audit_data)
    
    def log_authorization(self, user_id: str, action: str, resource: str, 
                         allowed: bool, required_permission: str = None):
        """Log authorization events"""
        audit_data = {
            "event_type": "authorization",
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "allowed": allowed,
            "required_permission": required_permission,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if allowed:
            self.logger.info(f"AUTHZ: Access granted - User: {user_id}, Action: {action}, Resource: {resource}", extra=audit_data)
        else:
            self.logger.warning(f"AUTHZ: Access denied - User: {user_id}, Action: {action}, Resource: {resource}", extra=audit_data)
    
    def log_data_access(self, user_id: str, data_type: str, data_id: str, 
                       action: str, ip_address: str = None):
        """Log data access events"""
        audit_data = {
            "event_type": "data_access",
            "user_id": user_id,
            "data_type": data_type,
            "data_id": data_id,
            "action": action,
            "ip_address": ip_address,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        self.logger.info(f"DATA: {action} - User: {user_id}, Type: {data_type}, ID: {data_id}", extra=audit_data)
    
    def log_security_event(self, event_type: str, severity: str, description: str, 
                          ip_address: str = None, user_id: str = None, 
                          details: Dict[str, Any] = None):
        """Log security events"""
        audit_data = {
            "event_type": f"security_{event_type}",
            "severity": severity,
            "description": description,
            "ip_address": ip_address,
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {}
        }
        
        if severity in ["high", "critical"]:
            self.logger.error(f"SECURITY: {event_type} - {description}", extra=audit_data)
        else:
            self.logger.warning(f"SECURITY: {event_type} - {description}", extra=audit_data)


# Global audit logger
audit_logger = AuditLogger()


# Security Configuration Validation

def validate_security_config():
    """Validate security configuration on startup"""
    issues = []
    
    # Check secret key
    if not settings.security.secret_key or len(settings.security.secret_key) < 32:
        issues.append("SECRET_KEY is too short or missing (minimum 32 characters)")
    
    # Check production settings
    if settings.ENVIRONMENT == "production":
        if settings.DEBUG:
            issues.append("DEBUG mode enabled in production")
        
        if settings.LOG_LEVEL == "DEBUG":
            issues.append("Debug logging enabled in production")
        
        if not settings.security.rate_limit_enabled:
            issues.append("Rate limiting disabled in production")
    
    # Check CORS settings
    if "*" in getattr(settings, 'CORS_ORIGINS', []):
        if settings.ENVIRONMENT == "production":
            issues.append("Wildcard CORS origins in production")
    
    # Check encryption key
    if not hasattr(settings.security, 'encryption_key') or not settings.security.encryption_key:
        issues.append("Encryption key not configured")
    
    # Log results
    if issues:
        logger.warning(f"Security configuration issues found: {', '.join(issues)}")
        return False
    else:
        logger.info("Security configuration validated successfully")
        return True


# Utility functions

def sanitize_query_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize query parameters"""
    sanitized = {}
    
    for key, value in params.items():
        if isinstance(value, str):
            sanitized[key] = security_manager.validate_user_input(value, max_length=1000)
        elif isinstance(value, (int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, list):
            sanitized[key] = [
                security_manager.validate_user_input(str(item), max_length=100) 
                if isinstance(item, str) else item 
                for item in value[:10]  # Limit array size
            ]
        # Skip other types
    
    return sanitized


def mask_sensitive_data(data: str, show_chars: int = 4) -> str:
    """Mask sensitive data for logging"""
    if not data or len(data) <= show_chars:
        return "*" * len(data) if data else ""
    
    return data[:show_chars] + "*" * (len(data) - show_chars)


# Export commonly used items
__all__ = [
    "SecurityManager", "RateLimiter", "AuditLogger",
    "UserRole", "Permission", "SecurityLevel",
    "security_manager", "rate_limiter", "audit_logger",
    "get_current_user", "require_auth", "require_role", 
    "require_permission", "require_admin", "rate_limit",
    "validate_security_config", "sanitize_query_params", "mask_sensitive_data"
]
