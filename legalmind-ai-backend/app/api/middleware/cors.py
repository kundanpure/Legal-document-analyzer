"""
Enhanced CORS middleware configuration for LegalMind AI
Production-ready with security features, environment-specific settings, and comprehensive headers
"""

import re
from typing import List, Optional, Union
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from config.settings import get_settings
from config.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        security_headers = {
            # Prevent clickjacking
            "X-Frame-Options": "DENY",
            
            # Enable XSS protection
            "X-XSS-Protection": "1; mode=block",
            
            # Prevent MIME type sniffing
            "X-Content-Type-Options": "nosniff",
            
            # Referrer policy for privacy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            
            # Permissions policy
            "Permissions-Policy": (
                "geolocation=(), microphone=(), camera=(), "
                "payment=(), usb=(), magnetometer=(), gyroscope=()"
            ),
            
            # Server information hiding
            "Server": "LegalMind-API"
        }
        
        # Add CSP for production
        if settings.ENVIRONMENT == "production":
            csp_directives = [
                "default-src 'self'",
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
                "style-src 'self' 'unsafe-inline'",
                "img-src 'self' data: https:",
                "font-src 'self' data:",
                "connect-src 'self' https:",
                "media-src 'self'",
                "object-src 'none'",
                "base-uri 'self'",
                "form-action 'self'",
                "frame-ancestors 'none'"
            ]
            security_headers["Content-Security-Policy"] = "; ".join(csp_directives)
            
            # HSTS for HTTPS
            security_headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        
        # Apply headers
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response


def get_allowed_origins() -> List[str]:
    """
    Get allowed origins based on environment and configuration
    
    Returns:
        List of allowed origin patterns
    """
    
    if settings.ENVIRONMENT == "development":
        # Development origins - more permissive
        return [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3001", 
            "http://127.0.0.1:3001",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "http://localhost:5173",  # Vite dev server
            "http://127.0.0.1:5173",
            "http://localhost:4200",  # Angular dev server
            "http://127.0.0.1:4200"
        ]
    
    elif settings.ENVIRONMENT == "staging":
        # Staging origins
        return [
            "https://staging.legalmind.ai",
            "https://test.legalmind.ai",
            "https://dev.legalmind.ai"
        ]
    
    else:
        # Production origins - use configured CORS origins from settings
        production_origins = []
        
        # Add configured CORS origins
        if hasattr(settings.security, 'cors_origins') and settings.security.cors_origins:
            for origin in settings.security.cors_origins:
                if origin != "*":  # Don't allow wildcard in production
                    production_origins.append(origin)
        
        # Default production origins if none configured
        if not production_origins:
            production_origins = [
                "https://legalmind.ai",
                "https://www.legalmind.ai",
                "https://app.legalmind.ai",
                "https://api.legalmind.ai"
            ]
            logger.warning("Using default production CORS origins. Configure CORS_ORIGINS environment variable.")
        
        return production_origins


def get_allowed_methods() -> List[str]:
    """Get allowed HTTP methods"""
    
    if settings.ENVIRONMENT == "development":
        # More permissive for development
        return ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]
    else:
        # Production methods
        return ["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"]


def get_allowed_headers() -> List[str]:
    """Get allowed request headers"""
    
    base_headers = [
        # Standard headers
        "Accept",
        "Accept-Language", 
        "Content-Language",
        "Content-Type",
        
        # Authentication
        "Authorization",
        "X-API-Key",
        
        # Custom headers
        "X-Requested-With",
        "X-Request-ID",
        "X-Session-ID",
        "X-User-Agent",
        
        # File upload headers
        "Content-Range",
        "X-Content-Range",
        "X-File-Name",
        "X-File-Size",
        
        # Cache control
        "Cache-Control",
        "If-Modified-Since",
        "If-None-Match"
    ]
    
    # Add development-specific headers
    if settings.ENVIRONMENT == "development":
        base_headers.extend([
            "X-Debug-Mode",
            "X-Test-Case",
            "X-Mock-Response"
        ])
    
    return base_headers


def get_exposed_headers() -> List[str]:
    """Get headers exposed to client"""
    
    return [
        # Pagination and counting
        "X-Total-Count",
        "X-Page-Count",
        "X-Current-Page",
        "X-Per-Page",
        "Content-Range",
        
        # Processing information
        "X-Processing-Time",
        "X-Processing-Status", 
        "X-Document-ID",
        "X-Session-ID",
        "X-Request-ID",
        
        # Rate limiting
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
        
        # API versioning
        "X-API-Version",
        
        # Content information
        "Content-Disposition",
        "Content-Length",
        "Last-Modified",
        "ETag"
    ]


class CORSOriginValidator:
    """Validate CORS origins with pattern matching"""
    
    def __init__(self, allowed_origins: List[str]):
        self.allowed_origins = allowed_origins
        self.origin_patterns = []
        
        # Compile regex patterns for wildcard origins
        for origin in allowed_origins:
            if "*" in origin:
                # Convert wildcard pattern to regex
                pattern = origin.replace("*", ".*")
                pattern = f"^{pattern}$"
                self.origin_patterns.append(re.compile(pattern))
    
    def is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed"""
        if not origin:
            return False
        
        # Check exact matches first
        if origin in self.allowed_origins:
            return True
        
        # Check wildcard patterns
        for pattern in self.origin_patterns:
            if pattern.match(origin):
                return True
        
        return False


def setup_cors(app: FastAPI) -> None:
    """
    Setup comprehensive CORS and security middleware
    
    Args:
        app: FastAPI application instance
    """
    
    logger.info(f"Setting up CORS for {settings.ENVIRONMENT} environment")
    
    # Get configuration
    allowed_origins = get_allowed_origins()
    allowed_methods = get_allowed_methods()
    allowed_headers = get_allowed_headers()
    exposed_headers = get_exposed_headers()
    
    # Log CORS configuration (mask sensitive origins in production)
    if settings.ENVIRONMENT == "development":
        logger.info(f"Allowed origins: {allowed_origins}")
    else:
        origin_count = len(allowed_origins)
        logger.info(f"Configured {origin_count} allowed origins for production")
    
    logger.info(f"Allowed methods: {allowed_methods}")
    logger.info(f"Exposed headers: {len(exposed_headers)} headers")
    
    # Setup trusted hosts middleware (production security)
    if settings.ENVIRONMENT == "production" and hasattr(settings.security, 'trusted_hosts'):
        if settings.security.trusted_hosts:
            logger.info(f"Setting up trusted hosts: {len(settings.security.trusted_hosts)} hosts")
            app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=settings.security.trusted_hosts
            )
    
    # Setup security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Setup CORS middleware
    cors_config = {
        "allow_origins": allowed_origins,
        "allow_credentials": True,
        "allow_methods": allowed_methods,
        "allow_headers": allowed_headers,
        "expose_headers": exposed_headers,
        "max_age": 86400 if settings.ENVIRONMENT == "production" else 3600,  # Cache preflight longer in prod
    }
    
    # Add origin validator for complex patterns
    origin_validator = CORSOriginValidator(allowed_origins)
    
    # Custom CORS middleware with validation
    class CustomCORSMiddleware(CORSMiddleware):
        def is_allowed_origin(self, origin: str) -> bool:
            return origin_validator.is_origin_allowed(origin)
    
    app.add_middleware(CustomCORSMiddleware, **cors_config)
    
    logger.info("CORS middleware setup completed successfully")


def setup_development_cors(app: FastAPI) -> None:
    """
    Setup permissive CORS for development (use with caution)
    
    Args:
        app: FastAPI application instance
    """
    
    logger.warning("Setting up permissive CORS for development - NOT for production use")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins - DEVELOPMENT ONLY
        allow_credentials=True,
        allow_methods=["*"],  # Allow all methods
        allow_headers=["*"],  # Allow all headers
        expose_headers=get_exposed_headers(),
        max_age=3600,
    )
    
    logger.warning("Permissive CORS enabled - ensure this is not used in production")


def setup_restricted_cors(app: FastAPI, allowed_origins: List[str]) -> None:
    """
    Setup highly restricted CORS for specific use cases
    
    Args:
        app: FastAPI application instance
        allowed_origins: Specific list of allowed origins
    """
    
    logger.info("Setting up restricted CORS configuration")
    
    # Very restrictive headers
    restricted_headers = [
        "Accept",
        "Content-Type",
        "Authorization",
        "X-API-Key"
    ]
    
    restricted_methods = ["GET", "POST", "OPTIONS"]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,  # No credentials for restricted mode
        allow_methods=restricted_methods,
        allow_headers=restricted_headers,
        expose_headers=["X-Request-ID", "X-Processing-Time"],
        max_age=7200,  # Longer cache for restricted mode
    )
    
    logger.info(f"Restricted CORS setup completed for {len(allowed_origins)} origins")


def validate_cors_config() -> bool:
    """
    Validate CORS configuration for security issues
    
    Returns:
        True if configuration is valid, False otherwise
    """
    
    issues = []
    
    # Check for wildcard origins in production
    if settings.ENVIRONMENT == "production":
        allowed_origins = get_allowed_origins()
        
        if "*" in allowed_origins:
            issues.append("Wildcard origin (*) not allowed in production")
        
        for origin in allowed_origins:
            if not origin.startswith("https://") and not origin.startswith("http://localhost"):
                issues.append(f"Non-HTTPS origin in production: {origin}")
    
    # Check for overly permissive headers
    allowed_headers = get_allowed_headers()
    if "*" in allowed_headers and settings.ENVIRONMENT == "production":
        issues.append("Wildcard headers (*) not recommended in production")
    
    # Log issues
    if issues:
        for issue in issues:
            logger.warning(f"CORS Security Issue: {issue}")
        return False
    
    logger.info("CORS configuration validation passed")
    return True


# Utility functions for dynamic CORS management
def add_cors_origin(app: FastAPI, origin: str) -> bool:
    """
    Dynamically add a CORS origin (use with caution)
    
    Args:
        app: FastAPI application instance
        origin: Origin to add
        
    Returns:
        True if added successfully
    """
    
    try:
        # Find CORS middleware
        for middleware in app.user_middleware:
            if middleware.cls == CORSMiddleware:
                current_origins = list(middleware.kwargs.get("allow_origins", []))
                if origin not in current_origins:
                    current_origins.append(origin)
                    middleware.kwargs["allow_origins"] = current_origins
                    logger.info(f"Added CORS origin: {origin}")
                    return True
                else:
                    logger.info(f"CORS origin already exists: {origin}")
                    return True
        
        logger.warning("CORS middleware not found, cannot add origin")
        return False
        
    except Exception as e:
        logger.error(f"Error adding CORS origin {origin}: {str(e)}")
        return False


def get_cors_info() -> dict:
    """
    Get current CORS configuration information
    
    Returns:
        Dictionary with CORS configuration details
    """
    
    return {
        "environment": settings.ENVIRONMENT,
        "allowed_origins": get_allowed_origins(),
        "allowed_methods": get_allowed_methods(),
        "allowed_headers_count": len(get_allowed_headers()),
        "exposed_headers_count": len(get_exposed_headers()),
        "credentials_allowed": True,
        "max_age": 86400 if settings.ENVIRONMENT == "production" else 3600,
        "security_headers_enabled": True,
        "validation_passed": validate_cors_config()
    }


# Export main setup function and utilities
__all__ = [
    "setup_cors",
    "setup_development_cors", 
    "setup_restricted_cors",
    "validate_cors_config",
    "add_cors_origin",
    "get_cors_info",
    "SecurityHeadersMiddleware",
    "CORSOriginValidator"
]
