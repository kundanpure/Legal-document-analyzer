"""
Advanced rate limiting middleware for LegalMind AI
Production-ready with Redis backend, user-based limits, and comprehensive monitoring
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict, deque

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config.settings import get_settings
from config.logging import get_logger
from app.utils.constants import RATE_LIMITS

settings = get_settings()
logger = get_logger(__name__)

# Try to import Redis with fallback to in-memory storage
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
    logger.info("✅ Redis available - using Redis for rate limiting")
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("⚠️ Redis not available - using in-memory rate limiting")


class InMemoryRateLimiter:
    """In-memory rate limiter with sliding window algorithm"""
    
    def __init__(self):
        self.requests = defaultdict(deque)
        self.lock = asyncio.Lock()
    
    async def is_allowed(self, key: str, limit: int, window: int) -> Tuple[bool, int, int]:
        """
        Check if request is allowed under rate limit
        
        Args:
            key: Rate limiting key (e.g., 'ip:endpoint')
            limit: Number of requests allowed
            window: Time window in seconds
            
        Returns:
            Tuple of (is_allowed, remaining_requests, reset_time)
        """
        async with self.lock:
            current_time = time.time()
            cutoff_time = current_time - window
            
            # Clean old entries
            request_times = self.requests[key]
            while request_times and request_times[0] <= cutoff_time:
                request_times.popleft()
            
            # Check if under limit
            current_count = len(request_times)
            
            if current_count >= limit:
                # Calculate reset time (when oldest request expires)
                reset_time = int(request_times[0] + window) if request_times else int(current_time + window)
                return False, 0, reset_time
            
            # Add current request
            request_times.append(current_time)
            remaining = limit - current_count - 1
            reset_time = int(current_time + window)
            
            return True, remaining, reset_time
    
    async def get_stats(self, key: str, window: int) -> Dict[str, Any]:
        """Get rate limiting stats for a key"""
        async with self.lock:
            current_time = time.time()
            cutoff_time = current_time - window
            
            request_times = self.requests[key]
            # Clean old entries
            while request_times and request_times[0] <= cutoff_time:
                request_times.popleft()
            
            return {
                'current_count': len(request_times),
                'first_request': request_times[0] if request_times else None,
                'last_request': request_times[-1] if request_times else None,
                'window_start': cutoff_time,
                'window_end': current_time
            }
    
    async def reset_key(self, key: str) -> bool:
        """Reset rate limit for a specific key"""
        async with self.lock:
            if key in self.requests:
                del self.requests[key]
                return True
            return False
    
    async def cleanup_old_entries(self, max_age: int = 3600):
        """Cleanup old entries to prevent memory leaks"""
        async with self.lock:
            current_time = time.time()
            keys_to_remove = []
            
            for key, request_times in self.requests.items():
                # Remove old entries
                while request_times and request_times[0] <= current_time - max_age:
                    request_times.popleft()
                
                # Remove empty queues
                if not request_times:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.requests[key]
            
            logger.debug(f"Cleaned up {len(keys_to_remove)} empty rate limit entries")


class RedisRateLimiter:
    """Redis-based rate limiter with Lua scripts for atomic operations"""
    
    def __init__(self):
        self.redis_client = None
        self.lua_script = None
        self._setup_redis()
    
    def _setup_redis(self):
        """Setup Redis connection and Lua script"""
        try:
            if settings.redis.url:
                self.redis_client = redis.from_url(settings.redis.url)
            else:
                self.redis_client = redis.Redis(
                    host=settings.redis.host,
                    port=settings.redis.port,
                    db=settings.redis.db,
                    password=settings.redis.password,
                    ssl=settings.redis.ssl,
                    decode_responses=True
                )
            
            # Lua script for atomic rate limiting
            self.lua_script = """
            local key = KEYS[1]
            local window = tonumber(ARGV[1])
            local limit = tonumber(ARGV[2])
            local current_time = tonumber(ARGV[3])
            local cutoff_time = current_time - window
            
            -- Remove old entries
            redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff_time)
            
            -- Get current count
            local current_count = redis.call('ZCARD', key)
            
            if current_count >= limit then
                -- Get reset time (oldest entry + window)
                local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
                local reset_time = current_time + window
                if #oldest > 0 then
                    reset_time = oldest[2] + window
                end
                return {0, 0, reset_time, current_count}
            else
                -- Add current request
                redis.call('ZADD', key, current_time, current_time)
                redis.call('EXPIRE', key, window * 2)  -- Set expiry for cleanup
                
                local remaining = limit - current_count - 1
                local reset_time = current_time + window
                return {1, remaining, reset_time, current_count + 1}
            end
            """
            
        except Exception as e:
            logger.error(f"Failed to setup Redis rate limiter: {str(e)}")
            self.redis_client = None
    
    async def is_allowed(self, key: str, limit: int, window: int) -> Tuple[bool, int, int]:
        """Check if request is allowed using Redis"""
        if not self.redis_client:
            return True, limit - 1, int(time.time() + window)  # Allow if Redis unavailable
        
        try:
            current_time = time.time()
            result = await self.redis_client.eval(
                self.lua_script,
                1,
                f"rate_limit:{key}",
                window,
                limit,
                current_time
            )
            
            is_allowed = bool(result[0])
            remaining = int(result[1])
            reset_time = int(result[2])
            
            return is_allowed, remaining, reset_time
            
        except Exception as e:
            logger.error(f"Redis rate limiting error: {str(e)}")
            return True, limit - 1, int(time.time() + window)  # Allow on error
    
    async def get_stats(self, key: str, window: int) -> Dict[str, Any]:
        """Get rate limiting stats from Redis"""
        if not self.redis_client:
            return {}
        
        try:
            current_time = time.time()
            cutoff_time = current_time - window
            redis_key = f"rate_limit:{key}"
            
            # Clean old entries
            await self.redis_client.zremrangebyscore(redis_key, '-inf', cutoff_time)
            
            # Get stats
            current_count = await self.redis_client.zcard(redis_key)
            all_requests = await self.redis_client.zrange(redis_key, 0, -1, withscores=True)
            
            first_request = min([score for _, score in all_requests]) if all_requests else None
            last_request = max([score for _, score in all_requests]) if all_requests else None
            
            return {
                'current_count': current_count,
                'first_request': first_request,
                'last_request': last_request,
                'window_start': cutoff_time,
                'window_end': current_time,
                'all_requests': len(all_requests)
            }
            
        except Exception as e:
            logger.error(f"Error getting Redis stats: {str(e)}")
            return {}
    
    async def reset_key(self, key: str) -> bool:
        """Reset rate limit for a specific key in Redis"""
        if not self.redis_client:
            return False
        
        try:
            result = await self.redis_client.delete(f"rate_limit:{key}")
            return bool(result)
        except Exception as e:
            logger.error(f"Error resetting Redis key: {str(e)}")
            return False
    
    async def health_check(self) -> bool:
        """Check if Redis is healthy"""
        if not self.redis_client:
            return False
        
        try:
            await self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
            return False


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Advanced rate limiting middleware with multiple strategies"""
    
    def __init__(self, app, rate_limiter=None):
        super().__init__(app)
        self.rate_limiter = rate_limiter or self._create_rate_limiter()
        self.bypass_ips = set()
        self.stats = defaultdict(int)
        self._setup_cleanup_task()
    
    def _create_rate_limiter(self):
        """Create appropriate rate limiter based on Redis availability"""
        if REDIS_AVAILABLE and settings.redis.url:
            return RedisRateLimiter()
        else:
            return InMemoryRateLimiter()
    
    def _setup_cleanup_task(self):
        """Setup periodic cleanup task for in-memory limiter"""
        if isinstance(self.rate_limiter, InMemoryRateLimiter):
            async def cleanup_loop():
                while True:
                    await asyncio.sleep(300)  # Clean every 5 minutes
                    try:
                        await self.rate_limiter.cleanup_old_entries()
                    except Exception as e:
                        logger.error(f"Rate limiter cleanup error: {str(e)}")
            
            # Start cleanup task
            asyncio.create_task(cleanup_loop())
    
    async def dispatch(self, request: Request, call_next):
        """Main rate limiting logic"""
        
        # Skip rate limiting if disabled
        if not settings.security.rate_limit_enabled:
            return await call_next(request)
        
        # Extract client information
        client_info = self._extract_client_info(request)
        
        # Check if IP should be bypassed
        if client_info['ip'] in self.bypass_ips:
            return await call_next(request)
        
        # Get rate limit configuration
        rate_config = self._get_rate_limit_config(request.url.path, request.method)
        
        if not rate_config:
            return await call_next(request)
        
        # Create rate limiting key
        rate_key = self._create_rate_key(client_info, request.url.path, rate_config)
        
        # Check rate limit
        is_allowed, remaining, reset_time = await self.rate_limiter.is_allowed(
            rate_key,
            rate_config['requests'],
            rate_config['window']
        )
        
        # Update stats
        self.stats['total_requests'] += 1
        
        if not is_allowed:
            self.stats['blocked_requests'] += 1
            
            # Log rate limit violation
            logger.warning(
                f"Rate limit exceeded for {client_info['ip']} on {request.url.path}",
                extra={
                    'client_ip': client_info['ip'],
                    'user_id': client_info.get('user_id'),
                    'path': request.url.path,
                    'method': request.method,
                    'rate_key': rate_key,
                    'limit': rate_config['requests'],
                    'window': rate_config['window']
                }
            )
            
            # Return rate limit error
            return self._create_rate_limit_response(
                rate_config,
                remaining,
                reset_time,
                client_info
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers.update(self._create_rate_limit_headers(
            rate_config,
            remaining,
            reset_time
        ))
        
        self.stats['successful_requests'] += 1
        return response
    
    def _extract_client_info(self, request: Request) -> Dict[str, Any]:
        """Extract client information for rate limiting"""
        
        # Get client IP
        client_ip = request.client.host if request.client else 'unknown'
        
        # Check for forwarded IP (load balancer, proxy)
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            client_ip = forwarded_for.split(',')[0].strip()
        
        # Check for real IP
        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            client_ip = real_ip.strip()
        
        # Get user information (if authenticated)
        user_id = None
        session_id = None
        
        # Try to extract from authorization header or session
        auth_header = request.headers.get('authorization')
        if auth_header:
            # Extract user info from JWT or API key
            # This would typically decode JWT to get user_id
            pass
        
        # Try to extract from custom headers
        user_id = request.headers.get('x-user-id')
        session_id = request.headers.get('x-session-id')
        
        return {
            'ip': client_ip,
            'user_id': user_id,
            'session_id': session_id,
            'user_agent': request.headers.get('user-agent', 'unknown')
        }
    
    def _get_rate_limit_config(self, path: str, method: str) -> Optional[Dict[str, Any]]:
        """Get rate limit configuration for path and method"""
        
        # Document upload endpoints
        if '/upload-document' in path or '/documents' in path and method == 'POST':
            config = RATE_LIMITS.get('document_upload', {})
            
        # Chat endpoints
        elif '/chat' in path:
            config = RATE_LIMITS.get('chat_message', {})
            
        # Report generation endpoints
        elif '/generate-report' in path or '/reports' in path:
            config = RATE_LIMITS.get('report_generation', {})
            
        # Voice generation endpoints
        elif '/generate-voice' in path or '/voice' in path:
            config = RATE_LIMITS.get('voice_generation', {})
            
        # Translation endpoints
        elif '/translate' in path:
            config = RATE_LIMITS.get('translation', {})
            
        # General API endpoints
        elif path.startswith('/api/'):
            config = RATE_LIMITS.get('general_api', {})
            
        else:
            return None
        
        # Ensure config has required fields
        if config and 'requests' in config and 'window' in config:
            return config
        
        return None
    
    def _create_rate_key(self, client_info: Dict[str, Any], path: str, rate_config: Dict[str, Any]) -> str:
        """Create unique rate limiting key"""
        
        # Use user ID if available for authenticated requests
        if client_info.get('user_id'):
            identifier = f"user:{client_info['user_id']}"
        else:
            identifier = f"ip:{client_info['ip']}"
        
        # Include path for endpoint-specific limiting
        path_component = path.split('/')[-1] if '/' in path else path
        
        # Create composite key
        key_parts = [identifier, path_component]
        
        # Add burst limiting for specific endpoints
        if rate_config.get('burst'):
            key_parts.append('burst')
        
        return ':'.join(key_parts)
    
    def _create_rate_limit_response(
        self,
        rate_config: Dict[str, Any],
        remaining: int,
        reset_time: int,
        client_info: Dict[str, Any]
    ) -> JSONResponse:
        """Create rate limit exceeded response"""
        
        retry_after = max(1, reset_time - int(time.time()))
        
        error_response = {
            'success': False,
            'error': {
                'code': 'RATE_LIMIT_EXCEEDED',
                'message': 'Rate limit exceeded. Please slow down your requests.',
                'details': {
                    'limit': rate_config['requests'],
                    'window': rate_config['window'],
                    'retry_after': retry_after,
                    'reset_time': reset_time
                }
            },
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'request_id': str(int(time.time() * 1000000))  # Simple request ID
        }
        
        # Add user-friendly message
        if rate_config['window'] < 3600:
            time_unit = f"{rate_config['window']} seconds"
        else:
            time_unit = f"{rate_config['window'] // 3600} hour(s)"
        
        error_response['error']['user_message'] = f"You've made too many requests. Limit: {rate_config['requests']} per {time_unit}. Please wait {retry_after} seconds."
        
        headers = self._create_rate_limit_headers(rate_config, remaining, reset_time)
        headers['Retry-After'] = str(retry_after)
        
        return JSONResponse(
            status_code=429,
            content=error_response,
            headers=headers
        )
    
    def _create_rate_limit_headers(
        self,
        rate_config: Dict[str, Any],
        remaining: int,
        reset_time: int
    ) -> Dict[str, str]:
        """Create rate limit headers for response"""
        
        return {
            'X-RateLimit-Limit': str(rate_config['requests']),
            'X-RateLimit-Remaining': str(max(0, remaining)),
            'X-RateLimit-Reset': str(reset_time),
            'X-RateLimit-Window': str(rate_config['window']),
            'X-RateLimit-Policy': f"{rate_config['requests']};w={rate_config['window']}"
        }
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        
        health_status = True
        if hasattr(self.rate_limiter, 'health_check'):
            health_status = await self.rate_limiter.health_check()
        
        return {
            'total_requests': self.stats['total_requests'],
            'successful_requests': self.stats['successful_requests'],
            'blocked_requests': self.stats['blocked_requests'],
            'block_rate': (
                self.stats['blocked_requests'] / max(self.stats['total_requests'], 1) * 100
            ),
            'limiter_type': type(self.rate_limiter).__name__,
            'limiter_healthy': health_status,
            'bypass_ips': len(self.bypass_ips)
        }
    
    def add_bypass_ip(self, ip: str):
        """Add IP to bypass list"""
        self.bypass_ips.add(ip)
        logger.info(f"Added IP to rate limit bypass: {ip}")
    
    def remove_bypass_ip(self, ip: str):
        """Remove IP from bypass list"""
        self.bypass_ips.discard(ip)
        logger.info(f"Removed IP from rate limit bypass: {ip}")


def setup_rate_limiting(app: FastAPI) -> RateLimitingMiddleware:
    """
    Setup rate limiting middleware for the application
    
    Args:
        app: FastAPI application instance
        
    Returns:
        RateLimitingMiddleware instance for management
    """
    
    logger.info("Setting up rate limiting middleware")
    
    # Create rate limiter
    if REDIS_AVAILABLE and settings.redis.url:
        logger.info("Using Redis-based rate limiting")
    else:
        logger.info("Using in-memory rate limiting")
    
    # Create and add middleware
    rate_limiting_middleware = RateLimitingMiddleware(app)
    app.add_middleware(RateLimitingMiddleware)
    
    logger.info("Rate limiting middleware setup completed")
    return rate_limiting_middleware


# Utility functions
async def reset_rate_limit(client_identifier: str, endpoint: str) -> bool:
    """Reset rate limit for specific client and endpoint"""
    # This would typically be called from an admin endpoint
    pass


def get_rate_limit_info() -> Dict[str, Any]:
    """Get rate limit configuration information"""
    return {
        'enabled': settings.security.rate_limit_enabled,
        'backend': 'Redis' if REDIS_AVAILABLE and settings.redis.url else 'In-Memory',
        'limits': RATE_LIMITS,
        'redis_available': REDIS_AVAILABLE,
        'redis_configured': bool(settings.redis.url)
    }


# Export main components
__all__ = [
    'setup_rate_limiting',
    'RateLimitingMiddleware',
    'InMemoryRateLimiter',
    'RedisRateLimiter',
    'get_rate_limit_info'
]
