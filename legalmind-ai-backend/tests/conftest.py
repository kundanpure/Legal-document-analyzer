"""
Test configuration and shared fixtures for LegalMind AI
Provides reusable test setup, mock data, and utility functions
"""

import pytest
import asyncio
import tempfile
import os
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from main import app
from app.core.database import Base, get_db
from config.settings import get_settings

# Test configuration
TEST_DATABASE_URL = "sqlite:///./test_legalmind.db"

# Test client setup
@pytest.fixture(scope="session")
def test_app():
    """Create test FastAPI application"""
    return app

@pytest.fixture(scope="session") 
def client(test_app):
    """Create test client"""
    return TestClient(test_app)

# Database fixtures
@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine"""
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def test_db(test_engine):
    """Create test database session"""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    
    def override_get_db():
        yield session
    
    app.dependency_overrides[get_db] = override_get_db
    yield session
    session.close()
    app.dependency_overrides.clear()

# Test data fixtures
@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing"""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000074 00000 n \n0000000120 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n179\n%%EOF"

@pytest.fixture
def sample_document_data():
    """Sample document data for testing"""
    return {
        "filename": "test_contract.pdf",
        "content": "This is a test rental agreement between landlord and tenant...",
        "document_type": "rental_agreement",
        "user_id": "test_user_123"
    }

@pytest.fixture
def mock_gemini_response():
    """Mock Gemini AI response"""
    return {
        "document_type": "rental_agreement",
        "title": "Test Rental Agreement",
        "summary": "A standard rental agreement with moderate risk level.",
        "risk_score": 6.5,
        "key_risks": ["Late payment penalties", "Security deposit conditions"],
        "recommendations": ["Review payment terms", "Clarify maintenance responsibilities"]
    }

@pytest.fixture
def mock_user():
    """Mock user for testing"""
    return {
        "user_id": "test_user_123",
        "email": "test@example.com",
        "role": "user",
        "is_active": True,
        "subscription_tier": "free"
    }

# Mock external services
@pytest.fixture(autouse=True)
def mock_external_services():
    """Mock external services for testing"""
    with patch('app.services.gemini_analyzer.GeminiAnalyzer') as mock_gemini, \
         patch('app.services.storage_manager.StorageManager') as mock_storage, \
         patch('app.services.voice_generator.VoiceGenerator') as mock_voice:
        
        # Mock Gemini
        mock_gemini_instance = AsyncMock()
        mock_gemini.return_value = mock_gemini_instance
        
        # Mock Storage
        mock_storage_instance = AsyncMock()
        mock_storage.return_value = mock_storage_instance
        
        # Mock Voice
        mock_voice_instance = AsyncMock()
        mock_voice.return_value = mock_voice_instance
        
        yield {
            'gemini': mock_gemini_instance,
            'storage': mock_storage_instance,
            'voice': mock_voice_instance
        }

# Async test support
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# Test utilities
def wait_for_processing(client, document_id, timeout=30):
    """Wait for document processing to complete"""
    import time
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        response = client.get(f"/documents/{document_id}/status")
        if response.status_code == 200:
            status = response.json().get("status")
            if status in ["completed", "failed"]:
                return status
        time.sleep(1)
    
    raise TimeoutError(f"Document processing timeout after {timeout}s")

def create_test_file(content: bytes, filename: str = "test.pdf"):
    """Create temporary test file"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}")
    temp_file.write(content)
    temp_file.close()
    return temp_file.name

# Cleanup
@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after each test"""
    yield
    # Clean up temporary files, clear caches, etc.
    pass
"""
Test configuration and shared fixtures for LegalMind AI
Provides reusable test setup, mock data, and utility functions
"""

import pytest
import asyncio
import tempfile
import os
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from main import app
from app.core.database import Base, get_db
from config.settings import get_settings

# Test configuration
TEST_DATABASE_URL = "sqlite:///./test_legalmind.db"

# Test client setup
@pytest.fixture(scope="session")
def test_app():
    """Create test FastAPI application"""
    return app

@pytest.fixture(scope="session") 
def client(test_app):
    """Create test client"""
    return TestClient(test_app)

# Database fixtures
@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine"""
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def test_db(test_engine):
    """Create test database session"""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    
    def override_get_db():
        yield session
    
    app.dependency_overrides[get_db] = override_get_db
    yield session
    session.close()
    app.dependency_overrides.clear()

# Test data fixtures
@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing"""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000074 00000 n \n0000000120 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n179\n%%EOF"

@pytest.fixture
def sample_document_data():
    """Sample document data for testing"""
    return {
        "filename": "test_contract.pdf",
        "content": "This is a test rental agreement between landlord and tenant...",
        "document_type": "rental_agreement",
        "user_id": "test_user_123"
    }

@pytest.fixture
def mock_gemini_response():
    """Mock Gemini AI response"""
    return {
        "document_type": "rental_agreement",
        "title": "Test Rental Agreement",
        "summary": "A standard rental agreement with moderate risk level.",
        "risk_score": 6.5,
        "key_risks": ["Late payment penalties", "Security deposit conditions"],
        "recommendations": ["Review payment terms", "Clarify maintenance responsibilities"]
    }

@pytest.fixture
def mock_user():
    """Mock user for testing"""
    return {
        "user_id": "test_user_123",
        "email": "test@example.com",
        "role": "user",
        "is_active": True,
        "subscription_tier": "free"
    }

# Mock external services
@pytest.fixture(autouse=True)
def mock_external_services():
    """Mock external services for testing"""
    with patch('app.services.gemini_analyzer.GeminiAnalyzer') as mock_gemini, \
         patch('app.services.storage_manager.StorageManager') as mock_storage, \
         patch('app.services.voice_generator.VoiceGenerator') as mock_voice:
        
        # Mock Gemini
        mock_gemini_instance = AsyncMock()
        mock_gemini.return_value = mock_gemini_instance
        
        # Mock Storage
        mock_storage_instance = AsyncMock()
        mock_storage.return_value = mock_storage_instance
        
        # Mock Voice
        mock_voice_instance = AsyncMock()
        mock_voice.return_value = mock_voice_instance
        
        yield {
            'gemini': mock_gemini_instance,
            'storage': mock_storage_instance,
            'voice': mock_voice_instance
        }

# Async test support
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# Test utilities
def wait_for_processing(client, document_id, timeout=30):
    """Wait for document processing to complete"""
    import time
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        response = client.get(f"/documents/{document_id}/status")
        if response.status_code == 200:
            status = response.json().get("status")
            if status in ["completed", "failed"]:
                return status
        time.sleep(1)
    
    raise TimeoutError(f"Document processing timeout after {timeout}s")

def create_test_file(content: bytes, filename: str = "test.pdf"):
    """Create temporary test file"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}")
    temp_file.write(content)
    temp_file.close()
    return temp_file.name

# Cleanup
@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after each test"""
    yield
    # Clean up temporary files, clear caches, etc.
    pass
