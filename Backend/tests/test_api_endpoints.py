"""
Comprehensive API endpoint tests for LegalMind AI
Tests all major API functionality including document upload, chat, voice, and reports
"""

import pytest
import io
import json
from unittest.mock import patch, AsyncMock

class TestDocumentEndpoints:
    """Test document management endpoints"""
    
    def test_document_upload_success(self, client, sample_pdf_content, mock_external_services):
        """Test successful document upload"""
        # Mock successful processing
        mock_external_services['gemini'].analyze_document.return_value = {
            "document_type": "rental_agreement",
            "title": "Test Contract",
            "summary": "Test summary",
            "risk_score": 7.5
        }
        
        files = {"document": ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf")}
        data = {"query": "Analyze this contract", "language": "en"}
        
        response = client.post("/documents/upload", files=files, data=data)
        
        assert response.status_code == 200
        result = response.json()
        assert "document_id" in result
        assert "session_id" in result
        assert result["status"] == "uploaded"
    
    def test_document_upload_invalid_file(self, client):
        """Test upload with invalid file"""
        files = {"document": ("test.txt", io.BytesIO(b"not a pdf"), "text/plain")}
        data = {"query": "Analyze this", "language": "en"}
        
        response = client.post("/documents/upload", files=files, data=data)
        
        assert response.status_code == 400
        assert "File type not allowed" in response.json()["error"]["message"]
    
    def test_document_upload_oversized_file(self, client):
        """Test upload with oversized file"""
        large_content = b"x" * (51 * 1024 * 1024)  # 51MB
        files = {"document": ("large.pdf", io.BytesIO(large_content), "application/pdf")}
        data = {"query": "Analyze this", "language": "en"}
        
        response = client.post("/documents/upload", files=files, data=data)
        
        assert response.status_code == 400
        assert "too large" in response.json()["error"]["message"].lower()
    
    def test_get_document_status(self, client, test_db, sample_document_data):
        """Test getting document status"""
        # Create test document in database
        from app.core.database import DatabaseOperations
        
        document = DatabaseOperations.save_document(test_db, sample_document_data)
        test_db.commit()
        
        response = client.get(f"/documents/{document.id}/status")
        
        assert response.status_code == 200
        result = response.json()
        assert result["document_id"] == document.id
        assert "status" in result
        assert "progress" in result
    
    def test_get_document_summary(self, client, test_db, sample_document_data):
        """Test getting document summary"""
        from app.core.database import DatabaseOperations
        
        # Create completed document
        sample_document_data["status"] = "completed"
        sample_document_data["summary"] = "Test summary"
        sample_document_data["risk_score"] = 6.5
        
        document = DatabaseOperations.save_document(test_db, sample_document_data)
        test_db.commit()
        
        response = client.get(f"/documents/{document.id}/summary")
        
        assert response.status_code == 200
        result = response.json()
        assert result["document_id"] == document.id
        assert "summary" in result
        assert "risk_score" in result
    
    def test_get_document_library(self, client, mock_user):
        """Test getting user's document library"""
        with patch('app.core.security.get_current_user', return_value=mock_user):
            response = client.get("/documents/library")
        
        assert response.status_code == 200
        result = response.json()
        assert "documents" in result
        assert "total_count" in result
        assert "page" in result


class TestChatEndpoints:
    """Test chat functionality endpoints"""
    
    def test_start_chat_session(self, client, test_db, sample_document_data):
        """Test starting a chat session"""
        from app.core.database import DatabaseOperations
        
        # Create completed document
        sample_document_data["status"] = "completed"
        document = DatabaseOperations.save_document(test_db, sample_document_data)
        test_db.commit()
        
        data = {
            "document_id": document.id,
            "language": "en",
            "preferences": {}
        }
        
        response = client.post("/chat/start", json=data)
        
        assert response.status_code == 200
        result = response.json()
        assert "session_id" in result
        assert result["document_id"] == document.id
    
    def test_send_chat_message(self, client, test_db, sample_document_data, mock_external_services):
        """Test sending chat message"""
        from app.core.database import DatabaseOperations, ChatSession
        
        # Create document and session
        document = DatabaseOperations.save_document(test_db, sample_document_data)
        session = ChatSession(
            document_id=document.id,
            user_id="test_user_123",
            language="en"
        )
        test_db.add(session)
        test_db.commit()
        
        # Mock AI response
        mock_external_services['gemini'].generate_chat_response.return_value = {
            "response": "This contract has moderate risk level.",
            "confidence": 0.85,
            "citations": ["Section 1.2"]
        }
        
        data = {
            "session_id": session.id,
            "document_id": document.id,
            "message": "What are the main risks in this contract?",
            "language": "en"
        }
        
        response = client.post("/chat/message", json=data)
        
        assert response.status_code == 200
        result = response.json()
        assert "response" in result
        assert "confidence" in result
        assert result["confidence"] > 0.5
    
    def test_get_chat_history(self, client, test_db, sample_document_data):
        """Test getting chat history"""
        from app.core.database import DatabaseOperations, ChatSession
        
        # Create document and session
        document = DatabaseOperations.save_document(test_db, sample_document_data)
        session = ChatSession(
            document_id=document.id,
            user_id="test_user_123",
            language="en"
        )
        test_db.add(session)
        test_db.commit()
        
        response = client.get(f"/chat/{session.id}/history")
        
        assert response.status_code == 200
        result = response.json()
        assert "messages" in result
        assert result["session_id"] == session.id


class TestVoiceEndpoints:
    """Test voice synthesis endpoints"""
    
    def test_generate_voice_summary(self, client, test_db, sample_document_data, mock_external_services):
        """Test voice summary generation"""
        from app.core.database import DatabaseOperations
        
        # Create completed document
        sample_document_data["status"] = "completed"
        sample_document_data["summary"] = "Test summary for voice generation"
        document = DatabaseOperations.save_document(test_db, sample_document_data)
        test_db.commit()
        
        # Mock voice generation
        mock_external_services['voice'].generate_speech.return_value = {
            "audio_url": "https://storage.googleapis.com/test/audio.mp3",
            "duration": "00:02:30"
        }
        
        data = {
            "document_id": document.id,
            "language": "en",
            "voice_type": "female",
            "content_type": "summary"
        }
        
        response = client.post("/voice/generate", json=data)
        
        assert response.status_code == 200
        result = response.json()
        assert "voice_id" in result
        assert result["status"] == "initializing"
    
    def test_get_voice_status(self, client, test_db):
        """Test getting voice generation status"""
        from app.api.routes.voice import voice_summaries_db
        
        # Create test voice entry
        voice_id = "test_voice_123"
        voice_summaries_db[voice_id] = {
            "voice_id": voice_id,
            "status": "completed",
            "audio_url": "https://storage.googleapis.com/test/audio.mp3",
            "duration": "00:02:30"
        }
        
        response = client.get(f"/voice/{voice_id}/status")
        
        assert response.status_code == 200
        result = response.json()
        assert result["voice_id"] == voice_id
        assert result["status"] == "completed"


class TestReportEndpoints:
    """Test report generation endpoints"""
    
    def test_generate_report(self, client, test_db, sample_document_data):
        """Test report generation"""
        from app.core.database import DatabaseOperations
        
        # Create completed document
        sample_document_data["status"] = "completed"
        document = DatabaseOperations.save_document(test_db, sample_document_data)
        test_db.commit()
        
        data = {
            "document_id": document.id,
            "template": "detailed",
            "language": "en",
            "export_format": "pdf",
            "include_sections": ["summary", "risks", "recommendations"]
        }
        
        response = client.post("/reports/generate", json=data)
        
        assert response.status_code == 200
        result = response.json()
        assert "report_id" in result
        assert result["status"] == "initializing"
    
    def test_get_report_status(self, client):
        """Test getting report status"""
        from app.api.routes.reports import reports_db
        
        # Create test report entry
        report_id = "test_report_123"
        reports_db[report_id] = {
            "report_id": report_id,
            "status": "completed",
            "download_url": "https://storage.googleapis.com/test/report.pdf"
        }
        
        response = client.get(f"/reports/{report_id}/status")
        
        assert response.status_code == 200
        result = response.json()
        assert result["report_id"] == report_id
        assert result["status"] == "completed"


class TestSystemEndpoints:
    """Test system management endpoints"""
    
    def test_health_check(self, client):
        """Test system health check"""
        response = client.get("/system/health")
        
        assert response.status_code == 200
        result = response.json()
        assert "status" in result
        assert "services" in result
        assert "version" in result
    
    def test_supported_languages(self, client):
        """Test getting supported languages"""
        response = client.get("/system/supported-languages")
        
        assert response.status_code == 200
        result = response.json()
        assert "languages" in result
        assert len(result["languages"]) > 0
        assert "default_language" in result
    
    def test_version_info(self, client):
        """Test getting version information"""
        response = client.get("/system/version")
        
        assert response.status_code == 200
        result = response.json()
        assert "api_version" in result
        assert "features" in result


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_404_not_found(self, client):
        """Test 404 error handling"""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
    
    def test_invalid_document_id(self, client):
        """Test invalid document ID handling"""
        response = client.get("/documents/invalid-id/status")
        assert response.status_code == 404
    
    def test_missing_required_field(self, client):
        """Test missing required field validation"""
        data = {"language": "en"}  # Missing document_id
        response = client.post("/chat/start", json=data)
        assert response.status_code == 422


class TestSecurity:
    """Test security features"""
    
    def test_rate_limiting(self, client):
        """Test rate limiting (if implemented)"""
        # Make multiple rapid requests
        responses = []
        for _ in range(10):
            response = client.get("/system/health")
            responses.append(response.status_code)
        
        # All should succeed with current rate limits
        assert all(status == 200 for status in responses)
    
    def test_malicious_file_upload(self, client):
        """Test malicious file upload prevention"""
        malicious_content = b"<script>alert('xss')</script>"
        files = {"document": ("malicious.pdf", io.BytesIO(malicious_content), "application/pdf")}
        data = {"query": "Analyze this", "language": "en"}
        
        response = client.post("/documents/upload", files=files, data=data)
        
        # Should be rejected due to invalid PDF format
        assert response.status_code == 400
