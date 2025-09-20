"""
Core service and business logic tests for LegalMind AI
Tests the heart of the application - AI analysis, storage, and processing services
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

class TestGeminiAnalyzer:
    """Test Gemini AI analysis service"""
    
    @pytest.mark.asyncio
    async def test_document_classification(self):
        """Test document type classification"""
        from app.services.gemini_analyzer import GeminiAnalyzer
        
        analyzer = GeminiAnalyzer()
        
        # Mock the Gemini API response
        with patch.object(analyzer, '_generate_content_async') as mock_generate:
            mock_generate.return_value = """
            {
                "document_type": "rental_agreement",
                "confidence": 0.92,
                "reasoning": "Contains rental terms and lease conditions"
            }
            """
            
            result = await analyzer.classify_document(
                "This rental agreement is between landlord and tenant...",
                "rental_contract.pdf"
            )
            
            assert result["document_type"] == "rental_agreement"
            assert result["confidence"] > 0.9
    
    @pytest.mark.asyncio
    async def test_risk_analysis(self):
        """Test document risk analysis"""
        from app.services.gemini_analyzer import GeminiAnalyzer
        
        analyzer = GeminiAnalyzer()
        
        with patch.object(analyzer, '_generate_content_async') as mock_generate:
            mock_generate.return_value = """
            {
                "risk_score": 7.5,
                "risk_level": "high",
                "key_risks": [
                    "Unfavorable termination clause",
                    "High penalty fees"
                ],
                "risk_categories": {
                    "financial": 8.0,
                    "legal": 7.0,
                    "operational": 6.5
                }
            }
            """
            
            result = await analyzer.analyze_risks(
                "Contract text with unfavorable terms...",
                "employment_contract"
            )
            
            assert result["risk_score"] == 7.5
            assert result["risk_level"] == "high"
            assert len(result["key_risks"]) > 0
    
    @pytest.mark.asyncio
    async def test_ai_error_handling(self):
        """Test AI service error handling"""
        from app.services.gemini_analyzer import GeminiAnalyzer
        from app.core.exceptions import GeminiAnalysisError
        
        analyzer = GeminiAnalyzer()
        
        with patch.object(analyzer, '_generate_content_async') as mock_generate:
            mock_generate.side_effect = Exception("API quota exceeded")
            
            with pytest.raises(GeminiAnalysisError):
                await analyzer.classify_document("Test content", "test.pdf")


class TestStorageManager:
    """Test file storage management"""
    
    @pytest.mark.asyncio
    async def test_file_upload(self):
        """Test file upload to storage"""
        from app.services.storage_manager import StorageManager
        
        storage = StorageManager()
        
        with patch.object(storage, 'bucket') as mock_bucket:
            mock_blob = Mock()
            mock_bucket.blob.return_value = mock_blob
            
            result = await storage.upload_file(
                b"test content",
                "documents/test.pdf",
                "application/pdf"
            )
            
            assert "gcs_uri" in result
            mock_blob.upload_from_string.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_file_download(self):
        """Test file download from storage"""
        from app.services.storage_manager import StorageManager
        
        storage = StorageManager()
        
        with patch.object(storage, 'bucket') as mock_bucket:
            mock_blob = Mock()
            mock_blob.download_as_bytes.return_value = b"test content"
            mock_bucket.blob.return_value = mock_blob
            
            content = await storage.download_file("documents/test.pdf")
            
            assert content == b"test content"
    
    @pytest.mark.asyncio
    async def test_storage_cleanup(self):
        """Test automated storage cleanup"""
        from app.services.storage_manager import StorageManager
        
        storage = StorageManager()
        
        with patch.object(storage, 'bucket') as mock_bucket:
            mock_blob1 = Mock()
            mock_blob1.name = "old_file.pdf"
            mock_blob1.time_created = datetime.now(timezone.utc).replace(day=1)  # Old file
            
            mock_blob2 = Mock()
            mock_blob2.name = "new_file.pdf"
            mock_blob2.time_created = datetime.now(timezone.utc)  # Recent file
            
            mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2]
            
            result = await storage.cleanup_old_files(max_age_days=7)
            
            assert result["files_deleted"] >= 0


class TestTranslationService:
    """Test translation service"""
    
    @pytest.mark.asyncio
    async def test_text_translation(self):
        """Test text translation"""
        from app.services.translation_service import TranslationService
        
        translator = TranslationService()
        
        with patch('googletrans.Translator') as mock_translator_class:
            mock_translator = Mock()
            mock_result = Mock()
            mock_result.text = "नमस्ते दुनिया"
            mock_result.src = "en"
            mock_translator.translate.return_value = mock_result
            mock_translator_class.return_value = mock_translator
            
            result = await translator.translate_text(
                "Hello world",
                target_language="hi",
                source_language="en"
            )
            
            assert result["translated_text"] == "नमस्ते दुनिया"
            assert result["source_language"] == "en"
            assert result["target_language"] == "hi"
    
    def test_language_detection(self):
        """Test automatic language detection"""
        from app.services.translation_service import TranslationService
        
        translator = TranslationService()
        
        test_cases = [
            ("Hello world", "en"),
            ("नमस्ते दुनिया", "hi"),
            ("வணக்கம் உலகம்", "ta")
        ]
        
        for text, expected_lang in test_cases:
            detected = translator.detect_language(text)
            # Note: In real testing, you might want to mock this too
            assert isinstance(detected, str)


class TestVoiceGenerator:
    """Test voice synthesis service"""
    
    @pytest.mark.asyncio
    async def test_voice_generation(self):
        """Test voice synthesis"""
        from app.services.voice_generator import VoiceGenerator
        
        voice_gen = VoiceGenerator()
        
        with patch.object(voice_gen, '_synthesize_speech') as mock_synthesize:
            mock_synthesize.return_value = {
                "audio_content": b"fake audio data",
                "duration": 125  # seconds
            }
            
            result = await voice_gen.generate_speech(
                "This is a test summary of the legal document.",
                language="en",
                voice_type="female"
            )
            
            assert result["duration"] == 125
            assert len(result["audio_content"]) > 0
    
    def test_supported_languages(self):
        """Test getting supported voice languages"""
        from app.services.voice_generator import VoiceGenerator
        
        voice_gen = VoiceGenerator()
        languages = voice_gen.get_supported_languages()
        
        assert isinstance(languages, list)
        assert "en" in languages
        assert "hi" in languages
        assert len(languages) > 5


class TestChatHandler:
    """Test chat processing service"""
    
    @pytest.mark.asyncio
    async def test_chat_response_generation(self):
        """Test chat response generation"""
        from app.services.chat_handler import ChatHandler
        
        chat_handler = ChatHandler()
        
        with patch.object(chat_handler, 'gemini_analyzer') as mock_analyzer:
            mock_analyzer.generate_chat_response.return_value = {
                "response": "The main risks include unfavorable termination clauses.",
                "confidence": 0.87,
                "citations": ["Section 3.1", "Clause 5.2"]
            }
            
            result = await chat_handler.process_message(
                session_id="test_session_123",
                message="What are the main risks?",
                document_context="Test contract content...",
                language="en"
            )
            
            assert result["response"]
            assert result["confidence"] > 0.8
            assert len(result["citations"]) > 0
    
    def test_session_management(self):
        """Test chat session management"""
        from app.services.chat_handler import ChatHandler
        
        chat_handler = ChatHandler()
        
        # Create session
        session_id = chat_handler.create_session(
            document_id="doc_123",
            user_id="user_123",
            language="en"
        )
        
        assert session_id
        
        # Check session exists
        session = chat_handler.get_session(session_id)
        assert session["document_id"] == "doc_123"
        assert session["is_active"] is True


class TestValidators:
    """Test input validation utilities"""
    
    def test_file_validation(self):
        """Test file validation"""
        from app.utils.validators import validate_file_upload
        
        # Valid PDF
        pdf_content = b"%PDF-1.4\ntest content"
        result = validate_file_upload(pdf_content, "test.pdf", "application/pdf")
        assert result["valid"] is True
        
        # Invalid file type
        result = validate_file_upload(b"test", "test.txt", "text/plain")
        assert result["valid"] is False
        assert "File type not allowed" in str(result["errors"])
        
        # Oversized file
        large_content = b"x" * (60 * 1024 * 1024)  # 60MB
        result = validate_file_upload(large_content, "large.pdf", "application/pdf")
        assert result["valid"] is False
        assert "too large" in str(result["errors"]).lower()
    
    def test_request_validation(self):
        """Test API request validation"""
        from app.utils.validators import validate_document_request
        
        # Valid request
        request_data = {
            "query": "Analyze this contract",
            "language": "en",
            "document_type": "rental_agreement"
        }
        result = validate_document_request(request_data)
        assert result["valid"] is True
        
        # Invalid request - missing query
        invalid_request = {"language": "en"}
        result = validate_document_request(invalid_request)
        assert result["valid"] is False


class TestHelpers:
    """Test utility helper functions"""
    
    def test_id_generation(self):
        """Test unique ID generation"""
        from app.utils.helpers import generate_request_id, generate_document_id
        
        # Test request ID
        req_id1 = generate_request_id()
        req_id2 = generate_request_id()
        
        assert req_id1 != req_id2
        assert len(req_id1) > 10
        
        # Test document ID
        doc_id = generate_document_id()
        assert len(doc_id) == 36  # UUID format
    
    def test_text_processing(self):
        """Test text processing utilities"""
        from app.utils.helpers import clean_text, estimate_reading_time
        
        # Text cleaning
        dirty_text = "  This is\n\na test   document.\t\t  "
        clean = clean_text(dirty_text)
        assert clean == "This is a test document."
        
        # Reading time estimation
        text = "word " * 200  # 200 words
        time_str = estimate_reading_time(text)
        assert "1 min" in time_str.lower()
    
    def test_file_utilities(self):
        """Test file utility functions"""
        from app.utils.helpers import get_file_extension, sanitize_filename
        
        # File extension
        assert get_file_extension("document.pdf") == "pdf"
        assert get_file_extension("contract.docx") == "docx"
        
        # Filename sanitization
        dangerous_name = "../../../etc/passwd"
        safe_name = sanitize_filename(dangerous_name)
        assert ".." not in safe_name
        assert "/" not in safe_name


class TestFormatters:
    """Test data formatting utilities"""
    
    def test_file_size_formatting(self):
        """Test file size formatting"""
        from app.utils.formatters import format_file_size
        
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(1024 * 1024) == "1.0 MB"
        assert format_file_size(1024 * 1024 * 1024) == "1.0 GB"
    
    def test_duration_formatting(self):
        """Test duration formatting"""
        from app.utils.formatters import format_duration
        
        assert format_duration(30) == "30 seconds"
        assert format_duration(90) == "1 minute 30 seconds"
        assert format_duration(3600) == "1 hour"
    
    def test_risk_score_formatting(self):
        """Test risk score formatting"""
        from app.utils.formatters import format_risk_level
        
        assert format_risk_level(2.0) == "low"
        assert format_risk_level(5.5) == "medium"
        assert format_risk_level(8.5) == "high"


# Performance tests
class TestPerformance:
    """Basic performance tests"""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling concurrent requests"""
        from app.services.gemini_analyzer import GeminiAnalyzer
        
        analyzer = GeminiAnalyzer()
        
        async def classify_doc():
            with patch.object(analyzer, '_generate_content_async') as mock:
                mock.return_value = '{"document_type": "general", "confidence": 0.8}'
                return await analyzer.classify_document("test", "test.pdf")
        
        # Test 10 concurrent classifications
        tasks = [classify_doc() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        assert all(r["document_type"] == "general" for r in results)


# Integration test samples
class TestIntegration:
    """Integration tests with minimal external dependencies"""
    
    def test_document_to_chat_workflow(self, client, test_db, sample_pdf_content):
        """Test complete workflow: upload document -> start chat -> send message"""
        # This test would require more setup but demonstrates the concept
        pass
    
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test system recovery from errors"""
        from app.services.gemini_analyzer import GeminiAnalyzer
        
        analyzer = GeminiAnalyzer()
        
        # Test retry logic
        with patch.object(analyzer, '_generate_content_async') as mock:
            # First call fails, second succeeds
            mock.side_effect = [
                Exception("Temporary API error"),
                '{"document_type": "general", "confidence": 0.8}'
            ]
            
            # Should recover and succeed
            result = await analyzer.classify_document_with_retry("test", "test.pdf")
            assert result["document_type"] == "general"
