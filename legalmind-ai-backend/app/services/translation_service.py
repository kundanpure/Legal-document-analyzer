"""
Google Cloud Translation API integration for multilingual support
Enhanced with fallbacks and comprehensive error handling
"""

import asyncio
import json
import re
from dotenv import load_dotenv
load_dotenv() 
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone

from config.logging import get_logger

logger = get_logger(__name__)

# Try to import Google Cloud Translation with fallback
try:
    from google.cloud import translate_v2 as translate
    GOOGLE_TRANSLATE_AVAILABLE = True
    logger.info("✅ Google Cloud Translation available - full translation services enabled")
except ImportError:
    GOOGLE_TRANSLATE_AVAILABLE = False
    logger.warning("⚠️ Google Cloud Translation not available - using mock translation service")

# Try to import settings with fallback
try:
    from config.settings import get_settings
    settings = get_settings()
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False
    logger.warning("⚠️ Settings not available - using default configuration")
    # Mock settings
    class MockSettings:
        GOOGLE_TRANSLATE_PROJECT_ID = "default-project"
        DEFAULT_LANGUAGE = "en"
    settings = MockSettings()

# Custom exceptions
class TranslationError(Exception):
    """Exception raised for translation-related errors"""
    pass

# Mock translation client for fallback
class MockTranslateClient:
    """Mock translation client when Google Cloud Translation isn't available"""
    
    def __init__(self):
        self.logger = logger
        # Basic word mappings for common legal terms
        self.basic_translations = {
            'hi': {
                'agreement': 'समझौता',
                'contract': 'अनुबंध',
                'payment': 'भुगतान',
                'liability': 'देयता',
                'termination': 'समाप्ति',
                'legal': 'कानूनी',
                'document': 'दस्तावेज़',
                'rental': 'किराया',
                'loan': 'ऋण',
                'employment': 'रोजगार'
            },
            'ta': {
                'agreement': 'ஒப்பந்தம்',
                'contract': 'ஒப்பந்தம்',
                'payment': 'கட்டணம்',
                'liability': 'பொறுப்பு',
                'termination': 'முடிவு',
                'legal': 'சட்ட',
                'document': 'ஆவணம்',
                'rental': 'வாடகை',
                'loan': 'கடன்',
                'employment': 'வேலைவாய்ப்பு'
            },
            'te': {
                'agreement': 'ఒప్పందం',
                'contract': 'ఒప్పందం',
                'payment': 'చెల్లింపు',
                'liability': 'బాధ్యత',
                'termination': 'ముగింపు',
                'legal': 'చట్టపరమైన',
                'document': 'పత్రం',
                'rental': 'అద్దె',
                'loan': 'రుణం',
                'employment': 'ఉపాధి'
            },
            'bn': {
                'agreement': 'চুক্তি',
                'contract': 'চুক্তি',
                'payment': 'পেমেন্ট',
                'liability': 'দায়বদ্ধতা',
                'termination': 'সমাপ্তি',
                'legal': 'আইনি',
                'document': 'নথি',
                'rental': 'ভাড়া',
                'loan': 'ঋণ',
                'employment': 'কর্মসংস্থান'
            }
        }
    
    def translate(self, text, target_language=None, source_language='en'):
        """Mock translation using basic word mapping"""
        if target_language == source_language or target_language == 'en':
            return {'translatedText': text, 'detectedSourceLanguage': source_language}
        
        # Simple word-by-word translation for demo
        if target_language in self.basic_translations:
            words = text.lower().split()
            translated_words = []
            
            for word in words:
                # Remove punctuation for lookup
                clean_word = re.sub(r'[^\w\s]', '', word)
                translated = self.basic_translations[target_language].get(clean_word, word)
                translated_words.append(translated)
            
            translated_text = ' '.join(translated_words)
            return {'translatedText': translated_text, 'detectedSourceLanguage': source_language}
        
        # If target language not supported, return original with note
        return {'translatedText': f"[{target_language}] {text}", 'detectedSourceLanguage': source_language}
    
    def detect_language(self, text):
        """Mock language detection"""
        # Simple detection based on script
        if re.search(r'[\u0900-\u097F]', text):  # Devanagari (Hindi)
            return {'language': 'hi', 'confidence': 0.8}
        elif re.search(r'[\u0B80-\u0BFF]', text):  # Tamil
            return {'language': 'ta', 'confidence': 0.8}
        elif re.search(r'[\u0C00-\u0C7F]', text):  # Telugu
            return {'language': 'te', 'confidence': 0.8}
        elif re.search(r'[\u0980-\u09FF]', text):  # Bengali
            return {'language': 'bn', 'confidence': 0.8}
        else:
            return {'language': 'en', 'confidence': 0.9}

class TranslationService:
    """Professional translation service with Google Cloud Translation and fallback"""
    
    def __init__(self):
        self.logger = logger
        self.use_google_translate = GOOGLE_TRANSLATE_AVAILABLE
        
        # Initialize Translation client
        if self.use_google_translate:
            try:
                self.client = translate.Client()
                logger.info("✅ Google Cloud Translation client initialized")
            except Exception as e:
                logger.warning(f"⚠️ Google Translation initialization failed: {e}, using mock client")
                self.use_google_translate = False
                self.client = MockTranslateClient()
        else:
            self.client = MockTranslateClient()
        
        # Supported languages with their names
        self.supported_languages = {
            'en': {'name': 'English', 'native': 'English'},
            'hi': {'name': 'Hindi', 'native': 'हिन्दी'},
            'ta': {'name': 'Tamil', 'native': 'தமிழ்'},
            'te': {'name': 'Telugu', 'native': 'తెలుగు'},
            'bn': {'name': 'Bengali', 'native': 'বাংলা'},
            'gu': {'name': 'Gujarati', 'native': 'ગુજરાતી'},
            'kn': {'name': 'Kannada', 'native': 'ಕನ್ನಡ'},
            'ml': {'name': 'Malayalam', 'native': 'മലയാളം'},
            'mr': {'name': 'Marathi', 'native': 'मराठी'},
            'or': {'name': 'Odia', 'native': 'ଓଡ଼ିଆ'},
            'pa': {'name': 'Punjabi', 'native': 'ਪੰਜਾਬੀ'},
            'ur': {'name': 'Urdu', 'native': 'اردو'},
            'es': {'name': 'Spanish', 'native': 'Español'},
            'fr': {'name': 'French', 'native': 'Français'},
            'de': {'name': 'German', 'native': 'Deutsch'},
            'zh': {'name': 'Chinese', 'native': '中文'},
            'ja': {'name': 'Japanese', 'native': '日本語'},
            'ko': {'name': 'Korean', 'native': '한국어'},
            'ar': {'name': 'Arabic', 'native': 'العربية'}
        }
        
        # Translation cache for efficiency
        self.translation_cache = {}
        self.cache_max_size = 1000
        
        self.logger.info(f"TranslationService initialized - Mode: {'Google Cloud' if self.use_google_translate else 'Mock'}")

    async def translate_document_summary(
        self, 
        content: Dict[str, Any], 
        target_language: str,
        source_language: str = 'en'
    ) -> Dict[str, Any]:
        """
        Translate document summary content
        
        Args:
            content: Dictionary containing summary, risks, recommendations
            target_language: Target language code
            source_language: Source language code
            
        Returns:
            Translated content dictionary
        """
        
        try:
            self.logger.info(f"Translating summary from {source_language} to {target_language}")
            
            if target_language == source_language:
                return content  # No translation needed
            
            # Validate languages
            if not self._is_language_supported(target_language):
                raise TranslationError(f"Unsupported target language: {target_language}")
            
            translated_content = {}
            
            # Translate main summary
            if 'summary' in content and content['summary']:
                translated_content['summary'] = await self._translate_text(
                    content['summary'], target_language, source_language
                )
            
            # Translate key risks
            if 'key_risks' in content and content['key_risks']:
                translated_content['key_risks'] = await self._translate_list(
                    content['key_risks'], target_language, source_language
                )
            
            # Translate recommendations
            if 'recommendations' in content and content['recommendations']:
                translated_content['recommendations'] = await self._translate_list(
                    content['recommendations'], target_language, source_language
                )
            
            # Translate user obligations
            if 'user_obligations' in content and content['user_obligations']:
                translated_content['user_obligations'] = await self._translate_list(
                    content['user_obligations'], target_language, source_language
                )
            
            # Translate user rights
            if 'user_rights' in content and content['user_rights']:
                translated_content['user_rights'] = await self._translate_list(
                    content['user_rights'], target_language, source_language
                )
            
            # Translate document type
            if 'document_type' in content:
                translated_content['document_type'] = await self._translate_document_type(
                    content['document_type'], target_language
                )
            
            # Copy non-translatable fields
            for key in ['overall_risk_score', 'fairness_score', 'page_count', 'word_count']:
                if key in content:
                    translated_content[key] = content[key]
            
            # Add metadata
            translated_content['translation_metadata'] = {
                'source_language': source_language,
                'target_language': target_language,
                'translated_at': datetime.now(timezone.utc).isoformat(),
                'service': 'google_translate' if self.use_google_translate else 'mock_translate',
                'version': '2.0'
            }
            
            self.logger.info(f"Translation completed successfully")
            return translated_content
            
        except Exception as e:
            self.logger.error(f"Error translating document summary: {str(e)}")
            raise TranslationError(f"Translation failed: {str(e)}")

    async def translate_chat_response(
        self, 
        response_text: str, 
        target_language: str,
        source_language: str = 'en'
    ) -> Dict[str, Any]:
        """
        Translate chat response to target language
        
        Args:
            response_text: Response text to translate
            target_language: Target language code
            source_language: Source language code
            
        Returns:
            Translation result with metadata
        """
        
        try:
            if target_language == source_language:
                return {
                    'translated_text': response_text,
                    'confidence': 1.0,
                    'detected_language': source_language,
                    'translation_time': 0.0
                }
            
            start_time = time.time()
            
            # Check cache first
            cache_key = f"{hash(response_text)}_{source_language}_{target_language}"
            if cache_key in self.translation_cache:
                cached_result = self.translation_cache[cache_key].copy()
                cached_result['from_cache'] = True
                return cached_result
            
            # Perform translation
            if self.use_google_translate:
                result = await asyncio.to_thread(
                    self.client.translate,
                    response_text,
                    target_language=target_language,
                    source_language=source_language
                )
            else:
                result = self.client.translate(
                    response_text,
                    target_language=target_language,
                    source_language=source_language
                )
            
            translation_time = time.time() - start_time
            
            translation_result = {
                'translated_text': result['translatedText'],
                'confidence': self._calculate_translation_confidence(result),
                'detected_language': result.get('detectedSourceLanguage', source_language),
                'source_text': response_text,
                'translation_time': translation_time,
                'from_cache': False
            }
            
            # Cache the result
            self._cache_translation(cache_key, translation_result)
            
            return translation_result
            
        except Exception as e:
            self.logger.error(f"Error translating chat response: {str(e)}")
            # Return original text if translation fails
            return {
                'translated_text': response_text,
                'confidence': 0.0,
                'detected_language': source_language,
                'error': str(e),
                'translation_time': 0.0,
                'from_cache': False
            }

    async def translate_legal_terms(
        self, 
        terms: List[str], 
        target_language: str,
        include_definitions: bool = True
    ) -> Dict[str, Dict[str, str]]:
        """
        Translate legal terms with definitions
        
        Args:
            terms: List of legal terms to translate
            target_language: Target language code
            include_definitions: Whether to include definitions
            
        Returns:
            Dictionary of translated terms with definitions
        """
        
        try:
            self.logger.info(f"Translating {len(terms)} legal terms to {target_language}")
            
            translated_terms = {}
            
            for term in terms:
                try:
                    # Translate the term
                    translated_term = await self._translate_text(
                        term, target_language, 'en'
                    )
                    
                    term_data = {
                        'translated': translated_term,
                        'original': term,
                        'language': target_language
                    }
                    
                    # Add definition if requested
                    if include_definitions:
                        definition = self._get_legal_term_definition(term)
                        if definition:
                            translated_definition = await self._translate_text(
                                definition, target_language, 'en'
                            )
                            term_data['definition'] = translated_definition
                            term_data['definition_original'] = definition
                    
                    translated_terms[term] = term_data
                    
                except Exception as e:
                    self.logger.warning(f"Failed to translate term '{term}': {e}")
                    # Add fallback data
                    translated_terms[term] = {
                        'translated': term,
                        'original': term,
                        'language': target_language,
                        'error': str(e)
                    }
            
            return translated_terms
            
        except Exception as e:
            self.logger.error(f"Error translating legal terms: {str(e)}")
            raise TranslationError(f"Legal terms translation failed: {str(e)}")

    async def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect language of given text
        
        Args:
            text: Text to analyze
            
        Returns:
            Language detection result
        """
        
        try:
            if self.use_google_translate:
                result = await asyncio.to_thread(
                    self.client.detect_language,
                    text
                )
            else:
                result = self.client.detect_language(text)
            
            return {
                'language': result['language'],
                'confidence': result['confidence'],
                'is_reliable': result['confidence'] > 0.8,
                'supported': self._is_language_supported(result['language']),
                'language_name': self.supported_languages.get(result['language'], {}).get('name', 'Unknown')
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting language: {str(e)}")
            return {
                'language': 'en',
                'confidence': 0.0,
                'is_reliable': False,
                'supported': True,
                'language_name': 'English',
                'error': str(e)
            }

    async def batch_translate(
        self, 
        texts: List[str], 
        target_language: str,
        source_language: str = 'en'
    ) -> List[Dict[str, Any]]:
        """
        Translate multiple texts in batch
        
        Args:
            texts: List of texts to translate
            target_language: Target language code
            source_language: Source language code
            
        Returns:
            List of translation results
        """
        
        try:
            self.logger.info(f"Batch translating {len(texts)} texts")
            
            if target_language == source_language:
                return [{'translated_text': text, 'confidence': 1.0, 'original_text': text} for text in texts]
            
            # Filter out empty texts
            non_empty_texts = [(i, text) for i, text in enumerate(texts) if text and text.strip()]
            
            if not non_empty_texts:
                return [{'translated_text': text, 'confidence': 0.0, 'original_text': text} for text in texts]
            
            # Extract just the texts for translation
            texts_to_translate = [text for _, text in non_empty_texts]
            
            # Perform batch translation
            if self.use_google_translate:
                results = await asyncio.to_thread(
                    self.client.translate,
                    texts_to_translate,
                    target_language=target_language,
                    source_language=source_language
                )
            else:
                # Mock client doesn't support batch, so translate individually
                results = []
                for text in texts_to_translate:
                    result = self.client.translate(
                        text,
                        target_language=target_language,
                        source_language=source_language
                    )
                    results.append(result)
            
            # Create full results array
            translated_results = [None] * len(texts)
            
            # Fill in translated results
            for j, (i, original_text) in enumerate(non_empty_texts):
                result = results[j]
                translated_results[i] = {
                    'original_text': original_text,
                    'translated_text': result['translatedText'],
                    'confidence': self._calculate_translation_confidence(result),
                    'detected_language': result.get('detectedSourceLanguage', source_language)
                }
            
            # Fill in empty results
            for i, result in enumerate(translated_results):
                if result is None:
                    translated_results[i] = {
                        'original_text': texts[i],
                        'translated_text': texts[i],
                        'confidence': 0.0,
                        'detected_language': source_language
                    }
            
            return translated_results
            
        except Exception as e:
            self.logger.error(f"Error in batch translation: {str(e)}")
            # Return original texts if translation fails
            return [
                {
                    'original_text': text,
                    'translated_text': text,
                    'confidence': 0.0,
                    'error': str(e),
                    'detected_language': source_language
                } for text in texts
            ]

    async def _translate_text(
        self, 
        text: str, 
        target_language: str, 
        source_language: str = 'en'
    ) -> str:
        """Private method to translate single text"""
        
        if not text or not text.strip():
            return text
        
        try:
            # Check cache first
            cache_key = f"{hash(text)}_{source_language}_{target_language}"
            if cache_key in self.translation_cache:
                return self.translation_cache[cache_key]['translated_text']
            
            if self.use_google_translate:
                result = await asyncio.to_thread(
                    self.client.translate,
                    text,
                    target_language=target_language,
                    source_language=source_language
                )
            else:
                result = self.client.translate(
                    text,
                    target_language=target_language,
                    source_language=source_language
                )
            
            translated_text = result['translatedText']
            
            # Cache the result
            cache_result = {
                'translated_text': translated_text,
                'confidence': self._calculate_translation_confidence(result)
            }
            self._cache_translation(cache_key, cache_result)
            
            return translated_text
            
        except Exception as e:
            self.logger.warning(f"Failed to translate text: {str(e)}")
            return text  # Return original if translation fails

    async def _translate_list(
        self, 
        items: List[str], 
        target_language: str, 
        source_language: str = 'en'
    ) -> List[str]:
        """Private method to translate list of strings"""
        
        try:
            if not items:
                return items
            
            # Use batch translation for efficiency
            if len(items) > 1:
                results = await self.batch_translate(items, target_language, source_language)
                return [result['translated_text'] for result in results]
            else:
                translated = await self._translate_text(items[0], target_language, source_language)
                return [translated]
                
        except Exception as e:
            self.logger.warning(f"Failed to translate list: {str(e)}")
            return items  # Return original if translation fails

    def _cache_translation(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache translation result"""
        try:
            # Simple cache with size limit
            if len(self.translation_cache) >= self.cache_max_size:
                # Remove oldest entries (FIFO)
                oldest_key = next(iter(self.translation_cache))
                del self.translation_cache[oldest_key]
            
            result['cached_at'] = time.time()
            self.translation_cache[cache_key] = result
            
        except Exception as e:
            self.logger.warning(f"Failed to cache translation: {e}")

    def _is_language_supported(self, language_code: str) -> bool:
        """Check if language is supported"""
        return language_code in self.supported_languages

    def _calculate_translation_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate confidence score for translation"""
        
        # Google Translate doesn't provide confidence scores directly
        # We estimate based on various factors
        confidence = 0.9 if self.use_google_translate else 0.7  # Base confidence
        
        # Check if source language was detected automatically
        if 'detectedSourceLanguage' in result:
            confidence -= 0.1  # Slightly less confident for auto-detected
        
        # Check translation length vs original (if available)
        if 'input' in result:
            original_len = len(result.get('input', ''))
            translated_len = len(result.get('translatedText', ''))
            
            if translated_len == 0:
                confidence = 0.0
            elif original_len > 0 and abs(translated_len - original_len) > original_len * 0.5:
                confidence -= 0.2  # Significant length difference
        
        # Reduce confidence for mock translations
        if not self.use_google_translate:
            confidence *= 0.6
        
        return max(0.0, min(1.0, confidence))

    async def _translate_document_type(self, doc_type: str, target_language: str) -> str:
        """Translate document type with context"""
        
        # Document type translations
        doc_type_translations = {
            'rental_agreement': {
                'hi': 'किराया समझौता',
                'ta': 'வாடகை ஒப்பந்தம்',
                'te': 'అద్దె ఒప్పందం',
                'bn': 'ভাড়া চুক্তি',
                'gu': 'ભાડા કરાર',
                'kn': 'ಬಾಡಿಗೆ ಒಪ್ಪಂದ',
                'ml': 'വാടക കരാർ',
                'mr': 'भाडे करार'
            },
            'loan_contract': {
                'hi': 'ऋण अनुबंध',
                'ta': 'கடன் ஒப்பந்தம்',
                'te': 'రుణ ఒప్పందం',
                'bn': 'ঋণ চুক্তি',
                'gu': 'લોન કરાર',
                'kn': 'ಸಾಲ ಒಪ್ಪಂದ',
                'ml': 'വായ്പാ കരാർ',
                'mr': 'कर्ज करार'
            },
            'employment_contract': {
                'hi': 'रोजगार अनुबंध',
                'ta': 'வேலைவாய்ப்பு ஒப்பந்தம்',
                'te': 'ఉపాధి ఒప్పందం',
                'bn': 'কর্মসংস্থান চুক্তি',
                'gu': 'રોજગાર કરાર',
                'kn': 'ಉದ್ಯೋಗ ಒಪ್ಪಂದ',
                'ml': 'തൊഴിൽ കരാർ',
                'mr': 'रोजगार करार'
            },
            'legal_document': {
                'hi': 'कानूनी दस्तावेज़',
                'ta': 'சட்ட ஆவணம்',
                'te': 'చట్టపరమైన పత్రం',
                'bn': 'আইনি নথি',
                'gu': 'કાનૂની દસ્તાવેજ',
                'kn': 'ಕಾನೂನು ದಾಖಲೆ',
                'ml': 'നിയമ രേഖ',
                'mr': 'कायदेशीर कागदपत्र'
            }
        }
        
        # Use predefined translation if available
        if doc_type in doc_type_translations and target_language in doc_type_translations[doc_type]:
            return doc_type_translations[doc_type][target_language]
        
        # Fallback to general translation
        formatted_type = doc_type.replace('_', ' ').title()
        return await self._translate_text(formatted_type, target_language)

    def _get_legal_term_definition(self, term: str) -> Optional[str]:
        """Get definition for legal term"""
        
        legal_definitions = {
            'liability': 'Legal responsibility for damages or debts incurred by another party',
            'indemnify': 'To compensate someone for harm or loss suffered',
            'breach': 'Failure to fulfill a contractual obligation or duty',
            'default': 'Failure to meet payment obligations or other contractual requirements',
            'termination': 'The end or cancellation of a contract or agreement',
            'penalty': 'A punishment or fine imposed for breaking rules or contractual terms',
            'warranty': 'A promise or guarantee about the quality or condition of a product or service',
            'jurisdiction': 'The legal authority of a court or government agency over a particular area',
            'arbitration': 'A method of settling disputes outside of court through a neutral third party',
            'force majeure': 'Unforeseeable circumstances that prevent a party from fulfilling a contract',
            'consideration': 'Something of value exchanged between parties to make a contract valid',
            'covenant': 'A formal promise or agreement in a contract',
            'lien': 'A legal claim on property as security for a debt',
            'novation': 'The replacement of an existing contract with a new contract',
            'rescission': 'The cancellation of a contract, returning parties to their original position'
        }
        
        return legal_definitions.get(term.lower())

    def get_supported_languages_info(self) -> Dict[str, Dict[str, str]]:
        """Get information about supported languages"""
        return self.supported_languages

    async def validate_translation_quality(
        self, 
        original_text: str, 
        translated_text: str,
        target_language: str
    ) -> Dict[str, Any]:
        """
        Validate translation quality by back-translation
        
        Args:
            original_text: Original text
            translated_text: Translated text
            target_language: Target language used
            
        Returns:
            Quality assessment
        """
        
        try:
            # Translate back to English
            back_translated = await self._translate_text(
                translated_text, 'en', target_language
            )
            
            # Calculate similarity metrics
            similarity = self._calculate_text_similarity(original_text, back_translated)
            length_ratio = len(translated_text) / max(len(original_text), 1)
            
            # Assess quality based on multiple factors
            quality_score = (similarity * 0.7) + (min(length_ratio, 2.0) / 2.0 * 0.3)
            
            return {
                'quality_score': round(quality_score, 3),
                'back_translation': back_translated,
                'similarity_score': round(similarity, 3),
                'length_ratio': round(length_ratio, 3),
                'is_high_quality': quality_score > 0.7,
                'assessment': 'high' if quality_score > 0.8 else 'medium' if quality_score > 0.6 else 'low',
                'service_used': 'google_translate' if self.use_google_translate else 'mock_translate'
            }
            
        except Exception as e:
            self.logger.error(f"Error validating translation quality: {str(e)}")
            return {
                'quality_score': 0.5,
                'back_translation': '',
                'similarity_score': 0.0,
                'length_ratio': 1.0,
                'is_high_quality': False,
                'assessment': 'unknown',
                'error': str(e)
            }

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using multiple metrics"""
        
        if not text1 or not text2:
            return 0.0
        
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        # Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Length similarity
        len1, len2 = len(text1), len(text2)
        length_sim = 1 - abs(len1 - len2) / max(len1, len2, 1)
        
        # Combined similarity
        return (jaccard * 0.8) + (length_sim * 0.2)

    async def create_multilingual_glossary(
        self, 
        legal_terms: List[str]
    ) -> Dict[str, Dict[str, str]]:
        """
        Create multilingual glossary for legal terms
        
        Args:
            legal_terms: List of legal terms
            
        Returns:
            Multilingual glossary
        """
        
        try:
            self.logger.info(f"Creating multilingual glossary for {len(legal_terms)} terms")
            
            glossary = {}
            
            # Priority languages for legal terms
            priority_languages = ['hi', 'ta', 'te', 'bn', 'gu', 'kn', 'ml', 'mr']
            
            for term in legal_terms:
                try:
                    term_translations = {'en': term}
                    
                    # Translate to priority languages
                    for lang_code in priority_languages:
                        if lang_code in self.supported_languages:
                            try:
                                translated = await self._translate_text(term, lang_code, 'en')
                                term_translations[lang_code] = translated
                            except Exception as e:
                                self.logger.warning(f"Failed to translate {term} to {lang_code}: {str(e)}")
                                term_translations[lang_code] = term
                    
                    # Add definition if available
                    definition = self._get_legal_term_definition(term)
                    if definition:
                        definition_translations = {'en': definition}
                        
                        for lang_code in priority_languages[:4]:  # Limit definition translations
                            try:
                                translated_def = await self._translate_text(definition, lang_code, 'en')
                                definition_translations[lang_code] = translated_def
                            except Exception as e:
                                self.logger.warning(f"Failed to translate definition for {term}: {e}")
                        
                        term_translations['definitions'] = definition_translations
                    
                    glossary[term] = term_translations
                    
                except Exception as e:
                    self.logger.error(f"Error processing term '{term}': {e}")
                    glossary[term] = {'en': term, 'error': str(e)}
            
            return glossary
            
        except Exception as e:
            self.logger.error(f"Error creating multilingual glossary: {str(e)}")
            return {}

    def get_translation_stats(self) -> Dict[str, Any]:
        """Get translation service statistics"""
        
        return {
            'service_type': 'Google Cloud Translation' if self.use_google_translate else 'Mock Translation',
            'supported_languages_count': len(self.supported_languages),
            'cache_size': len(self.translation_cache),
            'cache_max_size': self.cache_max_size,
            'capabilities': {
                'batch_translation': True,
                'language_detection': True,
                'quality_validation': True,
                'caching': True,
                'legal_terms': True,
                'document_types': True,
                'back_translation': True
            },
            'supported_language_codes': list(self.supported_languages.keys()),
            'priority_languages': ['hi', 'ta', 'te', 'bn', 'gu', 'kn', 'ml', 'mr'],
            'dependencies': {
                'google_cloud_translate': GOOGLE_TRANSLATE_AVAILABLE,
                'settings': SETTINGS_AVAILABLE
            }
        }

    def clear_cache(self) -> int:
        """Clear translation cache"""
        cache_size = len(self.translation_cache)
        self.translation_cache.clear()
        self.logger.info(f"Cleared translation cache ({cache_size} entries)")
        return cache_size


# Utility functions
def create_simple_translation_service() -> TranslationService:
    """Create a simple translation service for testing"""
    return TranslationService()

def is_supported_language(language_code: str) -> bool:
    """Check if language is supported"""
    service = TranslationService()
    return service._is_language_supported(language_code)

def get_language_name(language_code: str) -> str:
    """Get language name from code"""
    service = TranslationService()
    lang_info = service.supported_languages.get(language_code, {})
    return lang_info.get('name', 'Unknown')

def detect_script_language(text: str) -> Optional[str]:
    """Simple script-based language detection"""
    
    if re.search(r'[\u0900-\u097F]', text):  # Devanagari
        return 'hi'
    elif re.search(r'[\u0B80-\u0BFF]', text):  # Tamil
        return 'ta'
    elif re.search(r'[\u0C00-\u0C7F]', text):  # Telugu
        return 'te'
    elif re.search(r'[\u0980-\u09FF]', text):  # Bengali
        return 'bn'
    elif re.search(r'[\u0A80-\u0AFF]', text):  # Gujarati
        return 'gu'
    elif re.search(r'[\u0C80-\u0CFF]', text):  # Kannada
        return 'kn'
    elif re.search(r'[\u0D00-\u0D7F]', text):  # Malayalam
        return 'ml'
    elif re.search(r'[\u0A00-\u0A7F]', text):  # Punjabi
        return 'pa'
    elif re.search(r'[\u0600-\u06FF]', text):  # Arabic
        return 'ar'
    elif re.search(r'[\u4E00-\u9FFF]', text):  # Chinese
        return 'zh'
    else:
        return 'en'  # Default to English

async def translate_simple_text(text: str, target_language: str, source_language: str = 'en') -> str:
    """Simple text translation utility"""
    
    service = TranslationService()
    try:
        return await service._translate_text(text, target_language, source_language)
    except Exception as e:
        logger.error(f"Simple translation failed: {e}")
        return text
