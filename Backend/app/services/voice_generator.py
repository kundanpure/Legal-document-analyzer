from typing import List, Dict, Any, Optional
"""
Google Cloud Text-to-Speech integration for voice summary generation
"""

from google.cloud import texttospeech
import asyncio
import aiofiles

import os
from dotenv import load_dotenv
load_dotenv() 
from typing import Dict, Any, Optional
import uuid
from datetime import datetime, timedelta

from config.settings import get_settings
from config.logging import get_logger
from app.core.exceptions import VoiceGenerationError
from app.services.storage_manager import StorageManager

logger = get_logger(__name__)
settings = get_settings()

class VoiceGenerator:
    """Professional Google Cloud Text-to-Speech integration"""
    
    def __init__(self):
        self.settings = settings
        self.logger = logger
        self.storage_manager = StorageManager()
        
        # Initialize TTS client
        self.client = texttospeech.TextToSpeechClient()
        
        # Voice configuration mapping
        self.voice_configs = settings.VOICE_MODELS
        
        self.logger.info("VoiceGenerator initialized successfully")

    async def generate_voice_summary(
        self, 
        text_content: str, 
        language: str = "en",
        voice_type: str = "female",
        speed: float = 1.0
    ) -> Dict[str, Any]:
        """
        Generate voice summary from text content
        
        Args:
            text_content: Text to convert to speech
            language: Target language code
            voice_type: Voice type (male/female)
            speed: Speech speed (0.5 - 2.0)
            
        Returns:
            Voice generation result with audio URL
        """
        
        try:
            self.logger.info(f"Generating voice summary in {language}")
            
            # Validate inputs
            if not text_content or len(text_content.strip()) == 0:
                raise VoiceGenerationError("Text content cannot be empty")
            
            # Prepare text for speech synthesis
            processed_text = self._prepare_text_for_speech(text_content, language)
            
            # Get voice configuration
            voice_config = self._get_voice_configuration(language, voice_type)
            
            # Configure audio output
            audio_config = self._get_audio_configuration(speed)
            
            # Generate speech
            audio_data = await self._synthesize_speech(
                processed_text, voice_config, audio_config
            )
            
            # Generate unique filename
            voice_id = str(uuid.uuid4())
            filename = f"voice_summary_{voice_id}.mp3"
            
            # Save audio to storage
            audio_url = await self.storage_manager.save_audio_file(
                audio_data, filename
            )
            
            # Calculate duration (approximate)
            duration = self._estimate_audio_duration(processed_text, speed)
            
            # Create transcript
            transcript = self._create_transcript(processed_text, language)
            
            result = {
                'voice_id': voice_id,
                'audio_url': audio_url,
                'duration': duration,
                'language': language,
                'voice_type': voice_type,
                'transcript': transcript,
                'file_size': len(audio_data),
                'generated_at': datetime.utcnow().isoformat(),
                'expires_at': (datetime.utcnow() + timedelta(days=7)).isoformat()
            }
            
            self.logger.info(f"Voice summary generated successfully: {voice_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating voice summary: {str(e)}")
            raise VoiceGenerationError(f"Failed to generate voice: {str(e)}")

    async def generate_multilingual_summary(
        self, 
        text_content: str, 
        languages: list,
        voice_type: str = "female"
    ) -> Dict[str, Any]:
        """
        Generate voice summaries in multiple languages
        
        Args:
            text_content: Text content to convert
            languages: List of language codes
            voice_type: Voice type preference
            
        Returns:
            Dictionary of voice summaries by language
        """
        
        try:
            self.logger.info(f"Generating multilingual summaries for {len(languages)} languages")
            
            # Generate summaries concurrently
            tasks = []
            for lang in languages:
                task = self.generate_voice_summary(
                    text_content, lang, voice_type
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            multilingual_results = {}
            for i, result in enumerate(results):
                lang = languages[i]
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to generate voice for {lang}: {str(result)}")
                    multilingual_results[lang] = {'error': str(result)}
                else:
                    multilingual_results[lang] = result
            
            return {
                'multilingual_voices': multilingual_results,
                'generated_count': len([r for r in results if not isinstance(r, Exception)]),
                'total_requested': len(languages),
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in multilingual generation: {str(e)}")
            raise VoiceGenerationError(f"Multilingual generation failed: {str(e)}")

    async def _synthesize_speech(
        self, 
        text: str, 
        voice_config: Dict[str, Any],
        audio_config: texttospeech.AudioConfig
    ) -> bytes:
        """Synthesize speech using Google Cloud TTS"""
        
        try:
            # Prepare synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Configure voice
            voice = texttospeech.VoiceSelectionParams(
                language_code=voice_config['language_code'],
                name=voice_config['name'],
                ssml_gender=voice_config.get('gender', texttospeech.SsmlVoiceGender.FEMALE)
            )
            
            # Perform synthesis
            response = await asyncio.to_thread(
                self.client.synthesize_speech,
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            return response.audio_content
            
        except Exception as e:
            raise VoiceGenerationError(f"Speech synthesis failed: {str(e)}")

    def _prepare_text_for_speech(self, text: str, language: str) -> str:
        """Prepare text for optimal speech synthesis"""
        
        # Remove excessive whitespace
        cleaned_text = " ".join(text.split())
        
        # Limit text length (TTS has character limits)
        max_chars = 5000
        if len(cleaned_text) > max_chars:
            # Find a good breaking point
            truncated = cleaned_text[:max_chars]
            last_sentence = truncated.rfind('.')
            if last_sentence > max_chars * 0.8:  # If we find a sentence end close to limit
                cleaned_text = truncated[:last_sentence + 1]
            else:
                cleaned_text = truncated + "..."
        
        # Add pauses for better readability
        cleaned_text = cleaned_text.replace('. ', '. <break time="0.5s"/> ')
        cleaned_text = cleaned_text.replace('? ', '? <break time="0.5s"/> ')
        cleaned_text = cleaned_text.replace('! ', '! <break time="0.5s"/> ')
        
        # Add emphasis for important terms
        important_terms = ['risk', 'important', 'warning', 'notice', 'penalty', 'fee']
        for term in important_terms:
            cleaned_text = cleaned_text.replace(
                term, f'<emphasis level="moderate">{term}</emphasis>'
            )
        
        return f'<speak>{cleaned_text}</speak>'

    def _get_voice_configuration(self, language: str, voice_type: str) -> Dict[str, Any]:
        """Get voice configuration for language and type"""
        
        # Get base configuration
        if language in self.voice_configs:
            config = self.voice_configs[language].copy()
        else:
            # Fallback to English
            config = self.voice_configs['en'].copy()
        
        # Adjust for voice type
        if voice_type == "male":
            config['gender'] = texttospeech.SsmlVoiceGender.MALE
            config['name'] = config['name'].replace('-A', '-B')  # Male variant
        else:
            config['gender'] = texttospeech.SsmlVoiceGender.FEMALE
        
        return config

    def _get_audio_configuration(self, speed: float) -> texttospeech.AudioConfig:
        """Get audio output configuration"""
        
        return texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speed,
            pitch=0.0,  # Neutral pitch
            volume_gain_db=0.0,  # Neutral volume
            sample_rate_hertz=24000,  # High quality
            effects_profile_id=["headphone-class-device"]  # Optimize for headphones
        )

    def _estimate_audio_duration(self, text: str, speed: float) -> str:
        """Estimate audio duration based on text length and speed"""
        
        # Average speaking rate: 150-160 words per minute
        words = len(text.split())
        base_minutes = words / 150  # Base rate
        adjusted_minutes = base_minutes / speed  # Adjust for speed
        
        total_seconds = int(adjusted_minutes * 60)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        
        return f"{minutes:02d}:{seconds:02d}"

    def _create_transcript(self, processed_text: str, language: str) -> str:
        """Create clean transcript from processed text"""
        
        # Remove SSML tags
        import re
        clean_transcript = re.sub(r'<[^>]+>', '', processed_text)
        
        # Clean up extra whitespace
        clean_transcript = " ".join(clean_transcript.split())
        
        return clean_transcript

    async def get_available_voices(self, language: str = None) -> List[Dict[str, Any]]:
        """Get list of available voices for language"""
        
        try:
            # Get all available voices
            voices_response = await asyncio.to_thread(
                self.client.list_voices
            )
            
            available_voices = []
            for voice in voices_response.voices:
                for lang_code in voice.language_codes:
                    if language is None or lang_code.startswith(language):
                        voice_info = {
                            'name': voice.name,
                            'language_code': lang_code,
                            'gender': voice.ssml_gender.name,
                            'natural_sample_rate_hertz': voice.natural_sample_rate_hertz
                        }
                        available_voices.append(voice_info)
            
            return available_voices
            
        except Exception as e:
            self.logger.error(f"Error getting available voices: {str(e)}")
            return []

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages for voice generation"""
        return list(self.voice_configs.keys())

    async def generate_pronunciation_guide(
        self, 
        difficult_terms: List[str], 
        language: str = "en"
    ) -> Dict[str, str]:
        """Generate pronunciation guide for difficult legal terms"""
        
        pronunciation_guide = {}
        
        for term in difficult_terms:
            # Create phonetic representation using SSML
            phonetic = f'<phoneme alphabet="ipa" ph="{term}">{term}</phoneme>'
            pronunciation_guide[term] = phonetic
        
        return pronunciation_guide

    async def cleanup_old_audio_files(self, days_old: int = 7) -> int:
        """Clean up audio files older than specified days"""
        
        try:
            # This would interact with storage manager to clean old files
            deleted_count = await self.storage_manager.cleanup_audio_files(days_old)
            
            self.logger.info(f"Cleaned up {deleted_count} old audio files")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up audio files: {str(e)}")
            return 0
