"""
Chat handler service for managing interactive conversations with documents
"""

import asyncio
from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime, timedelta, timezone
import json

from config.settings import get_settings
from config.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

from dotenv import load_dotenv
load_dotenv() 

# Custom exceptions
class ChatError(Exception):
    """Exception raised for chat-related errors"""
    pass

class ChatHandler:
    """Professional chat handler for document-based conversations"""
    
    def __init__(self):
        self.settings = settings
        self.logger = logger
        
        # Initialize services with fallbacks
        self.gemini_analyzer = self._initialize_gemini_analyzer()
        self.translation_service = self._initialize_translation_service()
        
        # Chat session storage (use Redis in production)
        self.active_sessions = {}
        self.conversation_history = {}
        
        # Context management
        self.max_history_length = 10
        self.session_timeout = 3600  # 1 hour in seconds
        
        self.logger.info("ChatHandler initialized successfully")

    def _initialize_gemini_analyzer(self):
        """Initialize Gemini analyzer with fallback"""
        try:
            from app.services.gemini_analyzer import GeminiAnalyzer
            return GeminiAnalyzer()
        except ImportError:
            self.logger.warning("GeminiAnalyzer not available, using fallback")
            return self._create_fallback_gemini()

    def _initialize_translation_service(self):
        """Initialize translation service with fallback"""
        try:
            from app.services.translation_service import TranslationService
            return TranslationService()
        except ImportError:
            self.logger.warning("TranslationService not available, using fallback")
            return self._create_fallback_translator()

    def _create_fallback_gemini(self):
        """Create fallback gemini analyzer"""
        class FallbackGemini:
            async def get_chat_response(self, document_context=None, question="", language="en", conversation_history=None):
                return {
                    'response': f"I understand your question: '{question}'. This is a simulated response while the Gemini service is being configured.",
                    'confidence': 0.7,
                    'citations': []
                }
            
            async def generate_suggested_questions(self, doc_type, summary, risk_level):
                return [
                    "What are the key terms I should understand?",
                    "Are there any important deadlines?",
                    "What are my main obligations?"
                ]
        
        return FallbackGemini()

    def _create_fallback_translator(self):
        """Create fallback translation service"""
        class FallbackTranslator:
            async def translate_chat_response(self, text, target_lang, source_lang):
                return {
                    'translated_text': text,  # Return original text
                    'confidence': 0.5
                }
            
            async def batch_translate(self, texts, target_lang, source_lang):
                return [{'translated_text': text} for text in texts]
        
        return FallbackTranslator()

    async def get_ai_response(
        self,
        document_id: str,
        message: str,
        session_id: str,
        language: str = "en",
        document_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get AI response for user message
        
        Args:
            document_id: Document identifier
            message: User's message
            session_id: Session identifier
            language: Response language
            document_context: Document analysis context
            
        Returns:
            AI response with metadata
        """
        
        try:
            self.logger.info(f"Processing chat message for session: {session_id}")
            
            # Validate session
            if not self._is_session_valid(session_id):
                raise ChatError("Invalid or expired session")
            
            # Update session activity
            self._update_session_activity(session_id)
            
            # Get conversation history
            history = self._get_conversation_history(session_id)
            
            # Prepare context for AI
            context = self._prepare_ai_context(
                document_context, history, document_id
            )
            
            # Get AI response using Gemini
            ai_response = await self.gemini_analyzer.get_chat_response(
                document_context=context,
                question=message,
                language=language,
                conversation_history=history
            )
            
            # Process and enhance response
            enhanced_response = await self._enhance_response(
                ai_response, message, document_context, language
            )
            
            # Store conversation
            await self._store_conversation_turn(
                session_id, message, enhanced_response, language
            )
            
            # Generate follow-up suggestions
            suggestions = await self._generate_follow_up_suggestions(
                message, enhanced_response, document_context, language
            )
            
            result = {
                'message_id': str(uuid.uuid4()),
                'response': enhanced_response['response'],
                'citations': enhanced_response.get('citations', []),
                'confidence': enhanced_response.get('confidence', 0.8),
                'suggestions': suggestions,
                'language': language,
                'response_time': enhanced_response.get('response_time', 0),
                'context_used': len(history) > 0,
                'warnings': enhanced_response.get('warnings', [])
            }
            
            self.logger.info(f"Chat response generated successfully for session: {session_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating chat response: {str(e)}")
            raise ChatError(f"Failed to generate response: {str(e)}")

    async def initialize_chat_session(
        self,
        document_id: str,
        document_data: Dict[str, Any],
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Initialize a new chat session with welcome message
        
        Args:
            document_id: Document identifier
            document_data: Document analysis data
            language: Session language
            
        Returns:
            Session initialization result
        """
        
        try:
            session_id = str(uuid.uuid4())
            
            # Create session
            session_data = {
                'session_id': session_id,
                'document_id': document_id,
                'language': language,
                'created_at': datetime.now(timezone.utc),  # Fixed deprecated datetime.utcnow()
                'last_activity': datetime.now(timezone.utc),  # Fixed deprecated datetime.utcnow()
                'message_count': 0,
                'user_preferences': {},
                'context_summary': self._create_context_summary(document_data)
            }
            
            self.active_sessions[session_id] = session_data
            self.conversation_history[session_id] = []
            
            # Generate welcome message
            welcome_message = await self._generate_welcome_message(
                document_data, language
            )
            
            # Store welcome message
            await self._store_conversation_turn(
                session_id,
                None,  # No user message for welcome
                {'response': welcome_message, 'confidence': 1.0},
                language,
                message_type='system'
            )
            
            # Generate initial suggestions
            initial_suggestions = await self._generate_initial_suggestions(
                document_data, language
            )
            
            result = {
                'session_id': session_id,
                'welcome_message': welcome_message,
                'suggestions': initial_suggestions,
                'document_summary': {
                    'title': document_data.get('title', 'Document'),
                    'type': document_data.get('document_type', 'unknown'),
                    'risk_score': document_data.get('overall_risk_score', 0),
                    'key_risks_count': len(document_data.get('key_risks', []))
                },
                'session_expires_at': (datetime.now(timezone.utc) + timedelta(seconds=self.session_timeout)).isoformat()  # Fixed deprecated datetime.utcnow()
            }
            
            self.logger.info(f"Chat session initialized: {session_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error initializing chat session: {str(e)}")
            raise ChatError(f"Failed to initialize session: {str(e)}")

    async def get_session_history(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get conversation history for a session
        
        Args:
            session_id: Session identifier
            limit: Maximum messages to return
            offset: Offset for pagination
            
        Returns:
            Conversation history
        """
        
        try:
            if session_id not in self.conversation_history:
                return {
                    'session_id': session_id,
                    'messages': [],
                    'total_count': 0,
                    'has_more': False
                }
            
            history = self.conversation_history[session_id]
            
            # Apply pagination
            start_idx = offset
            end_idx = min(offset + limit, len(history))
            paginated_history = history[start_idx:end_idx]
            
            return {
                'session_id': session_id,
                'messages': paginated_history,
                'total_count': len(history),
                'has_more': end_idx < len(history),
                'session_info': self.active_sessions.get(session_id, {})
            }
            
        except Exception as e:
            self.logger.error(f"Error getting session history: {str(e)}")
            return {
                'session_id': session_id,
                'messages': [],
                'total_count': 0,
                'has_more': False,
                'error': str(e)
            }

    async def end_chat_session(self, session_id: str) -> Dict[str, Any]:
        """
        End a chat session and cleanup resources
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session summary
        """
        
        try:
            if session_id not in self.active_sessions:
                raise ChatError("Session not found")
            
            session_data = self.active_sessions[session_id]
            history = self.conversation_history.get(session_id, [])
            
            # Generate session summary
            summary = {
                'session_id': session_id,
                'document_id': session_data.get('document_id'),
                'duration': (datetime.now(timezone.utc) - session_data['created_at']).total_seconds(),  # Fixed deprecated datetime.utcnow()
                'message_count': len(history),
                'language': session_data.get('language', 'en'),
                'ended_at': datetime.now(timezone.utc).isoformat()  # Fixed deprecated datetime.utcnow()
            }
            
            # Clean up session data
            del self.active_sessions[session_id]
            
            # Archive conversation history (keep for longer term)
            # In production, move to persistent storage
            
            self.logger.info(f"Chat session ended: {session_id}")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error ending chat session: {str(e)}")
            raise ChatError(f"Failed to end session: {str(e)}")

    async def update_session_preferences(
        self,
        session_id: str,
        preferences: Dict[str, Any]
    ) -> bool:
        """
        Update user preferences for a session
        
        Args:
            session_id: Session identifier
            preferences: User preferences
            
        Returns:
            Success status
        """
        
        try:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['user_preferences'].update(preferences)
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error updating session preferences: {str(e)}")
            return False

    def _is_session_valid(self, session_id: str) -> bool:
        """Check if session is valid and not expired"""
        
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        last_activity = session.get('last_activity', datetime.now(timezone.utc))  # Fixed deprecated datetime.utcnow()
        
        # Check if session has expired
        if isinstance(last_activity, str):
            last_activity = datetime.fromisoformat(last_activity.replace('Z', '+00:00'))
        
        elapsed = (datetime.now(timezone.utc) - last_activity).total_seconds()  # Fixed deprecated datetime.utcnow()
        return elapsed < self.session_timeout

    def _update_session_activity(self, session_id: str):
        """Update session last activity timestamp"""
        
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['last_activity'] = datetime.now(timezone.utc)  # Fixed deprecated datetime.utcnow()
            self.active_sessions[session_id]['message_count'] = \
                self.active_sessions[session_id].get('message_count', 0) + 1

    def _get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get recent conversation history for context"""
        
        if session_id not in self.conversation_history:
            return []
        
        history = self.conversation_history[session_id]
        
        # Return last N messages for context (excluding system messages)
        user_messages = [msg for msg in history if msg.get('type') != 'system']
        return user_messages[-self.max_history_length:]

    def _prepare_ai_context(
        self,
        document_context: Optional[Dict[str, Any]],
        history: List[Dict[str, Any]],
        document_id: str
    ) -> Dict[str, Any]:
        """Prepare context for AI response generation"""
        
        context = {}
        
        if document_context:
            context.update({
                'document_summary': document_context.get('summary', ''),
                'key_risks': document_context.get('key_risks', []),
                'document_type': document_context.get('document_type', 'unknown'),
                'risk_score': document_context.get('overall_risk_score', 0),
                'recommendations': document_context.get('recommendations', [])
            })
        
        if history:
            context['recent_conversation'] = [
                {
                    'role': msg.get('type', 'user'),
                    'content': msg.get('content', msg.get('message', ''))
                }
                for msg in history[-5:]  # Last 5 exchanges
            ]
        
        context['document_id'] = document_id
        
        return context

    async def _enhance_response(
        self,
        ai_response: Dict[str, Any],
        user_message: str,
        document_context: Optional[Dict[str, Any]],
        language: str
    ) -> Dict[str, Any]:
        """Enhance AI response with additional processing"""
        
        enhanced = ai_response.copy()
        
        # Add response timing
        enhanced['response_time'] = 0.8  # Simulated response time
        
        # Check for sensitive topics and add warnings
        sensitive_keywords = ['penalty', 'termination', 'liability', 'breach', 'default']
        if any(keyword in user_message.lower() for keyword in sensitive_keywords):
            if 'warnings' not in enhanced:
                enhanced['warnings'] = []
            enhanced['warnings'].append(
                "This topic involves important legal implications. Consider consulting with a legal professional."
            )
        
        # Translate response if needed
        if language != 'en':
            try:
                translation_result = await self.translation_service.translate_chat_response(
                    enhanced['response'], language, 'en'
                )
                enhanced['response'] = translation_result['translated_text']
                enhanced['translation_confidence'] = translation_result.get('confidence', 0.8)
            except Exception as e:
                self.logger.warning(f"Failed to translate response: {str(e)}")
        
        return enhanced

    async def _store_conversation_turn(
        self,
        session_id: str,
        user_message: Optional[str],
        ai_response: Dict[str, Any],
        language: str,
        message_type: str = 'conversation'
    ):
        """Store a conversation turn in history"""
        
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        
        timestamp = datetime.now(timezone.utc).isoformat()  # Fixed deprecated datetime.utcnow()
        
        # Store user message
        if user_message and message_type == 'conversation':
            user_entry = {
                'message_id': str(uuid.uuid4()),
                'type': 'user',
                'content': user_message,
                'timestamp': timestamp,
                'language': language
            }
            self.conversation_history[session_id].append(user_entry)
        
        # Store AI response
        ai_entry = {
            'message_id': str(uuid.uuid4()),
            'type': 'assistant' if message_type == 'conversation' else 'system',
            'content': ai_response['response'],
            'timestamp': timestamp,
            'language': language,
            'confidence': ai_response.get('confidence', 0.8),
            'citations': ai_response.get('citations', []),
            'warnings': ai_response.get('warnings', [])
        }
        
        self.conversation_history[session_id].append(ai_entry)
        
        # Trim history if too long
        if len(self.conversation_history[session_id]) > 100:
            self.conversation_history[session_id] = self.conversation_history[session_id][-80:]

    async def _generate_welcome_message(
        self,
        document_data: Dict[str, Any],
        language: str
    ) -> str:
        """Generate personalized welcome message"""
        
        doc_type = document_data.get('document_type', 'document')
        risk_score = document_data.get('overall_risk_score', 0)
        key_risks_count = len(document_data.get('key_risks', []))
        
        if language == 'hi':
            if risk_score > 7:
                welcome = f"नमस्ते! मैंने आपके {doc_type} का विश्लेषण किया है। मुझे {key_risks_count} महत्वपूर्ण जोखिम मिले हैं। आप मुझसे कोई भी प्रश्न पूछ सकते हैं।"
            else:
                welcome = f"नमस्ते! मैंने आपके {doc_type} का विश्लेषण पूरा कर लिया है। इस दस्तावेज़ के बारे में आप क्या जानना चाहते हैं?"
        else:
            if risk_score > 7:
                welcome = f"Hello! I've analyzed your {doc_type.replace('_', ' ')} and identified {key_risks_count} key risks that need your attention. What would you like to know?"
            elif risk_score > 4:
                welcome = f"Hi! I've completed analyzing your {doc_type.replace('_', ' ')}. There are some areas that warrant attention. How can I help you understand this document?"
            else:
                welcome = f"Hello! I've reviewed your {doc_type.replace('_', ' ')} and it looks relatively straightforward. What questions do you have about it?"
        
        return welcome

    async def _generate_initial_suggestions(
        self,
        document_data: Dict[str, Any],
        language: str
    ) -> List[str]:
        """Generate initial conversation suggestions"""
        
        doc_type = document_data.get('document_type', 'document')
        
        if language == 'hi':
            base_suggestions = [
                "इस दस्तावेज़ में मुख्य जोखिम क्या हैं?",
                "मेरी मुख्य जिम्मेदारियां क्या हैं?",
                "क्या कोई छुपी हुई फीस है?",
                "मैं इस समझौते को कैसे समाप्त कर सकता हूं?"
            ]
        else:
            base_suggestions = [
                "What are the main risks in this document?",
                "What are my key obligations?",
                "Are there any hidden fees or costs?",
                "How can I terminate this agreement?",
                "What should I negotiate or change?"
            ]
        
        # Add document-type specific suggestions
        type_specific = await self.gemini_analyzer.generate_suggested_questions(
            doc_type, document_data.get('summary', ''), 'medium'
        )
        
        # Combine and limit to 6 suggestions
        all_suggestions = base_suggestions + type_specific
        return list(set(all_suggestions))[:6]

    async def _generate_follow_up_suggestions(
        self,
        user_message: str,
        ai_response: Dict[str, Any],
        document_context: Optional[Dict[str, Any]],
        language: str
    ) -> List[str]:
        """Generate follow-up question suggestions"""
        
        suggestions = []
        
        # Analyze user message for context
        message_lower = user_message.lower()
        
        if 'risk' in message_lower:
            suggestions.extend([
                "How can I reduce these risks?",
                "Which risks are most serious?",
                "What happens if these risks materialize?"
            ])
        elif 'terminate' in message_lower or 'cancel' in message_lower:
            suggestions.extend([
                "What are the termination penalties?",
                "How much notice is required?",
                "Can I terminate early?"
            ])
        elif 'fee' in message_lower or 'cost' in message_lower:
            suggestions.extend([
                "Are there any other costs?",
                "When are payments due?",
                "What happens if I miss a payment?"
            ])
        elif 'obligation' in message_lower or 'responsibility' in message_lower:
            suggestions.extend([
                "What happens if I don't meet my obligations?",
                "Are there any penalties for non-compliance?",
                "Can these obligations be modified?"
            ])
        
        # Add general follow-ups
        general_followups = [
            "Can you explain this in simpler terms?",
            "What should I be most concerned about?",
            "Is this clause standard or unusual?"
        ]
        
        suggestions.extend(general_followups)
        
        # Translate to target language if needed
        if language != 'en':
            try:
                translated_suggestions = await self.translation_service.batch_translate(
                    suggestions[:5], language, 'en'
                )
                suggestions = [result['translated_text'] for result in translated_suggestions]
            except Exception as e:
                self.logger.warning(f"Failed to translate suggestions: {str(e)}")
        
        return suggestions[:4]  # Limit to 4 suggestions

    def _create_context_summary(self, document_data: Dict[str, Any]) -> str:
        """Create a brief context summary for the session"""
        
        summary_parts = []
        
        doc_type = document_data.get('document_type', 'document')
        summary_parts.append(f"Document type: {doc_type.replace('_', ' ').title()}")
        
        risk_score = document_data.get('overall_risk_score', 0)
        risk_level = 'High' if risk_score > 7 else 'Medium' if risk_score > 4 else 'Low'
        summary_parts.append(f"Risk level: {risk_level} ({risk_score:.1f}/10)")
        
        key_risks_count = len(document_data.get('key_risks', []))
        summary_parts.append(f"Key risks identified: {key_risks_count}")
        
        return " | ".join(summary_parts)

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired chat sessions"""
        
        try:
            current_time = datetime.now(timezone.utc)  # Fixed deprecated datetime.utcnow()
            expired_sessions = []
            
            for session_id, session_data in self.active_sessions.items():
                last_activity = session_data.get('last_activity', current_time)
                if isinstance(last_activity, str):
                    last_activity = datetime.fromisoformat(last_activity.replace('Z', '+00:00'))
                
                if (current_time - last_activity).total_seconds() > self.session_timeout:
                    expired_sessions.append(session_id)
            
            # Clean up expired sessions
            for session_id in expired_sessions:
                del self.active_sessions[session_id]
                if session_id in self.conversation_history:
                    # Archive history before deleting (in production)
                    del self.conversation_history[session_id]
            
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            return len(expired_sessions)
            
        except Exception as e:
            self.logger.error(f"Error cleaning up expired sessions: {str(e)}")
            return 0

    def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        return len(self.active_sessions)

    def get_session_stats(self) -> Dict[str, Any]:
        """Get chat system statistics"""
        
        total_messages = sum(
            len(history) for history in self.conversation_history.values()
        )
        
        return {
            'active_sessions': len(self.active_sessions),
            'total_conversations': len(self.conversation_history),
            'total_messages': total_messages,
            'average_messages_per_session': total_messages / max(len(self.conversation_history), 1),
            'session_timeout_minutes': self.session_timeout / 60
        }
