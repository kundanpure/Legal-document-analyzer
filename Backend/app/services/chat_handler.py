"""
Strict Chat handler service (no fallbacks).
Requires: app.services.gemini_analyzer.GeminiAnalyzer
          app.services.translation_service.TranslationService
"""

from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime, timezone

from config.settings import get_settings
from config.logging import get_logger
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)
settings = get_settings()


class ChatError(Exception):
    """Raised for chat-related errors."""
    pass


class ChatHandler:
    """
    Strict chat handler:
    - Requires GeminiAnalyzer and TranslationService to import/initialize.
    - Requires `document_text` in the context when a file is used.
    - No graceful fallbacks; explicit errors if something is missing.
    """

    def __init__(self):
        # STRICT: import must succeed or raise
        from app.services.gemini_analyzer import GeminiAnalyzer
        from app.services.translation_service import TranslationService

        self.gemini_analyzer = GeminiAnalyzer()
        self.translation_service = TranslationService()

        # In-memory session/history (replace with Redis for prod)
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.conversation_history: Dict[str, List[Dict[str, Any]]] = {}

        self.max_history_length = 10
        self.session_timeout = 3600  # seconds

        logger.info("ChatHandler initialized successfully")

    # ----- Public API (used by /api/chat) -----
    async def process_message(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Process a single chat turn.
        Requires `context.document_text` whenever a document is referenced.
        """
        if not isinstance(message, str) or not message.strip():
            raise ChatError("Empty message")

        # Build strict AI context (must carry document_text if provided upstream)
        ai_context = self._prepare_ai_context_for_processing(
            context or {}, conversation_history or []
        )

        # If a document is in play, require text
        if context:
            doc_text = ai_context.get("document_text", "").strip()
            if not doc_text:
                raise ChatError("document_text is required in context for document-grounded chat")

        # Call Gemini
        ai_response = await self.gemini_analyzer.get_chat_response(
            document_context=ai_context,
            question=message,
            language=language,
            conversation_history=conversation_history or [],
        )

        response_text = (ai_response or {}).get("response", "")
        if not response_text.strip():
            raise ChatError("Empty response from GeminiAnalyzer")

        enhanced = await self._postprocess_response(
            response_text=response_text,
            user_message=message,
            language=language,
            citations=(ai_response or {}).get("citations", []),
            confidence=(ai_response or {}).get("confidence", 0.85),
            warnings=(ai_response or {}).get("warnings", []),
        )

        return {
            "response": enhanced["response"],
            "confidence": enhanced.get("confidence", 0.85),
            "citations": enhanced.get("citations", []),
            "warnings": enhanced.get("warnings", []),
            "suggestions": self._followups_for(message),
        }

    # ----- Internal helpers -----
    def _prepare_ai_context_for_processing(
        self,
        document_context: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Prepare the context we pass to Gemini.
        Ensures that if analysis exists, we forward summary/risks/etc.,
        and **always** forwards document_text if present.
        """
        ctx: Dict[str, Any] = {}

        if document_context:
            analysis = document_context.get("analysis", {}) or {}

            # Analysis-derived signals (optional but helpful)
            ctx.update({
                "document_summary": analysis.get("summary", "") or document_context.get("summary", ""),
                "key_risks": analysis.get("key_risks", []) or document_context.get("key_risks", []),
                "document_type": analysis.get("document_type", document_context.get("document_type", "unknown")),
                "risk_score": analysis.get("overall_risk_score", document_context.get("overall_risk_score", 0)),
                "recommendations": analysis.get("recommendations", document_context.get("recommendations", [])),
            })

            # CRITICAL: the raw text used for QA
            raw_text = document_context.get("document_text") or analysis.get("full_text", "")
            if raw_text:
                ctx["document_text"] = raw_text

        if history:
            ctx["recent_conversation"] = [
                {
                    "role": msg.get("type", msg.get("role", "user")),
                    "content": msg.get("content", msg.get("message", "")),
                }
                for msg in history[-5:]
            ]

        return ctx

    async def _postprocess_response(
        self,
        response_text: str,
        user_message: str,
        language: str,
        citations: List[Dict[str, Any]],
        confidence: float,
        warnings: List[str],
    ) -> Dict[str, Any]:
        """
        Minimal post-processing.
        - Optional translation (no fallback).
        """
        final_text = response_text

        if language and language != "en":
            tr = await self.translation_service.translate_chat_response(
                final_text, target_lang=language, source_lang="en"
            )
            final_text = tr.get("translated_text", "").strip()
            if not final_text:
                raise ChatError("Translation returned empty text")

        return {
            "response": final_text,
            "citations": citations or [],
            "confidence": confidence,
            "warnings": warnings or [],
        }

    # Small, static follow-ups (no model call, not a fallback)
    def _followups_for(self, user_message: str) -> List[str]:
        m = user_message.lower()
        if "risk" in m:
            return [
                "Which risks are most severe here?",
                "How can I mitigate these risks?",
                "What happens if these risks occur?",
                "Are there industry-standard risk clauses?"
            ]
        if "fee" in m or "cost" in m or "price" in m:
            return [
                "Are there any hidden or variable fees?",
                "What are the payment terms?",
                "Are late payment penalties specified?",
                "Can fees be negotiated?"
            ]
        if "terminate" in m or "cancel" in m:
            return [
                "What are the termination conditions?",
                "Is there an early termination penalty?",
                "How much notice is required?",
                "Who can terminate and under what circumstances?"
            ]
        return [
            "Can you explain that in simpler terms?",
            "What should I be most careful about?",
            "Is this clause standard?",
            "What are my key obligations?"
        ]
