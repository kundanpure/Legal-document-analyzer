"""
Strict Gemini analyzer (no graceful fallbacks).
- Exposes `get_chat_response` and `analyze_document_comprehensive`.
- Chat prompt ALWAYS includes `document_text` (truncated).
"""

import asyncio
import json
import os
import re
from typing import Dict, Any, List, Optional

import google.generativeai as genai

from config.settings import get_settings
from config.logging import get_logger
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)
settings = get_settings()


class GeminiAnalyzer:
    """
    Strict Gemini wrapper:
    - Requires GOOGLE_API_KEY.
    - Lists available models and picks a supported one; if none available -> raises.
    - No content fallbacks; errors bubble up.
    """

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY") or getattr(settings, "GOOGLE_API_KEY", "")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set")

        genai.configure(api_key=api_key)

        self.generation_config = {
            "temperature": 0.4,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        # Keep safety sane; raise on policy block
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        # Discover available models
        self.available_models = self._list_supported_models()
        if not self.available_models:
            raise RuntimeError("No Gemini models with generateContent capability are available to your API key")

        # Preferred order for chat / analysis
        self.preferred_chat = ["gemini-1.5-flash", "gemini-1.5-pro"]
        self.preferred_analysis = ["gemini-1.5-pro", "gemini-1.5-flash"]

        logger.info(f"✅ GeminiAnalyzer initialized. Available models: {self.available_models}")

    # ---------- Public Methods ----------

    async def get_chat_response(
        self,
        document_context: Dict[str, Any],
        question: str,
        language: str = "en",
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a chat response using document-grounded context.
        Requires document_context['document_text'] (non-empty).
        """
        if not isinstance(question, str) or not question.strip():
            raise RuntimeError("Empty question")

        doc_text = (document_context or {}).get("document_text", "").strip()
        if not doc_text:
            raise RuntimeError("document_text missing in document_context")

        prompt = self._build_context_prompt(document_context, question, language, conversation_history)

        text = await self._generate_with_models(
            prompt=prompt,
            candidates=self._select_models(self.preferred_chat),
        )
        if not text:
            raise RuntimeError("Model returned empty response")

        return {
            "response": text,
            "confidence": 0.85,
            "citations": self._extract_citations(document_context),
            "language": language,
        }

    async def analyze_document_comprehensive(
        self,
        text: str,
        query: str = "",
        language: str = "en",
        filename: str = "",
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis (structured JSON). Strict: raises on failure.
        """
        cleaned = self._clean_text_for_analysis(text)
        if not cleaned or len(cleaned) < 20:
            raise RuntimeError("Provided text is too short for analysis")

        prompt = self._build_analysis_prompt(cleaned, query, language, filename)

        analysis = await self._generate_with_models(
            prompt=prompt,
            candidates=self._select_models(self.preferred_analysis),
        )
        if not analysis:
            raise RuntimeError("Model returned empty analysis")

        parsed = self._parse_analysis_result(analysis)
        return parsed

    # ---------- Internal: Model Selection & Generation ----------

    def _list_supported_models(self) -> List[str]:
        models = []
        try:
            for m in genai.list_models():
                if "generateContent" in getattr(m, "supported_generation_methods", []):
                    models.append(m.name.replace("models/", ""))
        except Exception as e:
            raise RuntimeError(f"Failed to list models: {e}")
        return models

    def _select_models(self, preferred: List[str]) -> List[str]:
        selected = [m for m in preferred if m in self.available_models]
        if not selected:
            # If none of the preferred exist, try *any* available (still strict, no silent fallback outside available)
            selected = self.available_models[:2]
        return selected

    async def _generate_with_models(self, prompt: str, candidates: List[str]) -> str:
        last_err: Optional[Exception] = None
        for name in candidates:
            try:
                logger.info(f"Generation with model: {name}")
                model = genai.GenerativeModel(
                    model_name=name,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings,
                )
                resp = await asyncio.to_thread(model.generate_content, prompt)
                # Prefer resp.text if present
                if hasattr(resp, "text") and resp.text:
                    return resp.text.strip()
                # Otherwise try first candidate text if present
                if getattr(resp, "candidates", None):
                    cand = resp.candidates[0]
                    parts = getattr(getattr(cand, "content", None), "parts", None)
                    if parts and len(parts) and getattr(parts[0], "text", None):
                        return parts[0].text.strip()
                raise RuntimeError("Empty model response")
            except Exception as e:
                last_err = e
                logger.warning(f"Model {name} failed: {e}")
                # try next candidate
        raise RuntimeError(f"All candidate models failed. Last error: {last_err}")

    # ---------- Prompt Builders ----------

    def _build_context_prompt(
        self,
        document_context: Dict[str, Any],
        question: str,
        language: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        parts: List[str] = []
        parts.append(
            "You are an assistant specialized in legal/contract/document analysis. "
            "Answer **strictly** from the provided DOCUMENT TEXT. "
            "If the answer is not supported by that text, say you cannot find it in the document."
        )

        # High-level context (optional signals)
        if document_context:
            parts.append("\n--- DOCUMENT CONTEXT ---")
            if document_context.get("document_summary"):
                parts.append(f"Document Summary: {document_context['document_summary']}")
            if document_context.get("key_risks"):
                risks = document_context["key_risks"][:5]
                parts.append(f"Key Risks: {', '.join(risks)}")
            if document_context.get("document_type"):
                parts.append(f"Document Type: {document_context['document_type']}")
            if document_context.get("risk_score") is not None:
                parts.append(f"Risk Score: {document_context['risk_score']} / 10")

        # CRITICAL: actual document text
        raw_text = (document_context or {}).get("document_text", "")
        if raw_text:
            snippet = raw_text.strip()
            # We already truncated upstream to ~30k; truncate again to keep prompt safe
            max_chars = 12000
            if len(snippet) > max_chars:
                snippet = snippet[:max_chars] + "...[truncated]"
            parts.append("\n--- DOCUMENT TEXT (TRUNCATED) ---")
            parts.append(snippet)

        # Minimal recent conversation
        if conversation_history:
            parts.append("\n--- RECENT CONVERSATION ---")
            for msg in conversation_history[-3:]:
                role = msg.get("role", "user")
                content = (msg.get("content") or msg.get("message") or "")[:300]
                parts.append(f"{role.capitalize()}: {content}")

        parts.append("\n--- CURRENT QUESTION ---")
        parts.append(f"User asks: {question}")

        if language and language != "en":
            parts.append(f"\nPlease answer in {language}.")
        parts.append("\nAnswer only from the DOCUMENT TEXT above. If not present, say so clearly.")
        return "\n".join(parts)

    def _build_analysis_prompt(self, text: str, query: str, language: str, filename: str) -> str:
        return f"""Analyze the following document comprehensively and return ONLY JSON.

Document: {filename}
Content length (chars): {len(text)}

--- DOCUMENT CONTENT (TRUNCATED) ---
{text[:8000]}...

--- REQUIRED JSON SHAPE ---
{{
  "document_type": "contract | agreement | policy | other",
  "summary": "2-3 paragraphs summary",
  "key_risks": ["...", "...", "..."],
  "overall_risk_score": 0-10,
  "risk_categories": {{
    "financial": "string",
    "legal": "string",
    "operational": "string"
  }},
  "user_obligations": ["..."],
  "user_rights": ["..."],
  "recommendations": ["..."],
  "flagged_clauses": ["..."],
  "fairness_score": 0-10,
  "financial_implications": {{
    "costs": "string",
    "penalties": "string",
    "payment_terms": "string"
  }}
}}

Return only valid JSON with double quotes, no comments, no extra text.
"""

    # ---------- Parsing / Utilities ----------

    def _clean_text_for_analysis(self, text: str) -> str:
        if not text:
            return ""
        cleaned = text.strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"[^\w\s\.,;:!?\-\(\)\"/]", "", cleaned)
        # Hard cap to be safe
        return cleaned[:20000] + ("...[truncated]" if len(cleaned) > 20000 else "")

    def _parse_analysis_result(self, analysis_text: str) -> Dict[str, Any]:
        # Extract first {...} blob
        m = re.search(r"\{.*\}", analysis_text, flags=re.DOTALL)
        if not m:
            raise RuntimeError("Model did not return JSON")

        try:
            parsed = json.loads(m.group(0))
        except Exception as e:
            raise RuntimeError(f"Invalid JSON from model: {e}")

        # Minimal validation
        for req in ["summary", "key_risks", "overall_risk_score"]:
            if req not in parsed:
                raise RuntimeError(f"Missing '{req}' in analysis JSON")

        return parsed

    def _extract_citations(self, document_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        cites: List[Dict[str, Any]] = []
        if document_context.get("document_summary"):
            cites.append({
                "source": "Document Summary",
                "content": document_context["document_summary"][:200],
                "confidence": 0.9,
            })
        return cites
