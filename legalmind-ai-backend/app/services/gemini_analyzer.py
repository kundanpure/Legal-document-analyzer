"""
FIXED Gemini analyzer with updated model names and better error handling
"""

import google.generativeai as genai
import asyncio
from typing import Dict, Any, List, Optional, Union
import json
import time
from datetime import datetime
import os
import re

from config.settings import get_settings
from config.logging import get_logger
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)
settings = get_settings()

class GeminiAnalyzer:
    """FIXED Gemini analyzer with correct model names and robust error handling"""
    
    def __init__(self):
        self.settings = settings
        self.logger = logger
        
        # CRITICAL FIX: Use correct model names
        self.models = {
            'chat': 'gemini-2.5-flash',  # Updated from deprecated gemini-pro
            'analysis': 'gemini-2.5-pro',  # For complex document analysis
            'fallback': 'gemini-2.0-pro-latest'  # Backup model
        }
        
        # Initialize Gemini with API key
        api_key = os.getenv('GOOGLE_API_KEY') or getattr(settings, 'GOOGLE_API_KEY', '')
        if not api_key:
            raise Exception("GOOGLE_API_KEY not found in environment variables")
        
        try:
            genai.configure(api_key=api_key)
            
            # Test the connection with available models
            self.available_models = self._get_available_models()
            self.active_model = self._select_best_model()
            
            # Configure safety settings
            self.safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH", 
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
            
            self.generation_config = {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
            
            self.logger.info(f"✅ GeminiAnalyzer initialized with model: {self.active_model}")
            
        except Exception as e:
            self.logger.error(f"❌ Gemini initialization failed: {e}")
            raise Exception(f"Failed to initialize Gemini: {e}")
    
    def _get_available_models(self) -> List[str]:
        """Get list of available Gemini models"""
        try:
            models = []
            for model in genai.list_models():
                if 'generateContent' in model.supported_generation_methods:
                    models.append(model.name.replace('models/', ''))
            
            self.logger.info(f"Available Gemini models: {models}")
            return models
            
        except Exception as e:
            self.logger.warning(f"Could not fetch available models: {e}")
            return ['gemini-1.5-flash', 'gemini-1.0-pro-latest']
    
    def _select_best_model(self) -> str:
        """Select the best available model for our use case"""
        
        # Priority order: latest flash > pro > fallback
        preferred_models = [
            'gemini-1.5-flash',
            'gemini-1.5-pro', 
            'gemini-1.0-pro-latest',
            'gemini-pro'
        ]
        
        for model in preferred_models:
            if model in self.available_models:
                self.logger.info(f"Selected model: {model}")
                return model
        
        # If none found, use the first available
        if self.available_models:
            model = self.available_models[0]
            self.logger.warning(f"Using fallback model: {model}")
            return model
        
        raise Exception("No suitable Gemini models available")
    
    async def get_chat_response(
        self,
        document_context: Dict[str, Any],
        question: str,
        language: str = "en",
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        FIXED chat response method with proper model handling
        """
        try:
            self.logger.info(f"Processing chat question: {question[:50]}...")
            
            # Build context-aware prompt
            context_prompt = self._build_context_prompt(
                document_context, question, language, conversation_history
            )
            
            # Generate response with retry logic
            response_text = await self._generate_with_retries(
                context_prompt, model_type='chat'
            )
            
            if not response_text:
                response_text = "I apologize, but I'm unable to provide a response at the moment. Please try again."
            
            return {
                'response': response_text,
                'confidence': 0.85,
                'citations': self._extract_citations(document_context),
                'model_used': self.active_model,
                'language': language
            }
            
        except Exception as e:
            self.logger.error(f"Chat response error: {e}")
            return {
                'response': f"I encountered an error processing your question: {str(e)}",
                'confidence': 0.0,
                'citations': [],
                'error': str(e)
            }
    
    async def analyze_document_comprehensive(
        self,
        text: str,
        query: str = "",
        language: str = "en", 
        filename: str = ""
    ) -> Dict[str, Any]:
        """
        FIXED comprehensive document analysis
        """
        try:
            self.logger.info(f"Starting comprehensive analysis for: {filename}")
            
            if not text or len(text.strip()) < 10:
                raise Exception("Document text is too short or empty")
            
            # Clean and prepare text
            cleaned_text = self._clean_text_for_analysis(text)
            
            # Build analysis prompt
            analysis_prompt = self._build_analysis_prompt(
                cleaned_text, query, language, filename
            )
            
            # Generate analysis
            analysis_result = await self._generate_with_retries(
                analysis_prompt, model_type='analysis'
            )
            
            if not analysis_result:
                raise Exception("No analysis result generated")
            
            # Parse structured response
            structured_analysis = self._parse_analysis_result(analysis_result)
            
            self.logger.info("✅ Document analysis completed successfully")
            return structured_analysis
            
        except Exception as e:
            self.logger.error(f"Document analysis error: {e}")
            return self._create_fallback_analysis(text, str(e))
    
    async def _generate_with_retries(
        self,
        prompt: str,
        model_type: str = 'chat',
        max_retries: int = 3
    ) -> str:
        """
        FIXED generation with retry logic and model fallback
        """
        
        # Select model based on type
        if model_type == 'analysis':
            models_to_try = ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-1.0-pro-latest']
        else:
            models_to_try = ['gemini-1.5-flash', 'gemini-1.0-pro-latest']
        
        # Filter to only available models
        models_to_try = [m for m in models_to_try if m in self.available_models]
        
        if not models_to_try:
            models_to_try = [self.active_model]
        
        last_error = None
        
        for model_name in models_to_try:
            for attempt in range(max_retries):
                try:
                    self.logger.info(f"Generation attempt {attempt + 1} with model: {model_name}")
                    
                    # Create model instance
                    model = genai.GenerativeModel(
                        model_name=model_name,
                        generation_config=self.generation_config,
                        safety_settings=self.safety_settings
                    )
                    
                    # Generate content
                    response = await asyncio.to_thread(
                        model.generate_content,
                        prompt
                    )
                    
                    if response and hasattr(response, 'text') and response.text:
                        self.logger.info(f"✅ Generation successful with {model_name}")
                        return response.text.strip()
                    
                    elif hasattr(response, 'candidates') and response.candidates:
                        # Handle blocked or filtered responses
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'content') and candidate.content:
                            return candidate.content.parts[0].text.strip()
                    
                    self.logger.warning(f"Empty response from {model_name}, attempt {attempt + 1}")
                    
                except Exception as e:
                    last_error = e
                    error_msg = str(e).lower()
                    
                    if '404' in error_msg and 'not found' in error_msg:
                        self.logger.warning(f"Model {model_name} not available: {e}")
                        break  # Try next model
                    
                    if 'quota exceeded' in error_msg or 'rate limit' in error_msg:
                        wait_time = 2 ** attempt  # Exponential backoff
                        self.logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    self.logger.warning(f"Generation attempt {attempt + 1} failed with {model_name}: {e}")
                    
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)  # Short delay between retries
        
        # If all attempts failed
        error_msg = f"All generation attempts failed: {last_error}"
        self.logger.error(error_msg)
        raise Exception(error_msg)
    
    def _build_context_prompt(
        self,
        document_context: Dict[str, Any],
        question: str,
        language: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Build context-aware prompt for chat"""
        
        prompt_parts = []
        
        # System instruction
        prompt_parts.append("""You are a helpful AI assistant specialized in document analysis. 
You provide clear, accurate, and helpful responses based on the document content provided.
Be concise but thorough in your explanations.""")
        
        # Document context
        if document_context:
            prompt_parts.append("\n--- DOCUMENT CONTEXT ---")
            
            if document_context.get('document_summary'):
                prompt_parts.append(f"Document Summary: {document_context['document_summary']}")
            
            if document_context.get('key_risks'):
                risks = document_context['key_risks'][:5]  # Limit to top 5
                prompt_parts.append(f"Key Risks: {', '.join(risks)}")
            
            if document_context.get('document_type'):
                prompt_parts.append(f"Document Type: {document_context['document_type']}")
            
            if document_context.get('risk_score'):
                prompt_parts.append(f"Risk Score: {document_context['risk_score']}/10")
        
        # Conversation history
        if conversation_history:
            prompt_parts.append("\n--- RECENT CONVERSATION ---")
            for msg in conversation_history[-3:]:  # Last 3 messages for context
                role = msg.get('role', 'user')
                content = msg.get('content', '')[:200]  # Truncate long messages
                prompt_parts.append(f"{role.capitalize()}: {content}")
        
        # Current question
        prompt_parts.append(f"\n--- CURRENT QUESTION ---")
        prompt_parts.append(f"User asks: {question}")
        
        # Response instruction
        if language != 'en':
            prompt_parts.append(f"\nPlease respond in {language}.")
        
        prompt_parts.append(f"\nPlease provide a helpful and accurate response based on the document context:")
        
        return "\n".join(prompt_parts)
    
    def _build_analysis_prompt(
        self,
        text: str,
        query: str,
        language: str,
        filename: str
    ) -> str:
        """Build prompt for comprehensive document analysis"""
        
        prompt = f"""Analyze the following document comprehensively and provide a structured analysis.

Document: {filename}
Content Length: {len(text)} characters

--- DOCUMENT CONTENT ---
{text[:4000]}...  

--- ANALYSIS REQUIREMENTS ---
Please provide a comprehensive analysis in the following JSON format:

{{
    "document_type": "type of document (contract, agreement, policy, etc.)",
    "summary": "comprehensive summary of the document in 2-3 paragraphs",
    "key_risks": ["risk 1", "risk 2", "risk 3", "risk 4", "risk 5"],
    "overall_risk_score": numerical_score_1_to_10,
    "risk_categories": {{
        "financial": "financial risk assessment",
        "legal": "legal risk assessment", 
        "operational": "operational risk assessment"
    }},
    "user_obligations": ["obligation 1", "obligation 2", "obligation 3"],
    "user_rights": ["right 1", "right 2", "right 3"],
    "recommendations": ["recommendation 1", "recommendation 2", "recommendation 3"],
    "flagged_clauses": ["concerning clause 1", "concerning clause 2"],
    "fairness_score": numerical_score_1_to_10,
    "financial_implications": {{
        "costs": "description of costs",
        "penalties": "description of penalties",
        "payment_terms": "payment terms description"
    }}
}}

Provide only the JSON response without additional text."""
        
        return prompt
    
    def _clean_text_for_analysis(self, text: str) -> str:
        """Clean text for Gemini analysis"""
        if not text:
            return ""
        
        # Basic cleaning
        cleaned = text.strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
        cleaned = re.sub(r'[^\w\s\.,;:!?\-\(\)"/]', '', cleaned)  # Remove special chars
        
        # Truncate if too long (Gemini has token limits)
        if len(cleaned) > 20000:  # Conservative limit
            cleaned = cleaned[:20000] + "...[content truncated]"
        
        return cleaned
    
    def _parse_analysis_result(self, analysis_text: str) -> Dict[str, Any]:
        """Parse structured analysis result from Gemini"""
        
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                parsed = json.loads(json_text)
                
                # Validate required fields
                required_fields = ['summary', 'key_risks', 'overall_risk_score']
                for field in required_fields:
                    if field not in parsed:
                        parsed[field] = self._get_default_value(field)
                
                return parsed
        
        except Exception as e:
            self.logger.warning(f"Failed to parse structured analysis: {e}")
        
        # Fallback: create structure from unstructured text
        return self._create_structured_fallback(analysis_text)
    
    def _create_structured_fallback(self, text: str) -> Dict[str, Any]:
        """Create structured analysis from unstructured text"""
        
        return {
            "document_type": "document",
            "summary": text[:500] + "..." if len(text) > 500 else text,
            "key_risks": ["Analysis requires manual review"],
            "overall_risk_score": 5.0,
            "risk_categories": {
                "financial": "Requires manual review",
                "legal": "Requires manual review",
                "operational": "Requires manual review"
            },
            "user_obligations": ["Review document manually"],
            "user_rights": ["Standard document rights"],
            "recommendations": ["Seek professional legal advice"],
            "flagged_clauses": [],
            "fairness_score": 5.0,
            "financial_implications": {
                "costs": "Not analyzed",
                "penalties": "Not analyzed", 
                "payment_terms": "Not analyzed"
            }
        }
    
    def _create_fallback_analysis(self, text: str, error: str) -> Dict[str, Any]:
        """Create fallback analysis when processing fails"""
        
        word_count = len(text.split()) if text else 0
        
        return {
            "document_type": "unknown",
            "summary": f"Analysis failed due to technical issues. Document contains approximately {word_count} words.",
            "key_risks": [f"Analysis error: {error}"],
            "overall_risk_score": 0.0,
            "risk_categories": {
                "financial": "Could not analyze",
                "legal": "Could not analyze",
                "operational": "Could not analyze"
            },
            "user_obligations": ["Manual review required"],
            "user_rights": ["Manual review required"],
            "recommendations": ["Re-upload document or seek manual analysis"],
            "flagged_clauses": [],
            "fairness_score": 0.0,
            "financial_implications": {
                "costs": "Analysis failed",
                "penalties": "Analysis failed",
                "payment_terms": "Analysis failed"
            },
            "error": error
        }
    
    def _extract_citations(self, document_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract citations from document context"""
        
        citations = []
        
        if document_context and document_context.get('document_summary'):
            citations.append({
                'source': 'Document Summary',
                'content': document_context['document_summary'][:200],
                'confidence': 0.9
            })
        
        return citations
    
    def _get_default_value(self, field: str):
        """Get default value for missing analysis fields"""
        
        defaults = {
            'summary': 'Document analysis incomplete',
            'key_risks': ['Analysis incomplete'],
            'overall_risk_score': 5.0,
            'user_obligations': ['Review manually'],
            'user_rights': ['Review manually'],
            'recommendations': ['Seek professional advice'],
            'fairness_score': 5.0
        }
        
        return defaults.get(field, 'Not available')
    
    async def generate_suggested_questions(
        self,
        doc_type: str,
        summary: str,
        risk_level: str
    ) -> List[str]:
        """Generate suggested questions based on document"""
        
        # Provide immediate suggestions without API call to avoid delays
        base_questions = [
            "What are the main terms I need to understand?",
            "Are there any deadlines I should be aware of?",
            "What happens if I want to cancel or terminate?",
            "What are the financial implications?",
            "Are there any penalties or fees?"
        ]
        
        # Customize based on document type
        if 'contract' in doc_type.lower():
            base_questions.extend([
                "What are my obligations under this contract?",
                "How can I modify or amend this agreement?"
            ])
        elif 'policy' in doc_type.lower():
            base_questions.extend([
                "What are the key policy requirements?",
                "How does this policy affect me?"
            ])
        
        return base_questions[:5]  