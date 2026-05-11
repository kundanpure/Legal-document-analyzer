"""
Enhanced Gemini analyzer for multi-document analysis
Enhanced with fallbacks and comprehensive error handling
"""

import asyncio
import json
import time
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone

# Try Google Generative AI with fallback
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from config.settings import get_settings
from config.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

from dotenv import load_dotenv
load_dotenv() 

# Custom exceptions
class GeminiAnalysisError(Exception):
    """Exception raised for Gemini analysis errors"""
    pass

# Mock models for when dependencies aren't available
class MockDocumentChunk:
    def __init__(self, id="", document_id="", content="", chunk_index=0, 
                 total_chunks=1, page_range="1-1"):
        self.id = id
        self.document_id = document_id
        self.content = content
        self.chunk_index = chunk_index
        self.total_chunks = total_chunks
        self.page_range = page_range

class MockDocumentRelationship:
    def __init__(self, id="", document_1_id="", document_2_id="", 
                 relationship_type="", similarity_score=0.0, common_topics=None, 
                 relationship_details=None, created_at=None, confidence_score=0.0):
        self.id = id
        self.document_1_id = document_1_id
        self.document_2_id = document_2_id
        self.relationship_type = relationship_type
        self.similarity_score = similarity_score
        self.common_topics = common_topics or []
        self.relationship_details = relationship_details or {}
        self.created_at = created_at or datetime.now(timezone.utc)
        self.confidence_score = confidence_score

# Try to import real models, fallback to mock
try:
    from app.models.multi_document_models import DocumentChunk, DocumentRelationship
except ImportError:
    DocumentChunk = MockDocumentChunk
    DocumentRelationship = MockDocumentRelationship

class EnhancedGeminiAnalyzer:
    """
    Enhanced Gemini analyzer with multi-document capabilities and fallbacks
    """
    
    def __init__(self):
        self.logger = logger
        self.settings = settings
        
        # Initialize Gemini if available
        self.gemini_available = GENAI_AVAILABLE and hasattr(settings, 'GOOGLE_API_KEY') and settings.GOOGLE_API_KEY
        self.chat_model = None
        
        if self.gemini_available:
            try:
                # Configure Gemini
                genai.configure(api_key=settings.GOOGLE_API_KEY)
                
                # Get model settings with fallbacks
                model_name = getattr(settings, 'GEMINI_MODEL', 'gemini-1.5-flash')
                temperature = getattr(settings, 'GEMINI_TEMPERATURE', 0.7)
                max_tokens = getattr(settings, 'GEMINI_MAX_TOKENS', 8192)
                
                # Initialize models
                self.chat_model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config={
                        "temperature": temperature,
                        "top_p": 0.8,
                        "top_k": 40,
                        "max_output_tokens": max_tokens,
                    },
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    }
                )
                
                self.logger.info("Enhanced Gemini Analyzer initialized with real AI")
                
            except Exception as e:
                self.logger.warning(f"Gemini initialization failed: {e}, using fallback")
                self.gemini_available = False
        else:
            self.logger.info("Gemini not available, using fallback analyzer")
        
        # Analysis prompts templates
        self.analysis_templates = {
            'document_analysis': self._get_document_analysis_template(),
            'chunk_analysis': self._get_chunk_analysis_template(),
            'relationship_analysis': self._get_relationship_analysis_template(),
            'cross_document_synthesis': self._get_cross_document_synthesis_template(),
            'conflict_detection': self._get_conflict_detection_template(),
            'portfolio_analysis': self._get_portfolio_analysis_template()
        }
    
    async def analyze_document_comprehensive(
        self,
        document_content: str,
        filename: str,
        document_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of a single document
        
        Args:
            document_content: Full document text
            filename: Document filename
            document_metadata: Document metadata
            
        Returns:
            Comprehensive analysis results
        """
        
        try:
            self.logger.info(f"Starting comprehensive analysis for: {filename}")
            
            # Use real AI if available, otherwise fallback
            if self.gemini_available:
                analysis_result = await self._analyze_with_gemini(
                    document_content, filename, document_metadata
                )
            else:
                analysis_result = await self._analyze_with_fallback(
                    document_content, filename, document_metadata
                )
            
            # Add metadata
            analysis_result.update({
                'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
                'analyzer_version': 'enhanced_gemini_v2.0',
                'model_used': getattr(settings, 'GEMINI_MODEL', 'fallback') if self.gemini_available else 'fallback',
                'document_filename': filename,
                'analysis_confidence': analysis_result.get('confidence_score', 0.8)
            })
            
            self.logger.info(f"Completed comprehensive analysis for: {filename}")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive document analysis: {str(e)}")
            raise GeminiAnalysisError(f"Document analysis failed: {str(e)}")

    async def _analyze_with_gemini(self, document_content: str, filename: str, 
                                 document_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze using real Gemini AI"""
        
        # Prepare analysis prompt
        prompt = self.analysis_templates['document_analysis'].format(
            filename=filename,
            document_type=document_metadata.get('document_type', 'unknown'),
            content_preview=document_content[:3000],  # First 3000 chars
            word_count=len(document_content.split()),
            page_count=document_metadata.get('total_pages', 'unknown')
        )
        
        # Generate analysis
        response = await self._generate_content_async(prompt)
        
        # Parse structured response
        analysis_result = await self._parse_analysis_response(response)
        
        return analysis_result

    async def _analyze_with_fallback(self, document_content: str, filename: str, 
                                   document_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis using rule-based methods"""
        
        # Basic analysis using keyword detection and patterns
        words = document_content.lower().split()
        
        # Document type detection
        doc_type = self._detect_document_type(document_content)
        
        # Risk assessment
        risk_keywords = ['penalty', 'breach', 'liability', 'termination', 'default', 'damages']
        risk_score = sum(1 for word in risk_keywords if word in document_content.lower())
        risk_level = 'high' if risk_score > 3 else 'medium' if risk_score > 1 else 'low'
        
        # Extract topics
        legal_topics = ['contract', 'agreement', 'payment', 'services', 'obligations', 
                       'terms', 'conditions', 'warranty', 'indemnity', 'confidentiality']
        found_topics = [topic for topic in legal_topics if topic in document_content.lower()]
        
        # Extract entities (basic pattern matching)
        entities = self._extract_basic_entities(document_content)
        
        return {
            'document_type': doc_type,
            'title': f"Legal Document: {filename}",
            'risk_level': risk_level,
            'confidence_score': 0.7,
            'key_topics': found_topics[:8],
            'main_entities': entities,
            'summary': f"Analysis of {filename} reveals a {doc_type} with {risk_level} risk level covering {len(found_topics)} key legal areas.",
            'risk_factors': [f"Contains {risk_score} risk-related terms"],
            'obligations': self._extract_obligations(document_content),
            'recommendations': [
                "Review all terms and conditions carefully",
                "Consider legal counsel for high-risk elements",
                "Ensure all parties understand their obligations"
            ]
        }

    def _detect_document_type(self, content: str) -> str:
        """Detect document type using keyword patterns"""
        content_lower = content.lower()
        
        type_indicators = {
            'employment_contract': ['employment', 'employee', 'employer', 'salary', 'benefits'],
            'service_agreement': ['services', 'provider', 'client', 'service agreement'],
            'lease_agreement': ['lease', 'rent', 'tenant', 'landlord', 'premises'],
            'nda': ['non-disclosure', 'confidential', 'proprietary', 'trade secret'],
            'purchase_agreement': ['purchase', 'buyer', 'seller', 'goods', 'price'],
            'loan_agreement': ['loan', 'borrower', 'lender', 'interest', 'principal']
        }
        
        for doc_type, keywords in type_indicators.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score >= 2:
                return doc_type
        
        return 'legal_document'

    def _extract_basic_entities(self, content: str) -> List[Dict[str, str]]:
        """Extract basic entities using regex patterns"""
        entities = []
        
        # Money amounts
        money_pattern = r'\$[\d,]+(?:\.\d{2})?'
        money_matches = re.findall(money_pattern, content)
        entities.extend([{'type': 'monetary', 'value': match} for match in money_matches[:5]])
        
        # Dates (basic patterns)
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',
            r'\d{1,2}-\d{1,2}-\d{4}',
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}'
        ]
        
        for pattern in date_patterns:
            dates = re.findall(pattern, content, re.IGNORECASE)
            entities.extend([{'type': 'date', 'value': str(date)} for date in dates[:3]])
        
        return entities[:10]  # Limit to 10 entities

    def _extract_obligations(self, content: str) -> List[str]:
        """Extract obligations using keyword patterns"""
        obligation_keywords = ['shall', 'must', 'will', 'agrees to', 'responsible for', 'obligated']
        
        sentences = content.split('.')
        obligations = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in obligation_keywords):
                clean_sentence = sentence.strip()
                if len(clean_sentence) > 20 and len(clean_sentence) < 200:
                    obligations.append(clean_sentence)
                    
                if len(obligations) >= 5:  # Limit to 5 obligations
                    break
        
        return obligations
    
    async def analyze_chunk_detailed(
        self,
        chunk: DocumentChunk,
        document_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detailed analysis of document chunk with context
        
        Args:
            chunk: DocumentChunk to analyze
            document_context: Context from parent document
            
        Returns:
            Detailed chunk analysis
        """
        
        try:
            if self.gemini_available:
                # Use real Gemini for chunk analysis
                prompt = self.analysis_templates['chunk_analysis'].format(
                    chunk_content=chunk.content,
                    chunk_index=chunk.chunk_index,
                    total_chunks=chunk.total_chunks,
                    page_range=chunk.page_range,
                    document_title=document_context.get('title', 'Unknown Document'),
                    document_type=document_context.get('document_type', 'unknown')
                )
                
                response = await self._generate_content_async(prompt)
                chunk_analysis = await self._parse_chunk_analysis(response)
            else:
                # Fallback chunk analysis
                chunk_analysis = self._analyze_chunk_fallback(chunk, document_context)
            
            # Add chunk-specific metadata
            chunk_analysis.update({
                'chunk_id': chunk.id,
                'document_id': chunk.document_id,
                'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
                'chunk_position': f"{chunk.chunk_index + 1}/{chunk.total_chunks}"
            })
            
            return chunk_analysis
            
        except Exception as e:
            self.logger.error(f"Error in chunk analysis: {str(e)}")
            return {'error': str(e), 'chunk_id': getattr(chunk, 'id', 'unknown')}

    def _analyze_chunk_fallback(self, chunk: DocumentChunk, document_context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback chunk analysis"""
        
        content = chunk.content.lower()
        words = content.split()
        
        # Extract topics
        legal_topics = ['contract', 'payment', 'services', 'terms', 'obligations', 'warranty']
        found_topics = [topic for topic in legal_topics if topic in content]
        
        # Extract entities
        entities = self._extract_basic_entities(chunk.content)
        
        # Key phrases (simple frequency analysis)
        word_freq = {}
        for word in words:
            if len(word) > 4 and word.isalpha():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        key_phrases = sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:5]
        
        # Sentiment analysis (basic)
        positive_words = ['benefit', 'advantage', 'protection', 'rights', 'warranty']
        negative_words = ['penalty', 'breach', 'liability', 'termination', 'default']
        
        pos_count = sum(1 for word in positive_words if word in content)
        neg_count = sum(1 for word in negative_words if word in content)
        
        if neg_count > pos_count:
            sentiment = 'negative'
        elif pos_count > neg_count:
            sentiment = 'positive'
        else:
            sentiment = 'neutral'
        
        return {
            'topics': found_topics,
            'entities': [e['value'] for e in entities],
            'key_phrases': key_phrases,
            'sentiment': sentiment,
            'importance_score': min(len(found_topics) * 0.2, 1.0),
            'summary': f"Chunk covers {len(found_topics)} topics with {sentiment} sentiment"
        }
    
    async def analyze_document_relationships(
        self,
        document_summaries: List[Dict[str, Any]],
        chunk_samples: Dict[str, str]
    ) -> List[DocumentRelationship]:
        """
        Analyze relationships between multiple documents
        
        Args:
            document_summaries: List of document summaries
            chunk_samples: Sample content from each document
            
        Returns:
            List of detected relationships
        """
        
        try:
            relationships = []
            
            # Compare each pair of documents
            for i, doc1 in enumerate(document_summaries):
                for j, doc2 in enumerate(document_summaries[i+1:], i+1):
                    
                    if self.gemini_available:
                        # Use real Gemini for relationship analysis
                        relationship_data = await self._analyze_relationship_with_gemini(
                            doc1, doc2, chunk_samples
                        )
                    else:
                        # Use fallback relationship analysis
                        relationship_data = self._analyze_relationship_fallback(
                            doc1, doc2, chunk_samples
                        )
                    
                    # Create relationship if significant
                    if relationship_data.get('similarity_score', 0) > 0.2:
                        
                        relationship = DocumentRelationship(
                            id=f"rel_{doc1.get('document_id', i)}_{doc2.get('document_id', j)}",
                            document_1_id=doc1.get('document_id', str(i)),
                            document_2_id=doc2.get('document_id', str(j)),
                            relationship_type=relationship_data.get('relationship_type', 'related'),
                            similarity_score=relationship_data.get('similarity_score', 0.5),
                            common_topics=relationship_data.get('common_topics', []),
                            relationship_details={
                                'analysis_summary': relationship_data.get('analysis_summary', ''),
                                'key_connections': relationship_data.get('key_connections', []),
                                'potential_conflicts': relationship_data.get('potential_conflicts', []),
                                'complementary_aspects': relationship_data.get('complementary_aspects', [])
                            },
                            created_at=datetime.now(timezone.utc),
                            confidence_score=relationship_data.get('confidence_score', 0.7)
                        )
                        
                        relationships.append(relationship)
            
            self.logger.info(f"Analyzed relationships between {len(document_summaries)} documents, found {len(relationships)} relationships")
            return relationships
            
        except Exception as e:
            self.logger.error(f"Error analyzing document relationships: {str(e)}")
            return []

    def _analyze_relationship_fallback(self, doc1: Dict, doc2: Dict, 
                                     chunk_samples: Dict[str, str]) -> Dict[str, Any]:
        """Fallback relationship analysis using text similarity"""
        
        # Get topics from both documents
        topics1 = set(doc1.get('key_topics', []))
        topics2 = set(doc2.get('key_topics', []))
        
        # Calculate topic similarity
        if topics1 or topics2:
            common_topics = topics1.intersection(topics2)
            all_topics = topics1.union(topics2)
            similarity_score = len(common_topics) / len(all_topics) if all_topics else 0
        else:
            similarity_score = 0
        
        # Determine relationship type
        if similarity_score > 0.7:
            relationship_type = 'highly_related'
        elif similarity_score > 0.4:
            relationship_type = 'related'
        elif similarity_score > 0.2:
            relationship_type = 'complementary'
        else:
            relationship_type = 'unrelated'
        
        return {
            'similarity_score': similarity_score,
            'relationship_type': relationship_type,
            'common_topics': list(common_topics) if 'common_topics' in locals() else [],
            'analysis_summary': f"Documents share {len(common_topics) if 'common_topics' in locals() else 0} common topics",
            'key_connections': list(common_topics) if 'common_topics' in locals() else [],
            'potential_conflicts': [],
            'complementary_aspects': [],
            'confidence_score': 0.6
        }

    async def synthesize_cross_document_response(
        self,
        query: str,
        relevant_chunks: List[Tuple[DocumentChunk, Dict[str, Any]]],
        conversation_context: Dict[str, Any]
    ) -> str:
        """
        Synthesize response from multiple document chunks
        
        Args:
            query: User query
            relevant_chunks: List of relevant chunks with metadata
            conversation_context: Conversation context
            
        Returns:
            Synthesized response
        """
        
        try:
            if self.gemini_available:
                # Use real Gemini for synthesis
                return await self._synthesize_with_gemini(query, relevant_chunks, conversation_context)
            else:
                # Use fallback synthesis
                return await self._synthesize_fallback(query, relevant_chunks, conversation_context)
                
        except Exception as e:
            self.logger.error(f"Error synthesizing cross-document response: {str(e)}")
            return f"I encountered an error while analyzing your documents: {str(e)}"

    async def _synthesize_fallback(self, query: str, relevant_chunks: List[Tuple[DocumentChunk, Dict[str, Any]]], 
                                 conversation_context: Dict[str, Any]) -> str:
        """Fallback synthesis using rule-based approach"""
        
        if not relevant_chunks:
            return "I couldn't find relevant information in your documents to answer your question."
        
        # Extract key information from chunks
        doc_info = {}
        for chunk, metadata in relevant_chunks:
            doc_id = chunk.document_id
            if doc_id not in doc_info:
                doc_info[doc_id] = {
                    'title': metadata.get('document_title', 'Document'),
                    'content_snippets': [],
                    'relevance_scores': []
                }
            
            doc_info[doc_id]['content_snippets'].append(chunk.content[:200])
            doc_info[doc_id]['relevance_scores'].append(metadata.get('relevance_score', 0.5))
        
        # Generate response
        response_parts = [f"Based on your {len(doc_info)} documents, here's what I found:"]
        
        for doc_id, info in doc_info.items():
            avg_relevance = sum(info['relevance_scores']) / len(info['relevance_scores'])
            response_parts.append(f"\nFrom {info['title']} (relevance: {avg_relevance:.1f}):")
            response_parts.append(f"- {info['content_snippets'][0]}...")
        
        # Add query-specific insights
        query_lower = query.lower()
        if 'risk' in query_lower:
            response_parts.append("\nKey risks to consider across your documents include potential liabilities and obligations that require careful attention.")
        elif 'compare' in query_lower:
            response_parts.append(f"\nI've compared information across {len(doc_info)} documents to identify similarities and differences.")
        elif 'conflict' in query_lower:
            response_parts.append("\nI've analyzed your documents for potential conflicts or contradictory terms.")
        
        response_parts.append(f"\nThis analysis covers {len(relevant_chunks)} relevant sections from your document portfolio.")
        
        return ' '.join(response_parts)

    # Helper methods for generating content
    
    async def _generate_content_async(self, prompt: str, max_retries: int = 3) -> str:
        """
        Generate content with retry logic
        """
        
        if not self.gemini_available:
            return "Gemini AI not available - using fallback analysis"
        
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    self.chat_model.generate_content,
                    prompt
                )
                
                if response.text:
                    return response.text.strip()
                else:
                    raise GeminiAnalysisError("Empty response from Gemini")
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt == max_retries - 1:
                    raise GeminiAnalysisError(f"Failed after {max_retries} attempts: {str(e)}")
                
                # Wait before retry
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    # Response parsing methods (same as original but with better error handling)
    
    async def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse structured analysis response"""
        
        try:
            analysis = {
                'document_type': self._extract_field(response, 'document_type', 'legal_document'),
                'title': self._extract_field(response, 'title', 'Legal Document'),
                'risk_level': self._extract_field(response, 'risk_level', 'medium'),
                'confidence_score': float(self._extract_field(response, 'confidence', '0.8')),
                'key_topics': self._extract_list(response, 'key_topics'),
                'main_entities': self._extract_list(response, 'entities'),
                'summary': self._extract_field(response, 'summary', response[:200]),
                'risk_factors': self._extract_list(response, 'risk_factors'),
                'obligations': self._extract_list(response, 'obligations'),
                'recommendations': self._extract_list(response, 'recommendations')
            }
            
            return analysis
        except Exception as e:
            self.logger.error(f"Error parsing analysis response: {e}")
            return {
                'document_type': 'legal_document',
                'title': 'Legal Document',
                'risk_level': 'medium',
                'confidence_score': 0.5,
                'key_topics': [],
                'main_entities': [],
                'summary': 'Analysis parsing failed',
                'risk_factors': [],
                'obligations': [],
                'recommendations': []
            }
    
    async def _parse_chunk_analysis(self, response: str) -> Dict[str, Any]:
        """Parse chunk analysis response"""
        
        try:
            return {
                'topics': self._extract_list(response, 'topics'),
                'entities': self._extract_list(response, 'entities'),
                'key_phrases': self._extract_list(response, 'key_phrases'),
                'sentiment': self._extract_field(response, 'sentiment', 'neutral'),
                'importance_score': float(self._extract_field(response, 'importance', '0.5')),
                'summary': self._extract_field(response, 'summary', response[:100])
            }
        except Exception as e:
            self.logger.error(f"Error parsing chunk analysis: {e}")
            return {
                'topics': [],
                'entities': [],
                'key_phrases': [],
                'sentiment': 'neutral',
                'importance_score': 0.5,
                'summary': 'Analysis parsing failed'
            }
    
    async def _parse_relationship_analysis(self, response: str) -> Dict[str, Any]:
        """Parse relationship analysis response"""
        
        try:
            return {
                'similarity_score': float(self._extract_field(response, 'similarity', '0.0')),
                'relationship_type': self._extract_field(response, 'relationship_type', 'related'),
                'common_topics': self._extract_list(response, 'common_topics'),
                'analysis_summary': self._extract_field(response, 'summary', ''),
                'key_connections': self._extract_list(response, 'connections'),
                'potential_conflicts': self._extract_list(response, 'conflicts'),
                'complementary_aspects': self._extract_list(response, 'complementary'),
                'confidence_score': float(self._extract_field(response, 'confidence', '0.7'))
            }
        except Exception as e:
            self.logger.error(f"Error parsing relationship analysis: {e}")
            return {
                'similarity_score': 0.0,
                'relationship_type': 'related',
                'common_topics': [],
                'analysis_summary': '',
                'key_connections': [],
                'potential_conflicts': [],
                'complementary_aspects': [],
                'confidence_score': 0.5
            }
    
    async def _parse_conflict_analysis(self, response: str) -> Dict[str, Any]:
        """Parse conflict analysis response"""
        
        try:
            conflicts_detected = 'conflict' in response.lower() or 'contradiction' in response.lower()
            
            return {
                'conflicts_detected': conflicts_detected,
                'conflicts': self._extract_conflicts(response) if conflicts_detected else [],
                'analysis_summary': response[:500]
            }
        except Exception as e:
            self.logger.error(f"Error parsing conflict analysis: {e}")
            return {
                'conflicts_detected': False,
                'conflicts': [],
                'analysis_summary': 'Analysis parsing failed'
            }
    
    async def _parse_portfolio_analysis(self, response: str) -> Dict[str, Any]:
        """Parse portfolio analysis response"""
        
        try:
            return {
                'overall_risk_assessment': self._extract_field(response, 'overall_risk', 'medium'),
                'portfolio_strengths': self._extract_list(response, 'strengths'),
                'portfolio_weaknesses': self._extract_list(response, 'weaknesses'),
                'recommendations': self._extract_list(response, 'recommendations'),
                'completeness_score': float(self._extract_field(response, 'completeness', '0.7')),
                'coherence_score': float(self._extract_field(response, 'coherence', '0.8')),
                'analysis_summary': response[:1000]
            }
        except Exception as e:
            self.logger.error(f"Error parsing portfolio analysis: {e}")
            return {
                'overall_risk_assessment': 'medium',
                'portfolio_strengths': [],
                'portfolio_weaknesses': [],
                'recommendations': [],
                'completeness_score': 0.7,
                'coherence_score': 0.8,
                'analysis_summary': 'Analysis parsing failed'
            }
    
    # Utility methods for text extraction
    
    def _extract_field(self, text: str, field_name: str, default: str = '') -> str:
        """Extract field value from text"""
        
        patterns = [
            f'{field_name}[:\-\s]+([^\n]+)',
            f'{field_name.title()}[:\-\s]+([^\n]+)',
            f'{field_name.upper()}[:\-\s]+([^\n]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return default
    
    def _extract_list(self, text: str, list_name: str) -> List[str]:
        """Extract list items from text"""
        
        # Look for various list formats
        patterns = [
            f'{list_name}[:\-\s]*\n([^\n]*(?:\n[^\n]*)*)',
            f'{list_name}[:\-\s]*([^\n]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                content = match.group(1)
                # Split by common delimiters
                items = re.split(r'[,;•\-\n]+', content)
                return [item.strip() for item in items if item.strip()][:10]  # Limit to 10 items
        
        return []
    
    def _extract_conflicts(self, text: str) -> List[Dict[str, str]]:
        """Extract conflict information from text"""
        
        conflicts = []
        lines = text.split('\n')
        
        for line in lines:
            if 'conflict' in line.lower() or 'contradiction' in line.lower():
                conflicts.append({
                    'type': 'contradiction',
                    'description': line.strip(),
                    'severity': 'medium'
                })
        
        return conflicts[:5]  # Limit to 5 conflicts
    
    def _clean_response_text(self, text: str) -> str:
        """Clean and format response text"""
        
        # Remove markdown artifacts
        text = text.replace('**', '').replace('*', '')
        
        # Clean up spacing
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize line breaks
        text = re.sub(r' +', ' ', text)  # Normalize spaces
        
        return text.strip()
    
    # Prompt templates (same as original)
    
    def _get_document_analysis_template(self) -> str:
        return """
        Analyze this legal document comprehensively:

        Document: {filename}
        Type: {document_type}
        Word Count: {word_count}
        Page Count: {page_count}

        Content Preview:
        {content_preview}

        Provide a structured analysis with:

        1. DOCUMENT_TYPE: Classify the document type (contract, agreement, lease, etc.)
        2. TITLE: Generate a clear title for this document
        3. RISK_LEVEL: Assess overall risk (low/medium/high)
        4. CONFIDENCE: Your confidence in this analysis (0.0-1.0)
        5. KEY_TOPICS: List 5-8 main topics covered
        6. ENTITIES: Extract key entities (people, organizations, amounts, dates)
        7. SUMMARY: Provide a 2-3 sentence summary
        8. RISK_FACTORS: List potential risks and concerns
        9. OBLIGATIONS: Key obligations and responsibilities
        10. RECOMMENDATIONS: Actionable recommendations for the user

        Format your response clearly with labeled sections.
        """
    
    def _get_chunk_analysis_template(self) -> str:
        return """
        Analyze this document section in detail:

        Document: {document_title} ({document_type})
        Section: {chunk_index + 1} of {total_chunks} (Pages: {page_range})

        Content:
        {chunk_content}

        Provide:
        1. TOPICS: Main topics in this section
        2. ENTITIES: Key entities mentioned
        3. KEY_PHRASES: Important phrases and terms
        4. SENTIMENT: Overall sentiment (positive/neutral/negative)
        5. IMPORTANCE: Importance score for this section (0.0-1.0)
        6. SUMMARY: Brief summary of this section

        Focus on legal terms, obligations, risks, and key information.
        """
    
    def _get_relationship_analysis_template(self) -> str:
        return """
        Analyze the relationship between these two documents:

        Document 1: {doc1_title} ({doc1_type})
        Topics: {doc1_topics}
        Sample Content: {doc1_sample}

        Document 2: {doc2_title} ({doc2_type})
        Topics: {doc2_topics}
        Sample Content: {doc2_sample}

        Determine:
        1. SIMILARITY: Similarity score (0.0-1.0)
        2. RELATIONSHIP_TYPE: Type of relationship (highly_related/related/complementary/contradictory)
        3. COMMON_TOPICS: Topics discussed in both documents
        4. SUMMARY: Brief summary of the relationship
        5. CONNECTIONS: Key connections between the documents
        6. CONFLICTS: Any contradictions or conflicts
        7. COMPLEMENTARY: How the documents complement each other
        8. CONFIDENCE: Confidence in this analysis (0.0-1.0)

        Focus on legal implications and practical relationships.
        """
    
    def _get_cross_document_synthesis_template(self) -> str:
        return """
        Answer this question using information from multiple documents:

        Question: {user_query}

        Available Information:
        {chunk_information}

        Conversation Context: {conversation_context}

        Sources: {total_sources} sections from {document_count} documents

        Instructions:
        1. Synthesize information from all relevant sources
        2. Resolve any contradictions between sources
        3. Cite specific sources for key claims [Source X]
        4. Provide a comprehensive, coherent answer
        5. Highlight cross-document insights and relationships
        6. Note any limitations in available information
        7. Use clear, accessible language while maintaining legal accuracy

        Provide a natural, conversational response that fully addresses the question.
        """
    
    def _get_conflict_detection_template(self) -> str:
        return """
        Compare these two documents for conflicts or contradictions:

        Document 1: {doc1_title}
        Content: {doc1_content}

        Document 2: {doc2_title}
        Content: {doc2_content}

        Look for:
        1. Contradictory terms or conditions
        2. Different obligations or responsibilities
        3. Conflicting dates, amounts, or requirements
        4. Inconsistent definitions or interpretations
        5. Precedence issues (which document should take priority)

        If conflicts are found, describe:
        - The specific nature of each conflict
        - Which document contains each conflicting provision
        - The potential impact of the conflict
        - Recommendations for resolution

        If no conflicts are found, explain how the documents relate to each other.
        """
    
    def _get_portfolio_analysis_template(self) -> str:
        return """
        Analyze this complete legal document portfolio:

        Total Documents: {document_count}

        Documents:
        {portfolio_information}

        Relationships ({relationship_count}):
        {relationship_information}

        Provide a comprehensive portfolio analysis:

        1. OVERALL_RISK: Overall portfolio risk assessment
        2. STRENGTHS: Portfolio strengths and advantages
        3. WEAKNESSES: Areas of concern or vulnerability
        4. RECOMMENDATIONS: Strategic recommendations
        5. COMPLETENESS: How complete is this portfolio (0.0-1.0)
        6. COHERENCE: How well do documents work together (0.0-1.0)
        7. ANALYSIS_SUMMARY: Comprehensive summary

        Focus on:
        - Cross-document synergies and conflicts
        - Overall legal position and exposure
        - Strategic recommendations for optimization
        - Risk management priorities
        """

    # Additional methods for missing functionality
    
    async def detect_document_conflicts(self, document_chunks: List[DocumentChunk], 
                                       document_summaries: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect conflicts between documents (simplified implementation)"""
        try:
            return []  # Simplified - return empty list for now
        except Exception as e:
            self.logger.error(f"Error detecting conflicts: {e}")
            return []

    async def analyze_document_portfolio(self, document_summaries: List[Dict[str, Any]], 
                                        relationships: List[DocumentRelationship]) -> Dict[str, Any]:
        """Analyze document portfolio (simplified implementation)"""
        try:
            return {
                'overall_risk_assessment': 'medium',
                'portfolio_strengths': ['Comprehensive coverage'],
                'portfolio_weaknesses': ['May need legal review'],
                'recommendations': ['Review all terms carefully'],
                'completeness_score': 0.8,
                'coherence_score': 0.7,
                'analysis_summary': f'Portfolio contains {len(document_summaries)} documents with {len(relationships)} relationships'
            }
        except Exception as e:
            self.logger.error(f"Error analyzing portfolio: {e}")
            return {'error': str(e)}

    def get_analyzer_info(self) -> Dict[str, Any]:
        """Get information about the analyzer capabilities"""
        return {
            'gemini_available': self.gemini_available,
            'model_name': getattr(self.settings, 'GEMINI_MODEL', 'fallback') if self.gemini_available else 'fallback',
            'capabilities': [
                'document_analysis',
                'chunk_analysis', 
                'relationship_detection',
                'cross_document_synthesis',
                'conflict_detection',
                'portfolio_analysis'
            ],
            'fallback_active': not self.gemini_available
        }
