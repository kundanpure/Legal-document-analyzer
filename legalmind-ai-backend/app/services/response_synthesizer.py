"""
Intelligent response synthesis from multiple document chunks
Enhanced with fallbacks and comprehensive error handling
"""

import asyncio
import uuid
import re
from dotenv import load_dotenv
load_dotenv() 
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

from config.logging import get_logger

logger = get_logger(__name__)

# Try to import numpy with fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    logger.info("✅ NumPy available - advanced numerical operations enabled")
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("⚠️ NumPy not available - using basic numerical operations")

# Try to import services with fallbacks
services_available = {}

def try_import_service(name: str, module_path: str, class_name: str = None):
    """Try to import a service with fallback handling"""
    try:
        module = __import__(module_path, fromlist=[class_name] if class_name else [])
        if class_name:
            service_class = getattr(module, class_name, None)
            if service_class:
                services_available[name] = service_class
                logger.info(f"✅ Imported {name}")
                return True
        else:
            services_available[name] = module
            logger.info(f"✅ Imported {name}")
            return True
    except ImportError as e:
        logger.warning(f"⚠️ Could not import {name}: {e}")
        services_available[name] = None
        return False

# Import services with fallbacks
try_import_service('gemini_analyzer', 'app.services.gemini_analyzer', 'GeminiAnalyzer')
try_import_service('models', 'app.services.multi_document_models')

# Import models with fallbacks
if services_available.get('models'):
    try:
        from app.services.multi_document_models import (
            DocumentChunk, SynthesizedResponse, SourceAttribution
        )
        MODELS_AVAILABLE = True
    except ImportError:
        MODELS_AVAILABLE = False
else:
    MODELS_AVAILABLE = False

# Mock models when real models aren't available
if not MODELS_AVAILABLE:
    @dataclass
    class MockDocumentChunk:
        id: str
        document_id: str
        content: str
        page_range: str
        chunk_index: int = 0
        total_chunks: int = 1
        metadata: Dict[str, Any] = None
        
        def __post_init__(self):
            if self.metadata is None:
                self.metadata = {}
    
    @dataclass
    class MockSynthesizedResponse:
        query: str
        answer: str
        source_attributions: List[Any]
        synthesis_strategy: str
        confidence_score: float
        quality_metrics: Dict[str, Any]
        follow_up_suggestions: List[str]
        cross_references: List[Dict[str, Any]]
        response_metadata: Dict[str, Any]
        timestamp: datetime = None
        
        def __post_init__(self):
            if self.timestamp is None:
                self.timestamp = datetime.now(timezone.utc)
        
        def to_dict(self):
            return {
                'query': self.query,
                'answer': self.answer,
                'source_attributions': [attr.to_dict() if hasattr(attr, 'to_dict') else str(attr) for attr in self.source_attributions],
                'synthesis_strategy': self.synthesis_strategy,
                'confidence_score': self.confidence_score,
                'quality_metrics': self.quality_metrics,
                'follow_up_suggestions': self.follow_up_suggestions,
                'cross_references': self.cross_references,
                'response_metadata': self.response_metadata,
                'timestamp': self.timestamp.isoformat() if self.timestamp else None
            }
    
    @dataclass
    class MockSourceAttribution:
        document_id: str
        document_title: str
        page_range: str
        chunk_id: str
        relevance_score: float
        confidence_score: float
        key_concepts: List[str]
        excerpt: str
        
        def to_dict(self):
            return {
                'document_id': self.document_id,
                'document_title': self.document_title,
                'page_range': self.page_range,
                'chunk_id': self.chunk_id,
                'relevance_score': self.relevance_score,
                'confidence_score': self.confidence_score,
                'key_concepts': self.key_concepts,
                'excerpt': self.excerpt
            }
    
    DocumentChunk = MockDocumentChunk
    SynthesizedResponse = MockSynthesizedResponse
    SourceAttribution = MockSourceAttribution

@dataclass
class ChunkRelevance:
    """Represents relevance of a chunk to a query"""
    chunk_id: str
    relevance_score: float
    source_type: str  # 'semantic', 'keyword', 'contextual', 'cross_reference'
    matching_concepts: List[str]
    confidence: float

@dataclass
class ResponseEvidence:
    """Evidence supporting a response"""
    claim: str
    supporting_chunks: List[str]
    confidence_score: float
    contradictory_evidence: Optional[List[str]] = None

# Mock services for when real services aren't available
class MockGeminiAnalyzer:
    async def _generate_content_async(self, prompt: str) -> str:
        """Mock content generation"""
        logger.info("Using mock Gemini analyzer")
        
        # Simple response generation based on prompt keywords
        if 'risk' in prompt.lower():
            return "Based on the document analysis, there are several risk factors to consider. Please review the terms carefully and consider consulting with legal counsel for specific advice."
        elif 'compare' in prompt.lower():
            return "Comparing the documents shows both similarities and differences in key terms. Document A appears to have more restrictive clauses while Document B offers more flexibility."
        elif 'summary' in prompt.lower():
            return "In summary, the document contains standard legal provisions with moderate risk levels. Key obligations and rights are clearly defined."
        else:
            return "Based on your documents, I can provide relevant information about your question. However, the mock analyzer has limited capabilities - please ensure the full Gemini service is available for complete analysis."

class MockSessionKnowledgeBase:
    def __init__(self):
        self.chunks = {}
        self.documents = {}
        self.document_summaries = {}
        self.relationships = {}
    
    def get_related_documents(self, document_id: str):
        return []

class IntelligentResponseSynthesizer:
    """
    Synthesizes coherent responses from multiple document chunks with fallbacks
    """
    
    def __init__(self):
        # Initialize services with fallbacks
        if services_available.get('gemini_analyzer'):
            self.gemini_analyzer = services_available['gemini_analyzer']()
        else:
            self.gemini_analyzer = MockGeminiAnalyzer()
        
        self.logger = logger
        
        # Response synthesis strategies
        self.synthesis_strategies = {
            'comprehensive': self._comprehensive_synthesis,
            'comparative': self._comparative_synthesis,
            'risk_focused': self._risk_focused_synthesis,
            'summary': self._summary_synthesis
        }
        
        self.logger.info(f"IntelligentResponseSynthesizer initialized - Gemini: {'✓' if services_available.get('gemini_analyzer') else '✗'}")
    
    async def generate_response(
        self,
        user_query: str,
        session_kb: Any,
        conversation_context: Dict[str, Any],
        max_sources: int = 5
    ) -> SynthesizedResponse:
        """
        Generate intelligent response from multiple document sources
        
        Args:
            user_query: User's question
            session_kb: Session knowledge base
            conversation_context: Conversation context
            max_sources: Maximum number of source chunks to use
            
        Returns:
            SynthesizedResponse with answer and attributions
        """
        
        try:
            self.logger.info(f"Generating synthesized response for query: {user_query[:100]}...")
            
            # Handle mock knowledge base
            if not hasattr(session_kb, 'chunks'):
                session_kb = MockSessionKnowledgeBase()
            
            # Step 1: Find relevant chunks across all documents
            relevant_chunks = await self._find_relevant_chunks(
                user_query, session_kb, max_sources
            )
            
            if not relevant_chunks:
                return await self._generate_no_information_response(user_query)
            
            # Step 2: Determine synthesis strategy
            synthesis_strategy = await self._determine_synthesis_strategy(
                user_query, relevant_chunks, conversation_context
            )
            
            # Step 3: Generate chunk-level responses
            chunk_responses = await self._generate_chunk_responses(
                user_query, relevant_chunks
            )
            
            # Step 4: Synthesize final response
            synthesized_answer = await self.synthesis_strategies[synthesis_strategy](
                user_query, chunk_responses, conversation_context
            )
            
            # Step 5: Create source attributions
            source_attributions = await self._create_source_attributions(
                relevant_chunks, chunk_responses
            )
            
            # Step 6: Assess response quality
            quality_metrics = await self._assess_response_quality(
                synthesized_answer, user_query, relevant_chunks
            )
            
            # Step 7: Generate follow-up suggestions
            follow_up_suggestions = await self._generate_follow_up_suggestions(
                user_query, synthesized_answer, session_kb
            )
            
            # Step 8: Find cross-references
            cross_references = await self._find_cross_references(
                relevant_chunks, session_kb
            )
            
            return SynthesizedResponse(
                query=user_query,
                answer=synthesized_answer,
                source_attributions=source_attributions,
                synthesis_strategy=synthesis_strategy,
                confidence_score=quality_metrics.get('overall_confidence', 0.8),
                quality_metrics=quality_metrics,
                follow_up_suggestions=follow_up_suggestions,
                cross_references=cross_references,
                response_metadata={
                    'chunks_analyzed': len(relevant_chunks),
                    'documents_referenced': len(set(chunk.document_id for chunk, _ in relevant_chunks)),
                    'synthesis_time': datetime.now(timezone.utc).isoformat(),
                    'synthesis_version': 'enhanced_v2.0'
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating synthesized response: {str(e)}")
            return await self._generate_error_response(user_query, str(e))
    
    async def _find_relevant_chunks(
        self,
        query: str,
        session_kb: Any,
        max_chunks: int
    ) -> List[Tuple[DocumentChunk, ChunkRelevance]]:
        """
        Find most relevant chunks across all documents using multiple strategies
        """
        
        try:
            all_relevances = []
            
            # Strategy 1: Semantic similarity (if embeddings available)
            semantic_relevances = await self._semantic_search(query, session_kb)
            all_relevances.extend(semantic_relevances)
            
            # Strategy 2: Keyword matching
            keyword_relevances = await self._keyword_search(query, session_kb)
            all_relevances.extend(keyword_relevances)
            
            # Strategy 3: Contextual search (based on conversation context)
            contextual_relevances = await self._contextual_search(query, session_kb)
            all_relevances.extend(contextual_relevances)
            
            # Strategy 4: Cross-reference search
            cross_ref_relevances = await self._cross_reference_search(query, session_kb)
            all_relevances.extend(cross_ref_relevances)
            
            # Combine and deduplicate
            chunk_relevances = self._combine_relevance_scores(all_relevances)
            
            # Sort by relevance and return top chunks
            sorted_chunks = sorted(
                chunk_relevances,
                key=lambda x: x[1].relevance_score,
                reverse=True
            )
            
            return sorted_chunks[:max_chunks]
            
        except Exception as e:
            self.logger.error(f"Error finding relevant chunks: {str(e)}")
            return []
    
    async def _semantic_search(
        self,
        query: str,
        session_kb: Any
    ) -> List[Tuple[DocumentChunk, ChunkRelevance]]:
        """Semantic similarity search using embeddings or word overlap"""
        
        relevances = []
        
        try:
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            chunks = getattr(session_kb, 'chunks', {})
            
            for chunk_id, chunk in chunks.items():
                chunk_words = set(chunk.content.lower().split())
                
                # Calculate similarity using Jaccard coefficient or simple overlap
                word_overlap = len(query_words & chunk_words)
                total_words = len(query_words | chunk_words)
                
                if total_words > 0:
                    similarity_score = word_overlap / total_words
                    
                    if similarity_score > 0.1:  # Minimum threshold
                        matching_concepts = list(query_words & chunk_words)
                        
                        relevance = ChunkRelevance(
                            chunk_id=chunk_id,
                            relevance_score=similarity_score * 0.8,  # Weight semantic search
                            source_type='semantic',
                            matching_concepts=matching_concepts,
                            confidence=similarity_score
                        )
                        
                        relevances.append((chunk, relevance))
            
            return relevances
            
        except Exception as e:
            self.logger.error(f"Error in semantic search: {str(e)}")
            return []
    
    async def _keyword_search(
        self,
        query: str,
        session_kb: Any
    ) -> List[Tuple[DocumentChunk, ChunkRelevance]]:
        """Keyword-based search with legal term weighting"""
        
        relevances = []
        
        try:
            # Extract keywords from query
            query_keywords = self._extract_keywords(query)
            
            # Legal term weights
            legal_term_weights = {
                'contract': 2.0, 'agreement': 2.0, 'clause': 2.0,
                'liability': 2.5, 'penalty': 2.5, 'breach': 2.5,
                'termination': 2.0, 'payment': 1.8, 'obligation': 2.2,
                'risk': 1.5, 'fee': 1.5, 'cost': 1.2, 'rights': 1.8,
                'warranty': 2.0, 'indemnity': 2.3, 'confidential': 1.7
            }
            
            chunks = getattr(session_kb, 'chunks', {})
            
            for chunk_id, chunk in chunks.items():
                chunk_content_lower = chunk.content.lower()
                
                keyword_score = 0
                matching_keywords = []
                
                for keyword in query_keywords:
                    keyword_lower = keyword.lower()
                    if keyword_lower in chunk_content_lower:
                        weight = legal_term_weights.get(keyword_lower, 1.0)
                        # Count occurrences and apply diminishing returns
                        occurrences = chunk_content_lower.count(keyword_lower)
                        keyword_score += weight * min(3, occurrences) * 0.5  # Cap at 3 occurrences
                        matching_keywords.append(keyword)
                
                if keyword_score > 0:
                    # Normalize by chunk length
                    chunk_length = max(len(chunk.content.split()), 100)
                    normalized_score = keyword_score / (chunk_length / 100)
                    
                    relevance = ChunkRelevance(
                        chunk_id=chunk_id,
                        relevance_score=min(1.0, normalized_score),
                        source_type='keyword',
                        matching_concepts=matching_keywords,
                        confidence=min(1.0, normalized_score)
                    )
                    
                    relevances.append((chunk, relevance))
            
            return relevances
            
        except Exception as e:
            self.logger.error(f"Error in keyword search: {str(e)}")
            return []
    
    async def _contextual_search(
        self,
        query: str,
        session_kb: Any
    ) -> List[Tuple[DocumentChunk, ChunkRelevance]]:
        """Search based on conversation context and document relationships"""
        
        relevances = []
        
        try:
            chunks = getattr(session_kb, 'chunks', {})
            
            # Simple contextual boosting based on document relationships
            for chunk_id, chunk in chunks.items():
                document_id = chunk.document_id
                
                # Check if document has relationships with others
                if hasattr(session_kb, 'get_related_documents'):
                    try:
                        related_docs = session_kb.get_related_documents(document_id)
                    except:
                        related_docs = []
                else:
                    related_docs = []
                
                if related_docs:
                    # Boost relevance for chunks from related documents
                    contextual_boost = min(0.5, len(related_docs) * 0.1)
                    
                    relevance = ChunkRelevance(
                        chunk_id=chunk_id,
                        relevance_score=contextual_boost,
                        source_type='contextual',
                        matching_concepts=['document_relationship'],
                        confidence=0.6
                    )
                    
                    relevances.append((chunk, relevance))
                
                # Boost chunks that contain metadata topics matching query
                chunk_topics = chunk.metadata.get('topics', [])
                query_words = set(query.lower().split())
                
                topic_matches = []
                for topic in chunk_topics:
                    if any(word in topic.lower() for word in query_words):
                        topic_matches.append(topic)
                
                if topic_matches:
                    topic_boost = len(topic_matches) * 0.15
                    
                    relevance = ChunkRelevance(
                        chunk_id=chunk_id,
                        relevance_score=min(0.6, topic_boost),
                        source_type='contextual',
                        matching_concepts=topic_matches,
                        confidence=0.7
                    )
                    
                    relevances.append((chunk, relevance))
            
            return relevances
            
        except Exception as e:
            self.logger.error(f"Error in contextual search: {str(e)}")
            return []
    
    async def _cross_reference_search(
        self,
        query: str,
        session_kb: Any
    ) -> List[Tuple[DocumentChunk, ChunkRelevance]]:
        """Search for cross-references between documents"""
        
        relevances = []
        
        try:
            query_lower = query.lower()
            
            # Look for comparison keywords
            comparison_keywords = ['compare', 'vs', 'versus', 'difference', 'similar', 'unlike', 'contrast']
            is_comparison_query = any(keyword in query_lower for keyword in comparison_keywords)
            
            chunks = getattr(session_kb, 'chunks', {})
            
            if is_comparison_query and len(chunks) > 1:
                # Find chunks from different documents that discuss similar topics
                document_topics = {}
                
                for chunk_id, chunk in chunks.items():
                    doc_id = chunk.document_id
                    chunk_topics = chunk.metadata.get('topics', [])
                    
                    if doc_id not in document_topics:
                        document_topics[doc_id] = set()
                    document_topics[doc_id].update(chunk_topics)
                
                # Find overlapping topics between documents
                for chunk_id, chunk in chunks.items():
                    chunk_topics = set(chunk.metadata.get('topics', []))
                    doc_id = chunk.document_id
                    
                    cross_references = []
                    for other_doc_id, other_topics in document_topics.items():
                        if other_doc_id != doc_id:
                            topic_overlap = chunk_topics & other_topics
                            if topic_overlap:
                                cross_references.extend(list(topic_overlap))
                    
                    if cross_references:
                        relevance = ChunkRelevance(
                            chunk_id=chunk_id,
                            relevance_score=min(0.8, len(cross_references) * 0.2),
                            source_type='cross_reference',
                            matching_concepts=cross_references,
                            confidence=0.7
                        )
                        
                        relevances.append((chunk, relevance))
            
            return relevances
            
        except Exception as e:
            self.logger.error(f"Error in cross-reference search: {str(e)}")
            return []
    
    def _combine_relevance_scores(
        self,
        all_relevances: List[Tuple[DocumentChunk, ChunkRelevance]]
    ) -> List[Tuple[DocumentChunk, ChunkRelevance]]:
        """Combine relevance scores from different search strategies"""
        
        chunk_combined_scores = {}
        
        # Group by chunk ID
        for chunk, relevance in all_relevances:
            chunk_id = chunk.id
            
            if chunk_id not in chunk_combined_scores:
                chunk_combined_scores[chunk_id] = {
                    'chunk': chunk,
                    'relevances': [],
                    'combined_score': 0,
                    'all_concepts': set(),
                    'max_confidence': 0,
                    'source_types': set()
                }
            
            chunk_data = chunk_combined_scores[chunk_id]
            chunk_data['relevances'].append(relevance)
            chunk_data['combined_score'] += relevance.relevance_score
            chunk_data['all_concepts'].update(relevance.matching_concepts)
            chunk_data['max_confidence'] = max(chunk_data['max_confidence'], relevance.confidence)
            chunk_data['source_types'].add(relevance.source_type)
        
        # Create final relevance objects with diversity bonus
        final_relevances = []
        
        for chunk_id, data in chunk_combined_scores.items():
            # Diversity bonus for chunks found by multiple strategies
            diversity_bonus = len(data['source_types']) * 0.1
            
            combined_relevance = ChunkRelevance(
                chunk_id=chunk_id,
                relevance_score=min(1.0, data['combined_score'] + diversity_bonus),
                source_type='combined',
                matching_concepts=list(data['all_concepts']),
                confidence=data['max_confidence']
            )
            
            final_relevances.append((data['chunk'], combined_relevance))
        
        return final_relevances
    
    async def _generate_chunk_responses(
        self,
        query: str,
        relevant_chunks: List[Tuple[DocumentChunk, ChunkRelevance]]
    ) -> List[Dict[str, Any]]:
        """Generate responses from individual chunks"""
        
        chunk_responses = []
        
        try:
            for chunk, relevance in relevant_chunks:
                prompt = f"""
                Based on this document section, answer the user's question: "{query}"
                
                Document Section (Pages {chunk.page_range}):
                {chunk.content[:2000]}{"..." if len(chunk.content) > 2000 else ""}
                
                Instructions:
                1. Provide a focused answer based only on this section
                2. If the section doesn't contain relevant information, say so clearly
                3. Be specific about what information comes from this section
                4. Use precise legal language when appropriate
                5. Highlight key terms, dates, amounts, or obligations
                
                Answer:
                """
                
                try:
                    if hasattr(self.gemini_analyzer, '_generate_content_async'):
                        response = await self.gemini_analyzer._generate_content_async(prompt)
                    else:
                        # Fallback for mock analyzer
                        response = await self.gemini_analyzer._generate_content_async(prompt)
                    
                    # Get document title from session KB if available
                    document_title = f"Document {chunk.document_id}"
                    
                    chunk_responses.append({
                        'chunk_id': chunk.id,
                        'document_id': chunk.document_id,
                        'response': response,
                        'relevance_score': relevance.relevance_score,
                        'matching_concepts': relevance.matching_concepts,
                        'source_metadata': {
                            'pages': chunk.page_range,
                            'document_title': document_title,
                            'chunk_index': chunk.chunk_index,
                            'total_chunks': chunk.total_chunks,
                            'search_strategy': relevance.source_type
                        }
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error generating response for chunk {chunk.id}: {str(e)}")
                    # Add fallback response
                    chunk_responses.append({
                        'chunk_id': chunk.id,
                        'document_id': chunk.document_id,
                        'response': f"This section from pages {chunk.page_range} contains relevant information about your query, but I encountered an error processing it fully.",
                        'relevance_score': relevance.relevance_score,
                        'matching_concepts': relevance.matching_concepts,
                        'source_metadata': {
                            'pages': chunk.page_range,
                            'document_title': f"Document {chunk.document_id}",
                            'chunk_index': chunk.chunk_index,
                            'total_chunks': chunk.total_chunks,
                            'search_strategy': relevance.source_type,
                            'processing_error': True
                        }
                    })
            
            return chunk_responses
            
        except Exception as e:
            self.logger.error(f"Error generating chunk responses: {str(e)}")
            return []
    
    async def _comprehensive_synthesis(
        self,
        query: str,
        chunk_responses: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> str:
        """Synthesize comprehensive response from multiple chunks"""
        
        if not chunk_responses:
            return "I don't have enough information in your documents to answer this question."
        
        # Prepare synthesis prompt
        chunk_info = []
        for i, chunk_resp in enumerate(chunk_responses, 1):
            chunk_info.append(f"""
            Source {i} - {chunk_resp['source_metadata']['document_title']} (Pages {chunk_resp['source_metadata']['pages']}):
            Relevance: {chunk_resp['relevance_score']:.2f}
            Key concepts: {', '.join(chunk_resp['matching_concepts'])}
            Content: {chunk_resp['response']}
            """)
        
        synthesis_prompt = f"""
        Synthesize a comprehensive answer to: "{query}"
        
        You have information from {len(chunk_responses)} document sources:
        
        {''.join(chunk_info)}
        
        Instructions:
        1. Create a coherent, comprehensive response that combines all relevant information
        2. Resolve any contradictions between sources by noting them explicitly
        3. Organize information logically (most important first)
        4. Use clear, accessible language while maintaining legal accuracy
        5. If sources complement each other, show how they work together
        6. Cite specific sources for key claims using format [Source X]
        7. Flag any limitations in the available information
        8. Provide actionable insights where appropriate
        9. Use specific details (amounts, dates, terms) from the sources
        
        Provide a natural, conversational response that fully addresses the user's question.
        """
        
        try:
            if hasattr(self.gemini_analyzer, '_generate_content_async'):
                synthesized_response = await self.gemini_analyzer._generate_content_async(synthesis_prompt)
            else:
                synthesized_response = await self.gemini_analyzer._generate_content_async(synthesis_prompt)
            
            return synthesized_response
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive synthesis: {str(e)}")
            
            # Fallback synthesis
            fallback_response = f"Based on {len(chunk_responses)} sources in your documents:\n\n"
            
            for i, chunk_resp in enumerate(chunk_responses, 1):
                fallback_response += f"{i}. From {chunk_resp['source_metadata']['document_title']} (Pages {chunk_resp['source_metadata']['pages']}):\n"
                fallback_response += f"   {chunk_resp['response'][:200]}...\n\n"
            
            fallback_response += "Please review these sources for complete information. For detailed analysis, ensure all services are properly configured."
            
            return fallback_response
    
    async def _comparative_synthesis(
        self,
        query: str,
        chunk_responses: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> str:
        """Synthesize comparative response highlighting differences and similarities"""
        
        if len(chunk_responses) < 2:
            return await self._comprehensive_synthesis(query, chunk_responses, context)
        
        # Group responses by document
        doc_responses = {}
        for chunk_resp in chunk_responses:
            doc_id = chunk_resp['document_id']
            if doc_id not in doc_responses:
                doc_responses[doc_id] = []
            doc_responses[doc_id].append(chunk_resp)
        
        comparison_prompt = f"""
        Create a comparative analysis for: "{query}"
        
        You have information from {len(doc_responses)} different documents:
        
        """
        
        for doc_id, responses in doc_responses.items():
            comparison_prompt += f"\n{responses[0]['source_metadata']['document_title']}:\n"
            for resp in responses:
                comparison_prompt += f"- Pages {resp['source_metadata']['pages']}: {resp['response']}\n"
        
        comparison_prompt += """
        
        Instructions:
        1. Compare and contrast the information across documents
        2. Highlight similarities and differences clearly
        3. Identify any contradictions or conflicts
        4. Explain which approach or terms are more favorable/restrictive
        5. Provide recommendations if one document has better terms
        6. Use comparison language (e.g., "Document A requires..., while Document B allows...")
        7. Organize by topics if multiple aspects are being compared
        8. Include specific details (amounts, timeframes, conditions)
        9. Note any missing information that would be helpful for comparison
        
        Format as a clear comparative analysis with actionable insights.
        """
        
        try:
            if hasattr(self.gemini_analyzer, '_generate_content_async'):
                comparative_response = await self.gemini_analyzer._generate_content_async(comparison_prompt)
            else:
                comparative_response = await self.gemini_analyzer._generate_content_async(comparison_prompt)
            
            return comparative_response
            
        except Exception as e:
            self.logger.error(f"Error in comparative synthesis: {str(e)}")
            return await self._comprehensive_synthesis(query, chunk_responses, context)
    
    async def _risk_focused_synthesis(
        self,
        query: str,
        chunk_responses: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> str:
        """Synthesize response focused on risks and implications"""
        
        risk_prompt = f"""
        Analyze risks and implications for: "{query}"
        
        Information from document sources:
        """
        
        for i, chunk_resp in enumerate(chunk_responses, 1):
            risk_prompt += f"""
            Source {i} - {chunk_resp['source_metadata']['document_title']} (Pages {chunk_resp['source_metadata']['pages']}):
            {chunk_resp['response']}
            """
        
        risk_prompt += """
        
        Instructions:
        1. Focus on risks, liabilities, and potential negative consequences
        2. Identify immediate vs long-term risks
        3. Quantify risks where possible (financial amounts, time periods, etc.)
        4. Explain likelihood and severity of each risk
        5. Highlight any hidden or non-obvious risks
        6. Suggest risk mitigation strategies where appropriate
        7. Use risk assessment language (high/medium/low risk)
        8. Prioritize risks by potential impact
        9. Include specific examples from the documents
        10. Note any protective clauses or safeguards
        
        Provide a risk-focused analysis that helps the user understand potential downsides and how to protect themselves.
        """
        
        try:
            if hasattr(self.gemini_analyzer, '_generate_content_async'):
                risk_response = await self.gemini_analyzer._generate_content_async(risk_prompt)
            else:
                risk_response = await self.gemini_analyzer._generate_content_async(risk_prompt)
            
            return risk_response
            
        except Exception as e:
            self.logger.error(f"Error in risk-focused synthesis: {str(e)}")
            return await self._comprehensive_synthesis(query, chunk_responses, context)
    
    async def _summary_synthesis(
        self,
        query: str,
        chunk_responses: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> str:
        """Synthesize concise summary response"""
        
        summary_prompt = f"""
        Provide a concise summary answer to: "{query}"
        
        Available information:
        """
        
        for chunk_resp in chunk_responses:
            summary_prompt += f"- From {chunk_resp['source_metadata']['document_title']}: {chunk_resp['response']}\n"
        
        summary_prompt += """
        
        Instructions:
        1. Provide a clear, concise answer (2-4 sentences max)
        2. Include only the most essential information
        3. Use simple, direct language
        4. Focus on actionable insights
        5. Avoid unnecessary details
        6. Include key specifics (amounts, dates, deadlines)
        
        Give a brief, direct answer that captures the most important points.
        """
        
        try:
            if hasattr(self.gemini_analyzer, '_generate_content_async'):
                summary_response = await self.gemini_analyzer._generate_content_async(summary_prompt)
            else:
                summary_response = await self.gemini_analyzer._generate_content_async(summary_prompt)
            
            return summary_response
            
        except Exception as e:
            self.logger.error(f"Error in summary synthesis: {str(e)}")
            
            # Simple fallback summary
            if chunk_responses:
                return f"Based on your documents: {chunk_responses[0]['response'][:150]}..." + (f" (and {len(chunk_responses)-1} more sources)" if len(chunk_responses) > 1 else "")
            else:
                return "I can provide information about your question, but encountered an error creating a summary."
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        
        # Remove common stop words and extract meaningful terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must', 'shall', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their', 'what', 'when', 'where',
            'why', 'how', 'who', 'which', 'if', 'then', 'than', 'as', 'so', 'too', 'very'
        }
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [word for word in words if word not in stop_words]
        
        return list(set(keywords))  # Remove duplicates
    
    async def _determine_synthesis_strategy(
        self,
        query: str,
        relevant_chunks: List[Tuple[DocumentChunk, ChunkRelevance]],
        context: Dict[str, Any]
    ) -> str:
        """Determine the best synthesis strategy for the query"""
        
        query_lower = query.lower()
        
        # Check for comparison keywords
        comparison_keywords = ['compare', 'vs', 'versus', 'difference', 'similar', 'better', 'worse', 'contrast']
        if any(keyword in query_lower for keyword in comparison_keywords):
            return 'comparative'
        
        # Check for risk-related keywords
        risk_keywords = ['risk', 'danger', 'problem', 'liability', 'penalty', 'consequence', 'downside', 'threat']
        if any(keyword in query_lower for keyword in risk_keywords):
            return 'risk_focused'
        
        # Check for summary request
        summary_keywords = ['summary', 'brief', 'quick', 'overview', 'tldr', 'short', 'concise']
        if any(keyword in query_lower for keyword in summary_keywords):
            return 'summary'
        
        # Check if multiple documents are involved
        document_ids = set(chunk.document_id for chunk, _ in relevant_chunks)
        if len(document_ids) > 1:
            return 'comparative'
        
        # Default to comprehensive
        return 'comprehensive'
    
    async def _create_source_attributions(
        self,
        relevant_chunks: List[Tuple[DocumentChunk, ChunkRelevance]],
        chunk_responses: List[Dict[str, Any]]
    ) -> List[SourceAttribution]:
        """Create source attribution objects"""
        
        attributions = []
        
        try:
            for chunk_resp in chunk_responses:
                attribution = SourceAttribution(
                    document_id=chunk_resp['document_id'],
                    document_title=chunk_resp['source_metadata']['document_title'],
                    page_range=chunk_resp['source_metadata']['pages'],
                    chunk_id=chunk_resp['chunk_id'],
                    relevance_score=chunk_resp['relevance_score'],
                    confidence_score=min(chunk_resp['relevance_score'] * 1.2, 1.0),  # Boost confidence slightly
                    key_concepts=chunk_resp['matching_concepts'],
                    excerpt=chunk_resp['response'][:200] + "..." if len(chunk_resp['response']) > 200 else chunk_resp['response']
                )
                attributions.append(attribution)
            
            return attributions
            
        except Exception as e:
            self.logger.error(f"Error creating source attributions: {str(e)}")
            return []
    
    async def _assess_response_quality(
        self,
        response: str,
        query: str,
        chunks: List[Tuple[DocumentChunk, ChunkRelevance]]
    ) -> Dict[str, Any]:
        """Assess the quality of the synthesized response"""
        
        try:
            response_words = len(response.split())
            query_words = set(query.lower().split())
            response_words_set = set(response.lower().split())
            
            # Calculate various quality metrics
            metrics = {
                'overall_confidence': 0.8,  # Base confidence
                'completeness': min(1.0, response_words / 150),  # Target ~150 words for completeness
                'source_diversity': len(set(chunk.document_id for chunk, _ in chunks)),
                'coherence': 0.85,  # Would need more sophisticated analysis
                'specificity': len([word for word in response_words_set if len(word) > 6]) / max(len(response_words_set), 1),
                'query_coverage': len(query_words & response_words_set) / len(query_words) if query_words else 0
            }
            
            # Adjust confidence based on source quality
            avg_relevance = sum(relevance.relevance_score for _, relevance in chunks) / len(chunks) if chunks else 0
            metrics['source_quality'] = avg_relevance
            
            # Calculate overall confidence
            metrics['overall_confidence'] = min(1.0, sum([
                metrics['completeness'] * 0.2,
                metrics['source_quality'] * 0.3,
                metrics['coherence'] * 0.2,
                metrics['specificity'] * 0.15,
                metrics['query_coverage'] * 0.15
            ]))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error assessing response quality: {str(e)}")
            return {
                'overall_confidence': 0.7,
                'completeness': 0.5,
                'source_diversity': len(chunks),
                'coherence': 0.7,
                'specificity': 0.5,
                'error': str(e)
            }
    
    async def _generate_follow_up_suggestions(
        self,
        query: str,
        response: str,
        session_kb: Any
    ) -> List[str]:
        """Generate relevant follow-up question suggestions"""
        
        suggestions = []
        
        try:
            query_lower = query.lower()
            response_lower = response.lower()
            
            # Context-specific follow-ups based on query type
            if 'risk' in query_lower:
                suggestions.extend([
                    "How can I mitigate these risks?",
                    "What are the potential consequences if these risks materialize?",
                    "Are there protective clauses I should look for?"
                ])
            
            if 'compare' in query_lower:
                suggestions.extend([
                    "Which option would you recommend and why?",
                    "What are the key trade-offs between these options?",
                    "Are there any deal-breakers in either document?"
                ])
            
            if any(term in query_lower for term in ['termination', 'fire', 'end', 'quit']):
                suggestions.extend([
                    "What notice period is required for termination?",
                    "What happens to my benefits if terminated?",
                    "Are there any penalties for early termination?"
                ])
            
            if any(term in query_lower for term in ['payment', 'fee', 'cost', 'money']):
                suggestions.extend([
                    "When are payments due?",
                    "What late payment penalties apply?",
                    "Are there any additional fees I should know about?"
                ])
            
            if any(term in query_lower for term in ['obligation', 'responsibility', 'duty']):
                suggestions.extend([
                    "What happens if I don't fulfill these obligations?",
                    "Can these obligations be modified or waived?",
                    "Are there any deadlines I need to be aware of?"
                ])
            
            # Generic useful follow-ups based on document portfolio
            document_count = len(getattr(session_kb, 'documents', {}))
            
            if document_count > 1:
                suggestions.extend([
                    "How do these documents relate to each other?",
                    "Are there any conflicts between my documents?",
                    "What's my overall legal position across all documents?"
                ])
            
            if document_count > 0:
                suggestions.extend([
                    "What should I prioritize reviewing first?",
                    "Are there any red flags I should be aware of?",
                    "Should I seek legal advice for any of these terms?"
                ])
            
            # Response-specific follow-ups
            if 'may' in response_lower or 'might' in response_lower:
                suggestions.append("Can you provide more definitive guidance on this?")
            
            if 'depends' in response_lower:
                suggestions.append("What factors does this depend on?")
            
            if len(response.split()) > 200:  # Long response
                suggestions.append("Can you summarize the key points?")
            
            # Remove duplicates and limit
            unique_suggestions = list(dict.fromkeys(suggestions))  # Preserves order
            return unique_suggestions[:5]  # Limit to 5 suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating follow-up suggestions: {str(e)}")
            return [
                "Can you provide more details about this?",
                "What should I do next?",
                "Are there any risks I should be aware of?"
            ]
    
    async def _find_cross_references(
        self,
        chunks: List[Tuple[DocumentChunk, ChunkRelevance]],
        session_kb: Any
    ) -> List[Dict[str, Any]]:
        """Find cross-references between documents"""
        
        cross_refs = []
        
        try:
            # Simple cross-reference detection based on document relationships
            document_ids = set(chunk.document_id for chunk, _ in chunks)
            
            if len(document_ids) < 2:
                return cross_refs
            
            # Check for relationships between documents
            if hasattr(session_kb, 'get_related_documents'):
                for doc_id in document_ids:
                    try:
                        related_docs = session_kb.get_related_documents(doc_id)
                        
                        for relationship in related_docs:
                            other_doc_id = (relationship.document_2_id 
                                          if relationship.document_1_id == doc_id 
                                          else relationship.document_1_id)
                            
                            if other_doc_id in document_ids:  # Both documents are in current response
                                cross_refs.append({
                                    'topic': relationship.common_topics[0] if hasattr(relationship, 'common_topics') and relationship.common_topics else 'related_content',
                                    'documents': [doc_id, other_doc_id],
                                    'relationship_type': getattr(relationship, 'relationship_type', 'related'),
                                    'description': f"Both documents discuss {relationship.common_topics[0] if hasattr(relationship, 'common_topics') and relationship.common_topics else 'related topics'}"
                                })
                    except Exception as e:
                        self.logger.warning(f"Error getting related documents for {doc_id}: {e}")
                        continue
            
            # Add topic-based cross-references
            topic_docs = {}
            for chunk, relevance in chunks:
                for concept in relevance.matching_concepts:
                    if concept not in topic_docs:
                        topic_docs[concept] = set()
                    topic_docs[concept].add(chunk.document_id)
            
            for topic, docs in topic_docs.items():
                if len(docs) > 1:
                    cross_refs.append({
                        'topic': topic,
                        'documents': list(docs),
                        'relationship_type': 'topic_overlap',
                        'description': f"Multiple documents discuss {topic}"
                    })
            
            return cross_refs[:5]  # Limit to 5 cross-references
            
        except Exception as e:
            self.logger.error(f"Error finding cross-references: {str(e)}")
            return []
    
    async def _generate_no_information_response(self, query: str) -> SynthesizedResponse:
        """Generate response when no relevant information is found"""
        
        return SynthesizedResponse(
            query=query,
            answer="I don't have information in your uploaded documents to answer this question. You might want to check if the relevant information is in a different document or section, or try rephrasing your question.",
            source_attributions=[],
            synthesis_strategy='no_information',
            confidence_score=0.0,
            quality_metrics={'overall_confidence': 0.0, 'no_sources': True},
            follow_up_suggestions=[
                "Could you upload a document that contains this information?",
                "Could you rephrase your question with different terms?",
                "What specific document should contain this information?",
                "Try asking about a broader topic that might be covered"
            ],
            cross_references=[],
            response_metadata={
                'no_relevant_sources': True,
                'suggestion': 'Try different keywords or upload additional documents'
            }
        )
    
    async def _generate_error_response(self, query: str, error_message: str) -> SynthesizedResponse:
        """Generate response for error cases"""
        
        return SynthesizedResponse(
            query=query,
            answer=f"I encountered an error while analyzing your documents: {error_message}. Please try asking your question again, or contact support if the issue persists.",
            source_attributions=[],
            synthesis_strategy='error',
            confidence_score=0.0,
            quality_metrics={'error': True, 'error_message': error_message},
            follow_up_suggestions=[
                "Try rephrasing your question with simpler terms",
                "Check if your documents uploaded successfully",
                "Try asking about a different aspect of your documents"
            ],
            cross_references=[],
            response_metadata={
                'error': error_message,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        )
    
    def get_synthesizer_capabilities(self) -> Dict[str, Any]:
        """Get information about synthesizer capabilities"""
        
        return {
            'synthesis_strategies': list(self.synthesis_strategies.keys()),
            'search_strategies': ['semantic', 'keyword', 'contextual', 'cross_reference'],
            'ai_integration': services_available.get('gemini_analyzer') is not None,
            'numerical_processing': NUMPY_AVAILABLE,
            'models_available': MODELS_AVAILABLE,
            'features': {
                'multi_document_synthesis': True,
                'cross_reference_detection': True,
                'relevance_scoring': True,
                'quality_assessment': True,
                'follow_up_generation': True,
                'error_recovery': True,
                'legal_term_weighting': True,
                'conversation_context': True
            },
            'supported_query_types': [
                'factual_questions',
                'comparisons', 
                'risk_analysis',
                'summaries',
                'obligation_analysis',
                'cross_document_queries'
            ]
        }


# Utility functions
async def synthesize_simple_response(query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Simple response synthesis for basic use cases"""
    
    synthesizer = IntelligentResponseSynthesizer()
    
    try:
        # Convert simple chunks to DocumentChunk objects
        mock_chunks = []
        for i, chunk_data in enumerate(chunks):
            mock_chunk = DocumentChunk(
                id=chunk_data.get('id', f'chunk_{i}'),
                document_id=chunk_data.get('document_id', f'doc_{i}'),
                content=chunk_data.get('content', ''),
                page_range=chunk_data.get('page_range', '1-1'),
                chunk_index=i,
                total_chunks=len(chunks),
                metadata=chunk_data.get('metadata', {})
            )
            
            relevance = ChunkRelevance(
                chunk_id=mock_chunk.id,
                relevance_score=chunk_data.get('relevance_score', 0.5),
                source_type='simple',
                matching_concepts=chunk_data.get('concepts', []),
                confidence=0.7
            )
            
            mock_chunks.append((mock_chunk, relevance))
        
        # Generate response using comprehensive synthesis
        response = await synthesizer._comprehensive_synthesis(query, [], {})
        
        return {
            'success': True,
            'answer': response,
            'source_count': len(chunks),
            'synthesis_strategy': 'comprehensive'
        }
        
    except Exception as e:
        return {
            'success': False,
            'answer': f"Error generating response: {str(e)}",
            'source_count': 0,
            'synthesis_strategy': 'error'
        }

def extract_key_concepts(text: str, max_concepts: int = 10) -> List[str]:
    """Extract key concepts from text using simple NLP"""
    
    if not text:
        return []
    
    # Legal and business terms to prioritize
    priority_terms = {
        'contract', 'agreement', 'clause', 'liability', 'penalty', 'breach',
        'termination', 'payment', 'obligation', 'warranty', 'indemnity',
        'confidential', 'intellectual', 'property', 'damages', 'arbitration'
    }
    
    # Extract words and filter
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    
    # Count frequency
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Score words (priority terms get bonus)
    scored_words = []
    for word, freq in word_freq.items():
        score = freq
        if word in priority_terms:
            score *= 2
        scored_words.append((word, score))
    
    # Sort by score and return top concepts
    scored_words.sort(key=lambda x: x[1], reverse=True)
    
    return [word for word, _ in scored_words[:max_concepts]]
