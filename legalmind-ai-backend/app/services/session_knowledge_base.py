"""
Session-based knowledge management with cross-document intelligence
Enhanced with fallbacks and comprehensive error handling
"""

import asyncio
import uuid
import re
from dotenv import load_dotenv
load_dotenv() 
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

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

# Import services and models with fallbacks
try_import_service('gemini_analyzer', 'app.services.gemini_analyzer', 'GeminiAnalyzer')
try_import_service('models', 'app.services.multi_document_models')

# Import models with fallbacks
if services_available.get('models'):
    try:
        from app.services.multi_document_models import (
            StandardDocument, DocumentChunk, DocumentRelationship
        )
        MODELS_AVAILABLE = True
    except ImportError:
        MODELS_AVAILABLE = False
else:
    MODELS_AVAILABLE = False

# Mock models when real models aren't available
if not MODELS_AVAILABLE:
    @dataclass
    class MockStandardDocument:
        id: str
        filename: str
        content_type: str
        content: str
        page_count: int = 1
        word_count: int = 0
        metadata: Dict[str, Any] = None
        
        def __post_init__(self):
            if self.metadata is None:
                self.metadata = {}
            if self.word_count == 0 and self.content:
                self.word_count = len(self.content.split())
    
    @dataclass
    class MockDocumentChunk:
        id: str
        document_id: str
        content: str
        page_range: str = "1-1"
        chunk_index: int = 0
        total_chunks: int = 1
        metadata: Dict[str, Any] = None
        
        def __post_init__(self):
            if self.metadata is None:
                self.metadata = {}
    
    @dataclass 
    class MockDocumentRelationship:
        id: str
        document_1_id: str
        document_2_id: str
        relationship_type: str
        similarity_score: float
        common_topics: List[str]
        relationship_details: Dict[str, Any]
        created_at: datetime
        
        def to_dict(self):
            return {
                'id': self.id,
                'document_1_id': self.document_1_id,
                'document_2_id': self.document_2_id,
                'relationship_type': self.relationship_type,
                'similarity_score': self.similarity_score,
                'common_topics': self.common_topics,
                'relationship_details': self.relationship_details,
                'created_at': self.created_at.isoformat() if self.created_at else None
            }
    
    StandardDocument = MockStandardDocument
    DocumentChunk = MockDocumentChunk
    DocumentRelationship = MockDocumentRelationship

# Mock analyzer for when real analyzer isn't available
class MockGeminiAnalyzer:
    async def _generate_content_async(self, prompt: str) -> str:
        """Mock content generation with basic analysis"""
        logger.info("Using mock Gemini analyzer for content analysis")
        
        # Extract basic patterns from prompt content
        if 'analyze this document chunk' in prompt.lower():
            return """
            Topics: legal_terms, obligations, contract_provisions
            Entities: Party_A, Party_B, Agreement_Date
            Key_phrases: shall comply, liability, termination clause
            Sentiment: neutral
            Type: legal_document
            """
        elif 'create a comprehensive summary' in prompt.lower():
            return """
            Title: Legal Agreement Document
            Type: contract
            Summary: This document outlines terms and conditions with moderate complexity and standard legal provisions.
            Risk Level: medium
            """
        else:
            return "Analysis completed with mock analyzer - limited functionality available."

@dataclass 
class TopicCluster:
    """Represents a cluster of related topics across documents"""
    id: str
    name: str
    documents: Set[str] = field(default_factory=set)
    chunks: Set[str] = field(default_factory=set)
    keywords: List[str] = field(default_factory=list)
    similarity_score: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class DocumentSummary:
    """High-level summary of document for conversation context"""
    document_id: str
    title: str
    document_type: str
    key_topics: List[str]
    main_entities: List[str]
    risk_level: str
    summary_text: str
    chunk_count: int
    created_at: datetime
    word_count: Optional[int] = None
    page_count: Optional[int] = None
    
    def to_dict(self):
        return {
            'document_id': self.document_id,
            'title': self.title,
            'document_type': self.document_type,
            'key_topics': self.key_topics,
            'main_entities': self.main_entities,
            'risk_level': self.risk_level,
            'summary_text': self.summary_text,
            'chunk_count': self.chunk_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'word_count': self.word_count,
            'page_count': self.page_count
        }

class SessionKnowledgeBase:
    """
    Manages knowledge across multiple documents in a conversation session with fallbacks
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.documents: Dict[str, StandardDocument] = {}
        self.chunks: Dict[str, DocumentChunk] = {}
        self.document_summaries: Dict[str, DocumentSummary] = {}
        self.relationships: Dict[str, DocumentRelationship] = {}
        self.topic_clusters: Dict[str, TopicCluster] = {}
        
        # Initialize embeddings storage if numpy is available
        if NUMPY_AVAILABLE:
            self.chunk_embeddings: Dict[str, np.ndarray] = {}
        else:
            self.chunk_embeddings: Dict[str, List[float]] = {}
        
        # Initialize analyzer with fallback
        if services_available.get('gemini_analyzer'):
            self.gemini_analyzer = services_available['gemini_analyzer']()
        else:
            self.gemini_analyzer = MockGeminiAnalyzer()
        
        self.logger = logger
        self.created_at = datetime.now(timezone.utc)
        self.last_updated = datetime.now(timezone.utc)
        
        self.logger.info(f"SessionKnowledgeBase initialized for session {session_id}")
    
    async def add_document(self, document: StandardDocument, chunks: List[DocumentChunk]) -> DocumentSummary:
        """
        Add new document to the knowledge base
        
        Args:
            document: StandardDocument to add
            chunks: List of document chunks
            
        Returns:
            DocumentSummary for conversation context
        """
        
        try:
            self.logger.info(f"Adding document {document.id} to knowledge base {self.session_id}")
            
            # Store document and chunks
            self.documents[document.id] = document
            for chunk in chunks:
                self.chunks[chunk.id] = chunk
            
            # Process chunks in parallel with error handling
            chunk_analyses = []
            for chunk in chunks:
                try:
                    analysis = await self._analyze_chunk(chunk)
                    chunk_analyses.append(analysis)
                except Exception as e:
                    self.logger.warning(f"Error analyzing chunk {chunk.id}: {e}")
                    chunk_analyses.append({
                        'topics': [],
                        'entities': [],
                        'key_phrases': [],
                        'sentiment': 'neutral'
                    })
            
            # Store chunk analyses
            for chunk, analysis in zip(chunks, chunk_analyses):
                # Store embeddings if available
                if 'embeddings' in analysis:
                    if NUMPY_AVAILABLE:
                        self.chunk_embeddings[chunk.id] = np.array(analysis['embeddings'])
                    else:
                        self.chunk_embeddings[chunk.id] = analysis['embeddings']
                
                # Update chunk metadata with analysis
                chunk.metadata.update({
                    'topics': analysis.get('topics', []),
                    'entities': analysis.get('entities', []),
                    'key_phrases': analysis.get('key_phrases', []),
                    'sentiment': analysis.get('sentiment', 'neutral'),
                    'analysis_quality': analysis.get('quality', 'standard')
                })
            
            # Generate document summary
            doc_summary = await self._generate_document_summary(document, chunks)
            self.document_summaries[document.id] = doc_summary
            
            # Update cross-document relationships
            await self._update_relationships(document.id)
            
            # Update topic clusters
            await self._update_topic_clusters(document.id, doc_summary.key_topics)
            
            self.last_updated = datetime.now(timezone.utc)
            
            self.logger.info(f"Successfully added document {document.id} to knowledge base with {len(chunks)} chunks")
            
            return doc_summary
            
        except Exception as e:
            self.logger.error(f"Error adding document to knowledge base: {str(e)}")
            
            # Create fallback summary
            fallback_summary = DocumentSummary(
                document_id=document.id,
                title=document.filename,
                document_type='unknown',
                key_topics=['document_processing'],
                main_entities=[],
                risk_level='medium',
                summary_text=f'Document {document.filename} processed with limited analysis.',
                chunk_count=len(chunks),
                created_at=datetime.now(timezone.utc),
                word_count=document.word_count,
                page_count=document.page_count
            )
            
            self.document_summaries[document.id] = fallback_summary
            return fallback_summary
    
    async def _analyze_chunk(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """
        Analyze individual chunk for topics, entities, and embeddings
        """
        
        try:
            # Use Gemini for content analysis
            analysis_prompt = f"""
            Analyze this document chunk and extract:
            1. Main topics (3-5 key topics)
            2. Named entities (people, organizations, dates, amounts)
            3. Key phrases (important terms and concepts)
            4. Sentiment (positive, negative, neutral)
            5. Document type indicators

            Chunk content (Pages {chunk.page_range}):
            {chunk.content[:2000]}{"..." if len(chunk.content) > 2000 else ""}

            Respond in a structured format with the requested information.
            """
            
            if hasattr(self.gemini_analyzer, '_generate_content_async'):
                analysis_result = await self.gemini_analyzer._generate_content_async(analysis_prompt)
            else:
                analysis_result = await self.gemini_analyzer._generate_content_async(analysis_prompt)
            
            # Parse the analysis
            return {
                'topics': self._extract_topics_from_analysis(analysis_result),
                'entities': self._extract_entities_from_analysis(analysis_result),
                'key_phrases': self._extract_phrases_from_analysis(analysis_result),
                'sentiment': self._extract_sentiment_from_analysis(analysis_result),
                'analysis_text': analysis_result,
                'quality': 'high' if len(analysis_result) > 100 else 'standard'
            }
            
        except Exception as e:
            self.logger.warning(f"Error analyzing chunk {chunk.id}: {str(e)}")
            
            # Fallback analysis using simple text processing
            return self._fallback_chunk_analysis(chunk)
    
    def _fallback_chunk_analysis(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """Fallback analysis when AI analyzer fails"""
        
        content_lower = chunk.content.lower()
        
        # Extract simple topics based on legal keywords
        legal_keywords = {
            'contract': ['contract', 'agreement', 'terms'],
            'payment': ['payment', 'fee', 'cost', 'price'],
            'liability': ['liability', 'responsible', 'liable'],
            'termination': ['termination', 'end', 'expire'],
            'obligations': ['obligation', 'duty', 'must', 'shall'],
            'rights': ['rights', 'entitled', 'privilege']
        }
        
        detected_topics = []
        for topic, keywords in legal_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                detected_topics.append(topic)
        
        # Extract potential entities (simple pattern matching)
        entities = []
        # Find dates
        date_patterns = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', chunk.content)
        entities.extend([f"Date: {date}" for date in date_patterns[:3]])
        
        # Find monetary amounts
        money_patterns = re.findall(r'\$[\d,]+(?:\.\d{2})?', chunk.content)
        entities.extend([f"Amount: {amount}" for amount in money_patterns[:3]])
        
        return {
            'topics': detected_topics[:5],
            'entities': entities[:5],
            'key_phrases': detected_topics,
            'sentiment': 'neutral',
            'analysis_text': 'Fallback analysis completed',
            'quality': 'basic'
        }
    
    async def _generate_document_summary(self, document: StandardDocument, chunks: List[DocumentChunk]) -> DocumentSummary:
        """
        Generate comprehensive document summary
        """
        
        try:
            # Combine topics from all chunks
            all_topics = []
            all_entities = []
            
            for chunk in chunks:
                chunk_topics = chunk.metadata.get('topics', [])
                chunk_entities = chunk.metadata.get('entities', [])
                all_topics.extend(chunk_topics)
                all_entities.extend(chunk_entities)
            
            # Get unique topics and entities
            unique_topics = list(dict.fromkeys(all_topics))[:10]  # Preserve order, limit to 10
            unique_entities = list(dict.fromkeys(all_entities))[:10]
            
            # Generate summary using AI or fallback
            if hasattr(self.gemini_analyzer, '_generate_content_async') and not isinstance(self.gemini_analyzer, MockGeminiAnalyzer):
                summary_response = await self._generate_ai_summary(document, unique_topics, unique_entities)
            else:
                summary_response = self._generate_fallback_summary(document, unique_topics, unique_entities)
            
            # Extract information from response
            title = self._extract_title_from_summary(summary_response, document.filename)
            doc_type = self._extract_document_type(summary_response)
            risk_level = self._extract_risk_level(summary_response)
            summary_text = self._extract_summary_text(summary_response)
            
            return DocumentSummary(
                document_id=document.id,
                title=title,
                document_type=doc_type,
                key_topics=unique_topics,
                main_entities=unique_entities,
                risk_level=risk_level,
                summary_text=summary_text,
                chunk_count=len(chunks),
                created_at=datetime.now(timezone.utc),
                word_count=document.word_count,
                page_count=document.page_count
            )
            
        except Exception as e:
            self.logger.error(f"Error generating document summary: {str(e)}")
            
            # Return basic fallback summary
            return DocumentSummary(
                document_id=document.id,
                title=document.filename,
                document_type=self._guess_document_type(document.filename),
                key_topics=self._extract_basic_topics(document.content),
                main_entities=[],
                risk_level="medium",
                summary_text=f"Document analysis of {document.filename} with {len(chunks)} sections.",
                chunk_count=len(chunks),
                created_at=datetime.now(timezone.utc),
                word_count=document.word_count,
                page_count=document.page_count
            )
    
    async def _generate_ai_summary(self, document: StandardDocument, topics: List[str], entities: List[str]) -> str:
        """Generate summary using AI analyzer"""
        summary_prompt = f"""
        Create a comprehensive summary of this document:
        
        Document: {document.filename}
        Type: {document.content_type}
        Pages: {document.page_count}
        Word Count: {document.word_count}
        
        Key Topics: {', '.join(topics)}
        Main Entities: {', '.join(entities)}
        
        Content Preview:
        {document.content[:1000]}{"..." if len(document.content) > 1000 else ""}
        
        Provide:
        1. A concise title for the document
        2. Document type classification (contract, agreement, policy, etc.)
        3. 2-3 sentence summary
        4. Risk level assessment (low, medium, high)
        """
        
        return await self.gemini_analyzer._generate_content_async(summary_prompt)
    
    def _generate_fallback_summary(self, document: StandardDocument, topics: List[str], entities: List[str]) -> str:
        """Generate basic summary without AI"""
        
        doc_type = self._guess_document_type(document.filename)
        risk_level = 'medium'  # Default risk level
        
        # Create basic summary
        summary = f"""
        Title: {document.filename}
        Type: {doc_type}
        Summary: This {doc_type} contains {len(topics)} main topics and has {document.word_count} words across {document.page_count} pages.
        Risk Level: {risk_level}
        """
        
        return summary
    
    def _guess_document_type(self, filename: str) -> str:
        """Guess document type from filename"""
        filename_lower = filename.lower()
        
        type_indicators = {
            'contract': ['contract', 'agreement', 'terms'],
            'policy': ['policy', 'privacy', 'terms_of_service'],
            'lease': ['lease', 'rental', 'rent'],
            'employment': ['employment', 'job', 'work'],
            'legal_document': ['legal', 'law', 'court'],
            'financial': ['loan', 'mortgage', 'financial']
        }
        
        for doc_type, keywords in type_indicators.items():
            if any(keyword in filename_lower for keyword in keywords):
                return doc_type
        
        return 'document'
    
    def _extract_basic_topics(self, content: str) -> List[str]:
        """Extract basic topics from content without AI"""
        
        content_lower = content.lower()
        basic_topics = []
        
        topic_keywords = {
            'payment': ['payment', 'pay', 'fee', 'cost'],
            'termination': ['termination', 'end', 'expire', 'cancel'],
            'liability': ['liability', 'responsible', 'liable'],
            'obligations': ['obligation', 'duty', 'must', 'shall'],
            'confidentiality': ['confidential', 'private', 'secret'],
            'intellectual_property': ['intellectual', 'property', 'copyright']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                basic_topics.append(topic)
        
        return basic_topics[:5]
    
    async def _update_relationships(self, new_document_id: str):
        """
        Update cross-document relationships
        """
        
        try:
            if new_document_id not in self.document_summaries:
                return
            
            new_doc_summary = self.document_summaries[new_document_id]
            new_topics = set(new_doc_summary.key_topics)
            
            # Compare with existing documents
            for existing_doc_id, existing_summary in self.document_summaries.items():
                if existing_doc_id == new_document_id:
                    continue
                
                existing_topics = set(existing_summary.key_topics)
                
                # Calculate topic overlap
                common_topics = new_topics & existing_topics
                total_topics = new_topics | existing_topics
                
                if common_topics and total_topics:
                    similarity_score = len(common_topics) / len(total_topics)
                    
                    if similarity_score > 0.2:  # Threshold for related documents
                        relationship_id = f"{new_document_id}_{existing_doc_id}"
                        
                        relationship = DocumentRelationship(
                            id=relationship_id,
                            document_1_id=new_document_id,
                            document_2_id=existing_doc_id,
                            relationship_type=self._classify_relationship(similarity_score),
                            similarity_score=similarity_score,
                            common_topics=list(common_topics),
                            relationship_details={
                                'topic_overlap': len(common_topics),
                                'total_unique_topics': len(total_topics),
                                'document_1_title': new_doc_summary.title,
                                'document_2_title': existing_summary.title,
                                'relationship_strength': 'strong' if similarity_score > 0.6 else 'medium' if similarity_score > 0.4 else 'weak'
                            },
                            created_at=datetime.now(timezone.utc)
                        )
                        
                        self.relationships[relationship_id] = relationship
            
            self.logger.info(f"Updated relationships for document {new_document_id}")
            
        except Exception as e:
            self.logger.error(f"Error updating relationships: {str(e)}")
    
    async def _update_topic_clusters(self, document_id: str, topics: List[str]):
        """
        Update topic clusters with new document
        """
        
        try:
            if not topics:
                return
            
            doc_topics = set(topics)
            
            # Find existing clusters that overlap
            matching_clusters = []
            for cluster_id, cluster in self.topic_clusters.items():
                cluster_topics = set(cluster.keywords)
                overlap = doc_topics & cluster_topics
                
                if overlap:
                    overlap_ratio = len(overlap) / len(doc_topics | cluster_topics)
                    matching_clusters.append((cluster_id, overlap_ratio, overlap))
            
            if matching_clusters:
                # Add to best matching cluster
                best_cluster_id = max(matching_clusters, key=lambda x: x[1])[0]
                self.topic_clusters[best_cluster_id].documents.add(document_id)
                self.topic_clusters[best_cluster_id].keywords.extend(topics)
                
                # Remove duplicates while preserving order
                seen = set()
                unique_keywords = []
                for keyword in self.topic_clusters[best_cluster_id].keywords:
                    if keyword not in seen:
                        seen.add(keyword)
                        unique_keywords.append(keyword)
                
                self.topic_clusters[best_cluster_id].keywords = unique_keywords
                
                self.logger.info(f"Added document {document_id} to existing cluster {best_cluster_id}")
            
            else:
                # Create new cluster
                cluster_id = f"cluster_{len(self.topic_clusters) + 1}"
                cluster_name = f"Topic Cluster {len(self.topic_clusters) + 1}"
                
                # Use most common topic as cluster name if available
                if topics:
                    cluster_name = f"{topics[0].title()} Cluster"
                
                self.topic_clusters[cluster_id] = TopicCluster(
                    id=cluster_id,
                    name=cluster_name,
                    documents={document_id},
                    keywords=topics[:10],  # Limit keywords
                    similarity_score=1.0,  # New cluster starts with perfect self-similarity
                    created_at=datetime.now(timezone.utc)
                )
                
                self.logger.info(f"Created new topic cluster {cluster_id} for document {document_id}")
            
        except Exception as e:
            self.logger.error(f"Error updating topic clusters: {str(e)}")
    
    def get_related_documents(self, document_id: str, threshold: float = 0.3) -> List[DocumentRelationship]:
        """
        Get documents related to the specified document
        """
        
        try:
            related = []
            for relationship in self.relationships.values():
                if (relationship.document_1_id == document_id or relationship.document_2_id == document_id) \
                   and relationship.similarity_score >= threshold:
                    related.append(relationship)
            
            return sorted(related, key=lambda x: x.similarity_score, reverse=True)
        
        except Exception as e:
            self.logger.error(f"Error getting related documents: {str(e)}")
            return []
    
    def get_documents_by_topic(self, topic: str) -> List[str]:
        """
        Get documents that contain the specified topic
        """
        
        try:
            matching_docs = []
            topic_lower = topic.lower()
            
            for doc_id, summary in self.document_summaries.items():
                doc_topics = [t.lower() for t in summary.key_topics]
                if any(topic_lower in dt or dt in topic_lower for dt in doc_topics):
                    matching_docs.append(doc_id)
            
            return matching_docs
        
        except Exception as e:
            self.logger.error(f"Error getting documents by topic: {str(e)}")
            return []
    
    def get_session_overview(self) -> Dict[str, Any]:
        """
        Get comprehensive overview of the session knowledge base
        """
        
        try:
            return {
                'session_id': self.session_id,
                'created_at': self.created_at.isoformat() if self.created_at else None,
                'last_updated': self.last_updated.isoformat() if self.last_updated else None,
                'statistics': {
                    'total_documents': len(self.documents),
                    'total_chunks': len(self.chunks),
                    'total_relationships': len(self.relationships),
                    'topic_clusters': len(self.topic_clusters),
                    'total_words': sum(doc.word_count for doc in self.documents.values() if hasattr(doc, 'word_count')),
                    'avg_chunks_per_doc': len(self.chunks) / max(len(self.documents), 1),
                    'embeddings_available': len(self.chunk_embeddings),
                    'ai_analysis_quality': self._assess_analysis_quality()
                },
                'documents': [
                    {
                        'id': doc_id,
                        'title': summary.title,
                        'type': summary.document_type,
                        'topics': summary.key_topics[:5],
                        'risk_level': summary.risk_level,
                        'chunk_count': summary.chunk_count,
                        'word_count': summary.word_count,
                        'page_count': summary.page_count
                    }
                    for doc_id, summary in self.document_summaries.items()
                ],
                'top_topics': self._get_top_topics(),
                'document_relationships': len(self.relationships),
                'relationship_strength_distribution': self._get_relationship_strength_distribution(),
                'capabilities': {
                    'ai_analysis': not isinstance(self.gemini_analyzer, MockGeminiAnalyzer),
                    'embeddings': NUMPY_AVAILABLE,
                    'topic_clustering': True,
                    'cross_document_analysis': True
                }
            }
        
        except Exception as e:
            self.logger.error(f"Error generating session overview: {str(e)}")
            return {
                'session_id': self.session_id,
                'error': str(e),
                'statistics': {
                    'total_documents': len(self.documents),
                    'total_chunks': len(self.chunks)
                }
            }
    
    def _assess_analysis_quality(self) -> str:
        """Assess overall analysis quality"""
        
        if isinstance(self.gemini_analyzer, MockGeminiAnalyzer):
            return 'basic'
        
        # Check chunk analysis quality
        high_quality_chunks = 0
        for chunk in self.chunks.values():
            quality = chunk.metadata.get('analysis_quality', 'standard')
            if quality == 'high':
                high_quality_chunks += 1
        
        if len(self.chunks) == 0:
            return 'unknown'
        
        quality_ratio = high_quality_chunks / len(self.chunks)
        
        if quality_ratio > 0.7:
            return 'high'
        elif quality_ratio > 0.3:
            return 'standard'
        else:
            return 'basic'
    
    def _get_relationship_strength_distribution(self) -> Dict[str, int]:
        """Get distribution of relationship strengths"""
        
        distribution = {'strong': 0, 'medium': 0, 'weak': 0}
        
        for relationship in self.relationships.values():
            strength = relationship.relationship_details.get('relationship_strength', 'medium')
            if strength in distribution:
                distribution[strength] += 1
        
        return distribution
    
    def _get_top_topics(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most common topics across all documents"""
        
        try:
            topic_counts = defaultdict(int)
            topic_documents = defaultdict(set)
            
            for doc_id, summary in self.document_summaries.items():
                for topic in summary.key_topics:
                    topic_counts[topic] += 1
                    topic_documents[topic].add(doc_id)
            
            top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
            
            return [
                {
                    'topic': topic,
                    'frequency': count,
                    'document_count': len(topic_documents[topic]),
                    'documents': list(topic_documents[topic]),
                    'coverage': len(topic_documents[topic]) / max(len(self.documents), 1)
                }
                for topic, count in top_topics
            ]
        
        except Exception as e:
            self.logger.error(f"Error getting top topics: {str(e)}")
            return []
    
    # Enhanced helper methods for parsing responses
    def _extract_topics_from_analysis(self, analysis_text: str) -> List[str]:
        """Extract topics from analysis text"""
        try:
            # Look for topic patterns
            patterns = [
                r'topics?[:\-\s]+([^.\n]+)',
                r'main topics?[:\-\s]+([^.\n]+)',
                r'key topics?[:\-\s]+([^.\n]+)'
            ]
            
            topics = []
            for pattern in patterns:
                matches = re.findall(pattern, analysis_text, re.IGNORECASE)
                for match in matches:
                    # Split on common separators and clean
                    topic_items = re.split(r'[,;]', match)
                    for item in topic_items:
                        clean_topic = item.strip().lower()
                        if len(clean_topic) > 2 and clean_topic not in topics:
                            topics.append(clean_topic)
            
            return topics[:5]  # Limit to 5 topics
        
        except Exception as e:
            self.logger.warning(f"Error extracting topics: {e}")
            return []
    
    def _extract_entities_from_analysis(self, analysis_text: str) -> List[str]:
        """Extract entities from analysis text"""
        try:
            patterns = [
                r'entit[yi][e]?[s]?[:\-\s]+([^.\n]+)',
                r'named entit[yi][e]?[s]?[:\-\s]+([^.\n]+)'
            ]
            
            entities = []
            for pattern in patterns:
                matches = re.findall(pattern, analysis_text, re.IGNORECASE)
                for match in matches:
                    entity_items = re.split(r'[,;]', match)
                    for item in entity_items:
                        clean_entity = item.strip()
                        if len(clean_entity) > 2 and clean_entity not in entities:
                            entities.append(clean_entity)
            
            return entities[:5]
        
        except Exception as e:
            self.logger.warning(f"Error extracting entities: {e}")
            return []
    
    def _extract_phrases_from_analysis(self, analysis_text: str) -> List[str]:
        """Extract key phrases from analysis text"""
        try:
            patterns = [
                r'key phrases?[:\-\s]+([^.\n]+)',
                r'phrases?[:\-\s]+([^.\n]+)',
                r'important terms?[:\-\s]+([^.\n]+)'
            ]
            
            phrases = []
            for pattern in patterns:
                matches = re.findall(pattern, analysis_text, re.IGNORECASE)
                for match in matches:
                    phrase_items = re.split(r'[,;]', match)
                    for item in phrase_items:
                        clean_phrase = item.strip()
                        if len(clean_phrase) > 3 and clean_phrase not in phrases:
                            phrases.append(clean_phrase)
            
            return phrases[:5]
        
        except Exception as e:
            self.logger.warning(f"Error extracting phrases: {e}")
            return []
    
    def _extract_sentiment_from_analysis(self, analysis_text: str) -> str:
        """Extract sentiment from analysis"""
        analysis_lower = analysis_text.lower()
        
        if any(word in analysis_lower for word in ['positive', 'favorable', 'good']):
            return 'positive'
        elif any(word in analysis_lower for word in ['negative', 'unfavorable', 'bad', 'problematic']):
            return 'negative'
        else:
            return 'neutral'
    
    def _extract_title_from_summary(self, summary_text: str, fallback: str) -> str:
        """Extract title from summary"""
        try:
            patterns = [
                r'title[:\-\s]+([^.\n]+)',
                r'document title[:\-\s]+([^.\n]+)',
                r'name[:\-\s]+([^.\n]+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, summary_text, re.IGNORECASE)
                if match:
                    title = match.group(1).strip()
                    if len(title) > 3:
                        return title
            
            return fallback
        
        except Exception:
            return fallback
    
    def _extract_document_type(self, summary_text: str) -> str:
        """Extract document type from summary"""
        try:
            patterns = [
                r'type[:\-\s]+([^.\n]+)',
                r'document type[:\-\s]+([^.\n]+)',
                r'classification[:\-\s]+([^.\n]+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, summary_text, re.IGNORECASE)
                if match:
                    doc_type = match.group(1).strip().lower()
                    if len(doc_type) > 2:
                        return doc_type
            
            return "document"
        
        except Exception:
            return "document"
    
    def _extract_risk_level(self, summary_text: str) -> str:
        """Extract risk level from summary"""
        try:
            risk_keywords = {
                'low': ['low', 'minimal', 'minor', 'slight'],
                'medium': ['medium', 'moderate', 'average', 'standard'],
                'high': ['high', 'significant', 'major', 'severe', 'critical']
            }
            
            summary_lower = summary_text.lower()
            
            # Look for explicit risk level mentions
            for level, keywords in risk_keywords.items():
                if any(f"risk level: {keyword}" in summary_lower or f"risk: {keyword}" in summary_lower for keyword in keywords):
                    return level
            
            # Look for general risk indicators
            for level, keywords in risk_keywords.items():
                if any(keyword in summary_lower for keyword in keywords):
                    return level
            
            return "medium"  # Default
        
        except Exception:
            return "medium"
    
    def _extract_summary_text(self, summary_text: str) -> str:
        """Extract clean summary text"""
        try:
            # Look for summary section
            summary_patterns = [
                r'summary[:\-\s]+([^.\n]*\.)',
                r'overview[:\-\s]+([^.\n]*\.)',
                r'description[:\-\s]+([^.\n]*\.)'
            ]
            
            for pattern in summary_patterns:
                match = re.search(pattern, summary_text, re.IGNORECASE)
                if match:
                    summary = match.group(1).strip()
                    if len(summary) > 10:
                        return summary
            
            # Fallback: extract first meaningful sentence
            sentences = [s.strip() for s in summary_text.split('.') if len(s.strip()) > 10]
            if sentences:
                return sentences[0] + '.'
            
            return "Document processed successfully."
        
        except Exception:
            return "Document analysis completed."
    
    def _classify_relationship(self, similarity_score: float) -> str:
        """Classify relationship type based on similarity score"""
        if similarity_score > 0.7:
            return "highly_related"
        elif similarity_score > 0.5:
            return "related"
        elif similarity_score > 0.3:
            return "loosely_related"
        else:
            return "weakly_related"
    
    def search_chunks(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search chunks by content similarity"""
        
        try:
            results = []
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            for chunk_id, chunk in self.chunks.items():
                chunk_words = set(chunk.content.lower().split())
                
                # Calculate simple word overlap score
                overlap = len(query_words & chunk_words)
                total = len(query_words | chunk_words)
                
                if total > 0:
                    score = overlap / total
                    
                    if score > 0.1:  # Minimum threshold
                        results.append({
                            'chunk_id': chunk_id,
                            'document_id': chunk.document_id,
                            'score': score,
                            'content': chunk.content[:200] + "...",
                            'page_range': chunk.page_range,
                            'matching_words': list(query_words & chunk_words)
                        })
            
            # Sort by score and return top results
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:max_results]
        
        except Exception as e:
            self.logger.error(f"Error searching chunks: {str(e)}")
            return []

class KnowledgeBaseManager:
    """
    Manages multiple session knowledge bases with enhanced capabilities
    """
    
    def __init__(self):
        self.knowledge_bases: Dict[str, SessionKnowledgeBase] = {}
        self.logger = logger
        self.cleanup_interval_hours = 24
        self.max_sessions = 1000  # Prevent memory overflow
        
        self.logger.info("KnowledgeBaseManager initialized")
    
    async def get_session_kb(self, session_id: str) -> SessionKnowledgeBase:
        """
        Get or create knowledge base for session
        """
        
        try:
            if session_id not in self.knowledge_bases:
                # Check if we're at capacity
                if len(self.knowledge_bases) >= self.max_sessions:
                    await self.cleanup_expired_sessions(hours=1)  # More aggressive cleanup
                
                self.knowledge_bases[session_id] = SessionKnowledgeBase(session_id)
                self.logger.info(f"Created new knowledge base for session: {session_id}")
            
            return self.knowledge_bases[session_id]
        
        except Exception as e:
            self.logger.error(f"Error getting session KB: {str(e)}")
            # Return a new KB even if there's an error
            return SessionKnowledgeBase(session_id)
    
    async def cleanup_expired_sessions(self, hours: int = 24) -> int:
        """
        Cleanup expired session knowledge bases
        """
        
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            expired_sessions = [
                session_id for session_id, kb in self.knowledge_bases.items()
                if kb.last_updated < cutoff_time
            ]
            
            for session_id in expired_sessions:
                try:
                    del self.knowledge_bases[session_id]
                    self.logger.info(f"Cleaned up expired knowledge base: {session_id}")
                except Exception as e:
                    self.logger.warning(f"Error cleaning up session {session_id}: {e}")
            
            if expired_sessions:
                self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            
            return len(expired_sessions)
        
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            return 0
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """
        Get system-wide knowledge base statistics
        """
        
        try:
            if not self.knowledge_bases:
                return {
                    'active_sessions': 0,
                    'total_documents': 0,
                    'total_chunks': 0,
                    'total_relationships': 0,
                    'average_documents_per_session': 0,
                    'sessions_with_multiple_docs': 0,
                    'system_health': 'idle'
                }
            
            total_documents = sum(len(kb.documents) for kb in self.knowledge_bases.values())
            total_chunks = sum(len(kb.chunks) for kb in self.knowledge_bases.values())
            total_relationships = sum(len(kb.relationships) for kb in self.knowledge_bases.values())
            
            sessions_with_multiple = sum(1 for kb in self.knowledge_bases.values() if len(kb.documents) > 1)
            
            # System health assessment
            system_health = 'healthy'
            if len(self.knowledge_bases) > self.max_sessions * 0.9:
                system_health = 'near_capacity'
            elif len(self.knowledge_bases) > self.max_sessions * 0.7:
                system_health = 'high_usage'
            
            return {
                'active_sessions': len(self.knowledge_bases),
                'total_documents': total_documents,
                'total_chunks': total_chunks,
                'total_relationships': total_relationships,
                'average_documents_per_session': total_documents / len(self.knowledge_bases),
                'average_chunks_per_session': total_chunks / len(self.knowledge_bases),
                'sessions_with_multiple_docs': sessions_with_multiple,
                'multi_doc_percentage': (sessions_with_multiple / len(self.knowledge_bases)) * 100,
                'system_health': system_health,
                'memory_usage_estimate': len(self.knowledge_bases) * 50,  # Rough MB estimate
                'capabilities': {
                    'ai_analysis': services_available.get('gemini_analyzer') is not None,
                    'numpy_operations': NUMPY_AVAILABLE,
                    'models_available': MODELS_AVAILABLE
                }
            }
        
        except Exception as e:
            self.logger.error(f"Error getting system statistics: {str(e)}")
            return {
                'active_sessions': len(self.knowledge_bases),
                'error': str(e),
                'system_health': 'error'
            }
    
    async def get_cross_session_insights(self) -> Dict[str, Any]:
        """Get insights across all active sessions"""
        
        try:
            if not self.knowledge_bases:
                return {'message': 'No active sessions'}
            
            # Aggregate data across sessions
            all_topics = []
            all_doc_types = []
            all_risk_levels = []
            
            for kb in self.knowledge_bases.values():
                for summary in kb.document_summaries.values():
                    all_topics.extend(summary.key_topics)
                    all_doc_types.append(summary.document_type)
                    all_risk_levels.append(summary.risk_level)
            
            # Calculate distributions
            topic_freq = defaultdict(int)
            for topic in all_topics:
                topic_freq[topic] += 1
            
            type_freq = defaultdict(int)
            for doc_type in all_doc_types:
                type_freq[doc_type] += 1
            
            risk_freq = defaultdict(int)
            for risk in all_risk_levels:
                risk_freq[risk] += 1
            
            return {
                'total_sessions_analyzed': len(self.knowledge_bases),
                'most_common_topics': dict(sorted(topic_freq.items(), key=lambda x: x[1], reverse=True)[:10]),
                'document_type_distribution': dict(type_freq),
                'risk_level_distribution': dict(risk_freq),
                'average_documents_per_session': len(all_doc_types) / len(self.knowledge_bases),
                'sessions_by_activity': {
                    'single_document': sum(1 for kb in self.knowledge_bases.values() if len(kb.documents) == 1),
                    'multiple_documents': sum(1 for kb in self.knowledge_bases.values() if len(kb.documents) > 1),
                    'with_relationships': sum(1 for kb in self.knowledge_bases.values() if len(kb.relationships) > 0)
                }
            }
        
        except Exception as e:
            self.logger.error(f"Error generating cross-session insights: {str(e)}")
            return {'error': str(e)}
    
    def get_manager_capabilities(self) -> Dict[str, Any]:
        """Get manager capabilities and status"""
        
        return {
            'max_sessions': self.max_sessions,
            'cleanup_interval_hours': self.cleanup_interval_hours,
            'current_load': len(self.knowledge_bases) / self.max_sessions,
            'features': {
                'session_management': True,
                'automatic_cleanup': True,
                'cross_session_analytics': True,
                'system_monitoring': True,
                'capacity_management': True
            },
            'dependencies': {
                'ai_analysis': services_available.get('gemini_analyzer') is not None,
                'advanced_math': NUMPY_AVAILABLE,
                'data_models': MODELS_AVAILABLE
            }
        }


# Utility functions
def create_simple_knowledge_base(session_id: str = None) -> SessionKnowledgeBase:
    """Create a simple knowledge base for testing or basic use"""
    
    if not session_id:
        session_id = f"session_{uuid.uuid4().hex[:8]}"
    
    return SessionKnowledgeBase(session_id)

async def analyze_document_relationships(kb: SessionKnowledgeBase) -> Dict[str, Any]:
    """Analyze relationships within a knowledge base"""
    
    try:
        if len(kb.documents) < 2:
            return {
                'status': 'insufficient_documents',
                'message': 'Need at least 2 documents for relationship analysis'
            }
        
        relationship_analysis = {
            'total_relationships': len(kb.relationships),
            'document_count': len(kb.documents),
            'relationship_density': len(kb.relationships) / max((len(kb.documents) * (len(kb.documents) - 1)) / 2, 1),
            'relationship_types': {},
            'strongest_relationship': None,
            'topic_overlap_analysis': {}
        }
        
        # Analyze relationship types
        for rel in kb.relationships.values():
            rel_type = rel.relationship_type
            if rel_type not in relationship_analysis['relationship_types']:
                relationship_analysis['relationship_types'][rel_type] = 0
            relationship_analysis['relationship_types'][rel_type] += 1
        
        # Find strongest relationship
        if kb.relationships:
            strongest = max(kb.relationships.values(), key=lambda x: x.similarity_score)
            relationship_analysis['strongest_relationship'] = {
                'documents': [strongest.document_1_id, strongest.document_2_id],
                'similarity_score': strongest.similarity_score,
                'common_topics': strongest.common_topics
            }
        
        return relationship_analysis
    
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }
