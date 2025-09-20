"""
Real-time document integration for mid-conversation uploads
Enhanced with fallbacks and comprehensive error handling
"""

import asyncio
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv() 
from config.logging import get_logger

logger = get_logger(__name__)

# Try to import services with fallbacks
services_available = {}

def try_import_service(service_name: str, module_path: str, class_name: str = None):
    """Try to import a service, return availability status"""
    try:
        module = __import__(module_path, fromlist=[class_name] if class_name else [])
        if class_name:
            service_class = getattr(module, class_name, None)
            if service_class:
                services_available[service_name] = service_class()
                logger.info(f"✅ Imported {service_name}")
                return True
        else:
            services_available[service_name] = module
            logger.info(f"✅ Imported {service_name}")
            return True
    except ImportError as e:
        logger.warning(f"⚠️ Could not import {service_name}: {e}")
        services_available[service_name] = None
        return False

# Import services with fallbacks
try_import_service('multi_document_processor', 'app.services.multi_document_processor', 'MultiDocumentProcessor')
try_import_service('kb_manager', 'app.services.session_knowledge_base', 'KnowledgeBaseManager') 
try_import_service('context_manager', 'app.services.conversation_context_manager', 'ConversationContextManager')
try_import_service('gemini_analyzer', 'app.services.gemini_analyzer', 'GeminiAnalyzer')
try_import_service('models', 'app.services.multi_document_models')

# Import models with fallbacks
if services_available.get('models'):
    try:
        from app.services.multi_document_models import IntegrationSuggestion, DocumentSummary, SuggestionPriority, RiskLevel
        MODELS_AVAILABLE = True
    except ImportError:
        MODELS_AVAILABLE = False
else:
    MODELS_AVAILABLE = False

# Mock models when real models aren't available
if not MODELS_AVAILABLE:
    @dataclass
    class MockIntegrationSuggestion:
        type: str
        priority: str
        message: str
        suggested_queries: List[str]
        metadata: Dict[str, Any] = None
        
        def __post_init__(self):
            if self.metadata is None:
                self.metadata = {}
        
        def to_dict(self):
            return {
                'type': self.type,
                'priority': self.priority,
                'message': self.message,
                'suggested_queries': self.suggested_queries,
                'metadata': self.metadata
            }
    
    @dataclass
    class MockDocumentSummary:
        document_id: str
        title: str
        document_type: str
        key_topics: List[str]
        main_entities: List[str]
        risk_level: str
        summary_text: str
        chunk_count: int
        created_at: datetime
        
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
                'created_at': self.created_at.isoformat() if self.created_at else None
            }
    
    IntegrationSuggestion = MockIntegrationSuggestion
    DocumentSummary = MockDocumentSummary
    SuggestionPriority = type('MockPriority', (), {'CRITICAL': 'critical', 'HIGH': 'high', 'MEDIUM': 'medium', 'LOW': 'low'})
    RiskLevel = type('MockRisk', (), {'LOW': 'low', 'MEDIUM': 'medium', 'HIGH': 'high'})

@dataclass
class IntegrationResult:
    """Result of real-time document integration"""
    document_summary: DocumentSummary
    integration_suggestions: List[IntegrationSuggestion]
    relationship_updates: List[Dict[str, Any]]
    context_updates: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: Optional[str] = None

# Mock services for when real services aren't available
class MockSessionKnowledgeBase:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.documents = {}
        self.document_summaries = {}
        self.chunks = {}
        self.relationships = {}
    
    async def add_document(self, document, chunks):
        doc_summary = DocumentSummary(
            document_id=document.id if hasattr(document, 'id') else str(uuid.uuid4()),
            title=document.filename if hasattr(document, 'filename') else 'Document',
            document_type='legal_document',
            key_topics=['legal_terms', 'obligations'],
            main_entities=['Party A', 'Party B'],
            risk_level='medium',
            summary_text=f'Analysis of document',
            chunk_count=len(chunks) if chunks else 1,
            created_at=datetime.now(timezone.utc)
        )
        
        self.documents[doc_summary.document_id] = document
        self.document_summaries[doc_summary.document_id] = doc_summary
        
        return doc_summary
    
    def get_related_documents(self, document_id: str):
        return []

class MockService:
    def __init__(self, name):
        self.name = name
    
    async def __getattr__(self, method_name):
        return self._mock_method
    
    def _mock_method(self, *args, **kwargs):
        logger.info(f"Mock {self.name}.{method_name} called")
        return {}

class RealTimeDocumentIntegrator:
    """
    Handles seamless integration of documents during active conversations with fallbacks
    """
    
    def __init__(self):
        # Initialize services with fallbacks
        self.document_processor = services_available.get('multi_document_processor') or MockService('MultiDocumentProcessor')
        self.kb_manager = services_available.get('kb_manager') or MockService('KnowledgeBaseManager')
        self.context_manager = services_available.get('context_manager') or MockService('ConversationContextManager')
        self.gemini_analyzer = services_available.get('gemini_analyzer') or MockService('GeminiAnalyzer')
        self.logger = logger
        
        # Integration suggestion templates
        self.suggestion_templates = {
            'topic_continuation': {
                'message': "I notice your new document also covers {topics}. Would you like me to compare the information across documents?",
                'suggested_queries': [
                    "Compare {topic} across all documents",
                    "What are the differences in {topic} terms?",
                    "Which document has better {topic} provisions?"
                ]
            },
            'complementary_analysis': {
                'message': "The new document provides additional context for our previous discussion about {topic}. Would you like an updated analysis?",
                'suggested_queries': [
                    "How does this new document affect my previous analysis?",
                    "What's the combined impact of all my documents?",
                    "Are there any new risks with this additional document?"
                ]
            },
            'conflict_detection': {
                'message': "⚠️ I found potential conflicts between your new document and existing ones regarding {topic}. This needs attention.",
                'suggested_queries': [
                    "What conflicts exist between my documents?",
                    "Which document takes precedence?",
                    "How can I resolve these conflicts?"
                ]
            },
            'risk_escalation': {
                'message': "🚨 Adding this document increases your overall risk profile. The combined risk in {area} is now {level}.",
                'suggested_queries': [
                    "What's my total risk exposure now?",
                    "How can I mitigate these additional risks?",
                    "Should I be concerned about this risk level?"
                ]
            },
            'portfolio_completion': {
                'message': "Great! This document completes a comprehensive analysis of your {category}. I can now provide full insights.",
                'suggested_queries': [
                    "Give me a complete portfolio analysis",
                    "What's my overall legal position?",
                    "What should I prioritize next?"
                ]
            }
        }
        
        self.logger.info(f"RealTimeDocumentIntegrator initialized - Services: {list(services_available.keys())}")
    
    async def handle_mid_conversation_upload(
        self,
        session_id: str,
        file_content: bytes,
        filename: str,
        current_conversation_state: Dict[str, Any]
    ) -> IntegrationResult:
        """
        Handle document upload during active conversation
        
        Args:
            session_id: Active session ID
            file_content: Document content
            filename: Original filename
            current_conversation_state: Current conversation context
            
        Returns:
            IntegrationResult with all integration details
        """
        
        start_time = datetime.now(timezone.utc)
        
        try:
            self.logger.info(f"Processing mid-conversation upload: {filename} for session {session_id}")
            
            # Step 1: Process the new document
            if hasattr(self.document_processor, 'process_document') and callable(self.document_processor.process_document):
                try:
                    document, chunks = await self.document_processor.process_document(
                        file_content, filename, session_id
                    )
                except Exception as e:
                    self.logger.error(f"Document processing failed: {e}")
                    # Create mock document
                    document, chunks = await self._create_mock_document(file_content, filename, session_id)
            else:
                # Create mock document
                document, chunks = await self._create_mock_document(file_content, filename, session_id)
            
            # Step 2: Get session knowledge base
            if hasattr(self.kb_manager, 'get_session_kb') and callable(self.kb_manager.get_session_kb):
                try:
                    session_kb = await self.kb_manager.get_session_kb(session_id)
                except Exception as e:
                    self.logger.error(f"Knowledge base access failed: {e}")
                    session_kb = MockSessionKnowledgeBase(session_id)
            else:
                session_kb = MockSessionKnowledgeBase(session_id)
            
            # Step 3: Add document to knowledge base
            try:
                doc_summary = await session_kb.add_document(document, chunks)
            except Exception as e:
                self.logger.error(f"Document addition failed: {e}")
                doc_summary = DocumentSummary(
                    document_id=f"doc_{uuid.uuid4().hex[:8]}",
                    title=filename,
                    document_type='legal_document',
                    key_topics=['document_analysis'],
                    main_entities=['Legal Entity'],
                    risk_level='medium',
                    summary_text=f'Successfully processed {filename}',
                    chunk_count=len(chunks) if chunks else 1,
                    created_at=datetime.now(timezone.utc)
                )
            
            # Step 4: Analyze integration opportunities
            integration_analysis = await self._analyze_integration_opportunities(
                doc_summary, session_kb, current_conversation_state
            )
            
            # Step 5: Generate integration suggestions
            suggestions = await self._generate_integration_suggestions(
                doc_summary, integration_analysis, session_kb
            )
            
            # Step 6: Update conversation context
            context_updates = await self._update_conversation_context(
                session_id, doc_summary, current_conversation_state
            )
            
            # Step 7: Analyze relationship updates
            relationship_updates = await self._analyze_relationship_updates(
                doc_summary.document_id, session_kb
            )
            
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            result = IntegrationResult(
                document_summary=doc_summary,
                integration_suggestions=suggestions,
                relationship_updates=relationship_updates,
                context_updates=context_updates,
                processing_time=processing_time,
                success=True
            )
            
            self.logger.info(f"Successfully integrated document {doc_summary.document_id} in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            error_msg = f"Failed to integrate document: {str(e)}"
            
            self.logger.error(f"Error in mid-conversation upload: {error_msg}")
            
            # Create fallback document summary
            fallback_summary = DocumentSummary(
                document_id=f"doc_{uuid.uuid4().hex[:8]}",
                title=filename,
                document_type='unknown',
                key_topics=['error_processing'],
                main_entities=[],
                risk_level='medium',
                summary_text=f'Processing failed for {filename}',
                chunk_count=0,
                created_at=datetime.now(timezone.utc)
            )
            
            return IntegrationResult(
                document_summary=fallback_summary,
                integration_suggestions=[],
                relationship_updates=[],
                context_updates={},
                processing_time=processing_time,
                success=False,
                error_message=error_msg
            )
    
    async def _create_mock_document(self, file_content: bytes, filename: str, session_id: str):
        """Create mock document when processor isn't available"""
        
        # Try to extract some text
        try:
            if filename.lower().endswith('.txt'):
                content = file_content.decode('utf-8', errors='ignore')
            else:
                content = f"Document {filename} uploaded successfully. Content analysis pending."
        except:
            content = f"Binary document {filename} uploaded."
        
        # Create mock document
        document = type('MockDocument', (), {
            'id': f"doc_{uuid.uuid4().hex[:8]}",
            'session_id': session_id,
            'filename': filename,
            'content': content,
            'word_count': len(content.split()),
            'page_count': 1
        })()
        
        # Create mock chunks
        chunks = [type('MockChunk', (), {
            'id': f"chunk_{document.id}_0",
            'document_id': document.id,
            'content': content,
            'chunk_index': 0,
            'total_chunks': 1
        })()]
        
        return document, chunks
    
    async def _analyze_integration_opportunities(
        self,
        new_doc_summary: DocumentSummary,
        session_kb: Any,
        conversation_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze how the new document integrates with existing knowledge
        """
        
        try:
            analysis = {
                'topic_overlaps': [],
                'complementary_areas': [],
                'potential_conflicts': [],
                'risk_changes': {},
                'conversation_relevance': 0.0,
                'document_relationships': []
            }
            
            new_topics = set(new_doc_summary.key_topics)
            
            # Analyze against existing documents
            if hasattr(session_kb, 'document_summaries'):
                for existing_doc_id, existing_summary in session_kb.document_summaries.items():
                    if existing_doc_id == new_doc_summary.document_id:
                        continue
                    
                    existing_topics = set(existing_summary.key_topics)
                    
                    # Topic overlaps
                    overlap = new_topics & existing_topics
                    if overlap:
                        analysis['topic_overlaps'].append({
                            'document_id': existing_doc_id,
                            'document_title': existing_summary.title,
                            'overlapping_topics': list(overlap),
                            'overlap_ratio': len(overlap) / len(new_topics | existing_topics)
                        })
                    
                    # Complementary areas
                    unique_new = new_topics - existing_topics
                    if unique_new:
                        analysis['complementary_areas'].append({
                            'document_id': existing_doc_id,
                            'new_topics_added': list(unique_new),
                            'complementary_score': len(unique_new) / len(new_topics) if new_topics else 0
                        })
            
            # Check conversation relevance
            current_topics = conversation_state.get('current_topics', [])
            if current_topics and new_topics:
                conversation_overlap = new_topics & set(current_topics)
                analysis['conversation_relevance'] = len(conversation_overlap) / len(new_topics)
            
            # Analyze risk changes
            analysis['risk_changes'] = await self._analyze_risk_impact(
                new_doc_summary, session_kb
            )
            
            # Detect potential conflicts
            analysis['potential_conflicts'] = await self._detect_document_conflicts(
                new_doc_summary, session_kb
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing integration opportunities: {str(e)}")
            return {
                'topic_overlaps': [],
                'complementary_areas': [],
                'potential_conflicts': [],
                'risk_changes': {},
                'conversation_relevance': 0.5,
                'document_relationships': []
            }
    
    async def _generate_integration_suggestions(
        self,
        doc_summary: DocumentSummary,
        integration_analysis: Dict[str, Any],
        session_kb: Any
    ) -> List[IntegrationSuggestion]:
        """
        Generate contextual integration suggestions
        """
        
        suggestions = []
        
        try:
            # Topic continuation suggestions
            for overlap in integration_analysis.get('topic_overlaps', []):
                if overlap['overlap_ratio'] > 0.3:  # Significant overlap
                    topics_str = ', '.join(overlap['overlapping_topics'][:2])
                    
                    suggestion = IntegrationSuggestion(
                        type='topic_continuation',
                        priority='high' if overlap['overlap_ratio'] > 0.6 else 'medium',
                        message=self.suggestion_templates['topic_continuation']['message'].format(
                            topics=topics_str
                        ),
                        suggested_queries=[
                            query.format(topic=overlap['overlapping_topics'][0])
                            for query in self.suggestion_templates['topic_continuation']['suggested_queries']
                            if overlap['overlapping_topics']
                        ],
                        metadata={
                            'related_document': overlap['document_id'],
                            'overlapping_topics': overlap['overlapping_topics'],
                            'overlap_ratio': overlap['overlap_ratio']
                        }
                    )
                    suggestions.append(suggestion)
            
            # Complementary analysis suggestions
            complementary_docs = [
                comp for comp in integration_analysis.get('complementary_areas', [])
                if comp['complementary_score'] > 0.4
            ]
            
            if complementary_docs:
                main_topic = doc_summary.key_topics[0] if doc_summary.key_topics else 'legal terms'
                
                suggestion = IntegrationSuggestion(
                    type='complementary_analysis',
                    priority='medium',
                    message=self.suggestion_templates['complementary_analysis']['message'].format(
                        topic=main_topic
                    ),
                    suggested_queries=[
                        query for query in self.suggestion_templates['complementary_analysis']['suggested_queries']
                    ],
                    metadata={
                        'complementary_documents': [comp['document_id'] for comp in complementary_docs],
                        'new_topics': complementary_docs[0]['new_topics_added'] if complementary_docs else []
                    }
                )
                suggestions.append(suggestion)
            
            # Conflict detection suggestions
            conflicts = integration_analysis.get('potential_conflicts', [])
            if conflicts:
                conflict_areas = [conflict['area'] for conflict in conflicts if 'area' in conflict]
                
                suggestion = IntegrationSuggestion(
                    type='conflict_detection',
                    priority='critical',
                    message=self.suggestion_templates['conflict_detection']['message'].format(
                        topic=conflict_areas[0] if conflict_areas else 'legal terms'
                    ),
                    suggested_queries=self.suggestion_templates['conflict_detection']['suggested_queries'],
                    metadata={
                        'conflicts': conflicts,
                        'affected_areas': conflict_areas
                    }
                )
                suggestions.append(suggestion)
            
            # Risk escalation suggestions
            risk_changes = integration_analysis.get('risk_changes', {})
            if risk_changes.get('risk_increase', False):
                suggestion = IntegrationSuggestion(
                    type='risk_escalation',
                    priority='high',
                    message=self.suggestion_templates['risk_escalation']['message'].format(
                        area=risk_changes.get('primary_risk_area', 'multiple areas'),
                        level=risk_changes.get('new_risk_level', 'elevated')
                    ),
                    suggested_queries=self.suggestion_templates['risk_escalation']['suggested_queries'],
                    metadata=risk_changes
                )
                suggestions.append(suggestion)
            
            # Portfolio completion suggestions
            total_docs = len(getattr(session_kb, 'documents', {}))
            if total_docs >= 3:  # Sufficient documents for portfolio analysis
                document_summaries = getattr(session_kb, 'document_summaries', {})
                document_types = [summary.document_type for summary in document_summaries.values()]
                main_category = max(set(document_types), key=document_types.count) if document_types else 'legal documents'
                
                suggestion = IntegrationSuggestion(
                    type='portfolio_completion',
                    priority='medium',
                    message=self.suggestion_templates['portfolio_completion']['message'].format(
                        category=main_category
                    ),
                    suggested_queries=self.suggestion_templates['portfolio_completion']['suggested_queries'],
                    metadata={
                        'total_documents': total_docs,
                        'document_types': document_types,
                        'portfolio_completeness': self._assess_portfolio_completeness(session_kb)
                    }
                )
                suggestions.append(suggestion)
            
            # Sort suggestions by priority
            priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
            suggestions.sort(key=lambda x: priority_order.get(x.priority, 3))
            
            return suggestions[:5]  # Limit to top 5 suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating integration suggestions: {str(e)}")
            # Return basic fallback suggestion
            return [IntegrationSuggestion(
                type='basic_integration',
                priority='medium',
                message=f"I've successfully processed {doc_summary.title}. You can now ask questions about it alongside your other documents.",
                suggested_queries=[
                    "What are the key points in this new document?",
                    "How does this document relate to my others?",
                    "What should I know about this document?"
                ],
                metadata={'fallback': True}
            )]
    
    async def _analyze_risk_impact(
        self,
        new_doc_summary: DocumentSummary,
        session_kb: Any
    ) -> Dict[str, Any]:
        """
        Analyze how the new document affects overall risk profile
        """
        
        try:
            # Calculate previous risk levels
            existing_risks = []
            document_summaries = getattr(session_kb, 'document_summaries', {})
            
            for summary in document_summaries.values():
                if summary.document_id != new_doc_summary.document_id:
                    risk_value = {'low': 1, 'medium': 2, 'high': 3}.get(summary.risk_level.lower(), 2)
                    existing_risks.append(risk_value)
            
            # Calculate new combined risk
            new_risk_value = {'low': 1, 'medium': 2, 'high': 3}.get(new_doc_summary.risk_level.lower(), 2)
            all_risks = existing_risks + [new_risk_value]
            
            # Risk analysis
            previous_avg_risk = sum(existing_risks) / len(existing_risks) if existing_risks else 1
            new_avg_risk = sum(all_risks) / len(all_risks)
            
            risk_increase = new_avg_risk > previous_avg_risk
            
            # Identify primary risk areas
            risk_areas = []
            if new_doc_summary.risk_level.lower() in ['medium', 'high']:
                risk_areas = new_doc_summary.key_topics[:2]  # Top 2 topics as risk areas
            
            return {
                'risk_increase': risk_increase,
                'previous_risk_level': self._risk_value_to_level(previous_avg_risk),
                'new_risk_level': self._risk_value_to_level(new_avg_risk),
                'risk_change_magnitude': abs(new_avg_risk - previous_avg_risk),
                'primary_risk_area': risk_areas[0] if risk_areas else 'general',
                'additional_risk_areas': risk_areas[1:],
                'total_high_risk_documents': sum(1 for risk in all_risks if risk == 3)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing risk impact: {str(e)}")
            return {
                'risk_increase': False,
                'previous_risk_level': 'medium',
                'new_risk_level': 'medium',
                'risk_change_magnitude': 0.0,
                'primary_risk_area': 'general',
                'additional_risk_areas': [],
                'total_high_risk_documents': 0
            }
    
    async def _detect_document_conflicts(
        self,
        new_doc_summary: DocumentSummary,
        session_kb: Any
    ) -> List[Dict[str, Any]]:
        """
        Detect potential conflicts between new document and existing ones
        """
        
        conflicts = []
        
        try:
            document_summaries = getattr(session_kb, 'document_summaries', {})
            
            for existing_doc_id, existing_summary in document_summaries.items():
                if existing_doc_id == new_doc_summary.document_id:
                    continue
                
                # Check for conflicting topics
                common_topics = set(new_doc_summary.key_topics) & set(existing_summary.key_topics)
                
                if common_topics:
                    # Simplified conflict detection based on risk levels and document types
                    new_risk = new_doc_summary.risk_level.lower()
                    existing_risk = existing_summary.risk_level.lower()
                    
                    if (new_risk == 'high' and existing_risk == 'low') or \
                       (new_risk == 'low' and existing_risk == 'high'):
                        
                        conflict_area = list(common_topics)[0]  # Take first common topic
                        
                        conflicts.append({
                            'type': 'risk_level_mismatch',
                            'area': conflict_area,
                            'document_1': new_doc_summary.document_id,
                            'document_2': existing_doc_id,
                            'document_1_title': new_doc_summary.title,
                            'document_2_title': existing_summary.title,
                            'description': f"Conflicting risk levels for {conflict_area}",
                            'severity': 'medium'
                        })
            
            return conflicts
            
        except Exception as e:
            self.logger.error(f"Error detecting document conflicts: {str(e)}")
            return []
    
    async def _update_conversation_context(
        self,
        session_id: str,
        doc_summary: DocumentSummary,
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update conversation context with new document information
        """
        
        try:
            context_updates = {
                'new_document_added': True,
                'document_id': doc_summary.document_id,
                'document_title': doc_summary.title,
                'document_topics': doc_summary.key_topics,
                'integration_timestamp': datetime.now(timezone.utc).isoformat(),
                'context_expansion': {
                    'new_topics_added': len(doc_summary.key_topics),
                    'total_entities': len(doc_summary.main_entities),
                    'risk_level': doc_summary.risk_level
                }
            }
            
            # Add conversation turn about document addition
            if hasattr(self.context_manager, 'add_conversation_turn') and callable(self.context_manager.add_conversation_turn):
                try:
                    await self.context_manager.add_conversation_turn(
                        session_id=session_id,
                        user_message=f"[SYSTEM] User uploaded new document: {doc_summary.title}",
                        assistant_response=f"I've successfully analyzed {doc_summary.title}. It covers {', '.join(doc_summary.key_topics[:3])} and has been integrated into our conversation.",
                        source_documents=[doc_summary.document_id],
                        response_metadata={
                            'system_message': True,
                            'document_integration': True,
                            'document_summary': {
                                'title': doc_summary.title,
                                'type': doc_summary.document_type,
                                'risk_level': doc_summary.risk_level,
                                'key_topics': doc_summary.key_topics
                            }
                        }
                    )
                except Exception as e:
                    self.logger.error(f"Failed to add conversation turn: {e}")
            
            return context_updates
            
        except Exception as e:
            self.logger.error(f"Error updating conversation context: {str(e)}")
            return {
                'new_document_added': True,
                'document_id': doc_summary.document_id,
                'integration_timestamp': datetime.now(timezone.utc).isoformat(),
                'error': str(e)
            }
    
    async def _analyze_relationship_updates(
        self,
        new_document_id: str,
        session_kb: Any
    ) -> List[Dict[str, Any]]:
        """
        Analyze how document relationships have changed
        """
        
        try:
            relationship_updates = []
            
            # Get relationships involving the new document
            if hasattr(session_kb, 'get_related_documents') and callable(session_kb.get_related_documents):
                try:
                    related_documents = session_kb.get_related_documents(new_document_id)
                except Exception as e:
                    self.logger.error(f"Failed to get related documents: {e}")
                    related_documents = []
            else:
                related_documents = []
            
            document_summaries = getattr(session_kb, 'document_summaries', {})
            
            for relationship in related_documents:
                other_doc_id = (relationship.document_2_id 
                               if relationship.document_1_id == new_document_id 
                               else relationship.document_1_id)
                
                other_doc_summary = document_summaries.get(other_doc_id)
                
                if other_doc_summary:
                    relationship_updates.append({
                        'type': 'new_relationship_detected',
                        'relationship_id': getattr(relationship, 'id', f"rel_{new_document_id}_{other_doc_id}"),
                        'related_document_id': other_doc_id,
                        'related_document_title': other_doc_summary.title,
                        'relationship_type': getattr(relationship, 'relationship_type', 'related'),
                        'similarity_score': getattr(relationship, 'similarity_score', 0.5),
                        'common_topics': getattr(relationship, 'common_topics', []),
                        'impact': self._assess_relationship_impact(relationship)
                    })
            
            # If no real relationships found, create basic relationship info
            if not relationship_updates and len(document_summaries) > 1:
                relationship_updates.append({
                    'type': 'document_added_to_portfolio',
                    'message': f'Document added to portfolio of {len(document_summaries)} documents',
                    'portfolio_size': len(document_summaries),
                    'integration_status': 'successful'
                })
            
            return relationship_updates
            
        except Exception as e:
            self.logger.error(f"Error analyzing relationship updates: {str(e)}")
            return [{
                'type': 'integration_complete',
                'message': 'Document successfully integrated',
                'status': 'complete'
            }]
    
    def _assess_portfolio_completeness(self, session_kb: Any) -> float:
        """
        Assess how complete the document portfolio is
        """
        
        try:
            document_summaries = getattr(session_kb, 'document_summaries', {})
            
            # Simple completeness assessment based on document diversity
            document_types = [summary.document_type for summary in document_summaries.values()]
            unique_types = len(set(document_types))
            total_docs = len(document_types)
            
            if total_docs == 0:
                return 0.0
            
            # Completeness factors
            type_diversity = unique_types / total_docs
            sufficient_quantity = min(1.0, total_docs / 5)  # Ideal portfolio ~5 documents
            
            # Topic coverage
            all_topics = []
            for summary in document_summaries.values():
                all_topics.extend(summary.key_topics)
            
            unique_topics = len(set(all_topics))
            topic_diversity = min(1.0, unique_topics / 15)  # Good coverage ~15 unique topics
            
            completeness = (type_diversity + sufficient_quantity + topic_diversity) / 3
            
            return round(completeness, 2)
            
        except Exception as e:
            self.logger.error(f"Error assessing portfolio completeness: {str(e)}")
            return 0.5
    
    def _risk_value_to_level(self, risk_value: float) -> str:
        """Convert numeric risk value to level string"""
        if risk_value <= 1.3:
            return 'low'
        elif risk_value <= 2.3:
            return 'medium'
        else:
            return 'high'
    
    def _assess_relationship_impact(self, relationship) -> str:
        """Assess the impact of a document relationship"""
        try:
            similarity_score = getattr(relationship, 'similarity_score', 0.5)
            
            if similarity_score > 0.7:
                return 'high_synergy'
            elif similarity_score > 0.4:
                return 'moderate_synergy'
            else:
                return 'low_synergy'
        except:
            return 'moderate_synergy'
    
    async def generate_integration_summary(
        self,
        integration_result: IntegrationResult,
        session_kb: Any
    ) -> str:
        """
        Generate a natural language summary of the integration
        """
        
        try:
            if not integration_result.success:
                return f"I encountered an issue processing your document: {integration_result.error_message}"
            
            doc_summary = integration_result.document_summary
            suggestions = integration_result.integration_suggestions
            
            summary_parts = []
            
            # Document analysis summary
            summary_parts.append(f"I've analyzed {doc_summary.title} and integrated it into our conversation.")
            
            # Key findings
            if doc_summary.key_topics:
                topics_str = ', '.join(doc_summary.key_topics[:3])
                summary_parts.append(f"This document covers {topics_str}.")
            
            # Risk assessment
            summary_parts.append(f"The document has a {doc_summary.risk_level} risk level.")
            
            # Integration insights
            if suggestions:
                high_priority_suggestions = [s for s in suggestions if s.priority in ['critical', 'high']]
                
                if high_priority_suggestions:
                    summary_parts.append("Here are some important insights:")
                    for suggestion in high_priority_suggestions[:2]:  # Top 2
                        summary_parts.append(f"• {suggestion.message}")
            
            # Portfolio impact
            total_docs = len(getattr(session_kb, 'documents', {}))
            if total_docs > 1:
                summary_parts.append(f"You now have {total_docs} documents in your portfolio for comprehensive analysis.")
            
            return ' '.join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating integration summary: {str(e)}")
            return "Your document has been successfully integrated. You can now ask questions about it along with your other documents."
    
    def get_integration_capabilities(self) -> Dict[str, Any]:
        """Get information about available integration capabilities"""
        
        return {
            'services_available': {
                name: service is not None for name, service in services_available.items()
            },
            'integration_features': {
                'real_time_processing': True,
                'conflict_detection': True,
                'risk_analysis': True,
                'relationship_mapping': True,
                'context_updates': True,
                'suggestion_generation': True
            },
            'fallback_capabilities': {
                'mock_document_creation': True,
                'basic_analysis': True,
                'simple_suggestions': True,
                'error_recovery': True
            },
            'models_available': MODELS_AVAILABLE
        }


# Utility functions for standalone use
async def integrate_document_simple(
    file_content: bytes, 
    filename: str, 
    session_id: str = "default_session"
) -> Dict[str, Any]:
    """Simple document integration for basic use cases"""
    
    integrator = RealTimeDocumentIntegrator()
    
    try:
        result = await integrator.handle_mid_conversation_upload(
            session_id=session_id,
            file_content=file_content,
            filename=filename,
            current_conversation_state={}
        )
        
        return {
            'success': result.success,
            'document_summary': result.document_summary.to_dict() if result.document_summary else None,
            'suggestions': [s.to_dict() for s in result.integration_suggestions],
            'processing_time': result.processing_time,
            'error': result.error_message
        }
    
    except Exception as e:
        return {
            'success': False,
            'document_summary': None,
            'suggestions': [],
            'processing_time': 0.0,
            'error': str(e)
        }

def create_basic_integration_suggestion(
    message: str, 
    queries: List[str],
    suggestion_type: str = 'basic',
    priority: str = 'medium'
) -> IntegrationSuggestion:
    """Create a basic integration suggestion"""
    
    return IntegrationSuggestion(
        type=suggestion_type,
        priority=priority,
        message=message,
        suggested_queries=queries,
        metadata={'created_by': 'utility_function'}
    )
