"""
Enhanced document chunking with reliable context preservation
Simplified and robust implementation without external dependencies
"""

import re
import asyncio
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone

from config.logging import get_logger

logger = get_logger(__name__)

@dataclass
class StandardDocument:
    """Document model for chunking"""
    id: str
    content: str
    page_count: int = 1
    word_count: int = 0
    
    def __post_init__(self):
        if self.word_count == 0:
            self.word_count = len(self.content.split())

@dataclass 
class DocumentChunk:
    """Document chunk model"""
    id: str
    document_id: str
    chunk_index: int
    total_chunks: int
    content: str
    page_range: str
    char_range: str
    word_count: int
    metadata: Dict[str, Any]
    created_at: datetime

@dataclass
class ChunkBoundary:
    """Represents a potential chunk boundary"""
    position: int
    score: float
    boundary_type: str
    context: str
    quality: float = 0.5

class EnhancedChunker:
    """
    Enhanced document chunking with semantic awareness
    """
    
    def __init__(self, max_pages_per_chunk: int = 15, max_chars_per_chunk: int = 8000):
        self.max_pages_per_chunk = max_pages_per_chunk
        self.max_chars_per_chunk = max_chars_per_chunk
        self.min_chars_per_chunk = 500  # Minimum viable chunk size
        self.chunk_overlap_chars = 200  # Character overlap between chunks
        self.logger = logger
        
        # Enhanced patterns for identifying semantic boundaries
        self.high_priority_patterns = [
            # Document structure patterns
            (r'\n\s*(?:CHAPTER|Chapter)\s+[\d\w]+[\.\:\-\s]', 1.0, 'chapter'),
            (r'\n\s*(?:SECTION|Section)\s+[\d\w]+[\.\:\-\s]', 0.95, 'section'),
            (r'\n\s*(?:PART|Part)\s+[\d\w]+[\.\:\-\s]', 0.95, 'part'),
            (r'\n\s*(?:ARTICLE|Article)\s+[\d\w]+[\.\:\-\s]', 0.9, 'article'),
            
            # Academic/Educational patterns
            (r'\n\s*(?:Lesson|LESSON)\s+[\d\w]+[\.\:\-\s]', 0.9, 'lesson'),
            (r'\n\s*(?:Week|WEEK)\s+[\d\w]+[\.\:\-\s]', 0.85, 'week'),
            (r'\n\s*(?:Module|MODULE)\s+[\d\w]+[\.\:\-\s]', 0.9, 'module'),
            (r'\n\s*(?:Unit|UNIT)\s+[\d\w]+[\.\:\-\s]', 0.85, 'unit'),
            
            # Legal document patterns
            (r'\n\s*(?:WHEREAS|NOW THEREFORE|IN WITNESS WHEREOF)', 0.9, 'legal_clause'),
            (r'\n\s*(?:RECITALS|DEFINITIONS|TERMS AND CONDITIONS)', 0.95, 'legal_section'),
            
            # Numbered sections
            (r'\n\s*\d+\.\s+[A-Z]', 0.8, 'numbered_section'),
            (r'\n\s*\d+\.\d+\s+[A-Z]', 0.75, 'subsection'),
        ]
        
        self.medium_priority_patterns = [
            # Headers and subheaders
            (r'\n\s*[A-Z][A-Z\s]{5,30}\n', 0.7, 'caps_header'),
            (r'\n\s*[A-Za-z].*:\s*\n', 0.65, 'colon_header'),
            
            # Paragraph breaks with lettered subsections
            (r'\n\s*\([a-z]\)\s+', 0.6, 'lettered_subsection'),
            (r'\n\s*[a-z]\.\s+', 0.55, 'lettered_item'),
            
            # Double line breaks (paragraph boundaries)
            (r'\n\s*\n\s*\n', 0.5, 'paragraph_break'),
        ]
        
        self.low_priority_patterns = [
            # Sentence boundaries (emergency splitting)
            (r'[.!?]+\s+[A-Z]', 0.3, 'sentence'),
            (r'[.!?]+\n\s*[A-Z]', 0.35, 'sentence_newline'),
        ]
        
        self.logger.info("EnhancedChunker initialized successfully")
    
    async def create_chunks(self, document: StandardDocument) -> List[DocumentChunk]:
        """
        Create intelligent chunks from document
        
        Args:
            document: StandardDocument to chunk
            
        Returns:
            List of DocumentChunk objects
        """
        
        try:
            self.logger.info(f"Creating chunks for document: {document.id}")
            
            # Validate document content
            if not document.content or len(document.content.strip()) == 0:
                raise ValueError("Document content is empty")
            
            # Clean and normalize content
            content = self._normalize_content(document.content)
            document.content = content
            
            # Check if document needs chunking
            if not self._needs_chunking(document):
                chunks = [await self._create_single_chunk(document)]
            else:
                chunks = await self._create_intelligent_chunks(document)
            
            # Post-process and validate chunks
            chunks = self._post_process_chunks(chunks)
            
            self.logger.info(f"Created {len(chunks)} chunks for document {document.id}")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error creating chunks for document {document.id}: {str(e)}")
            # Create emergency single chunk
            return [self._create_emergency_chunk(document, str(e))]
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content for better chunking"""
        
        # Remove excessive whitespace while preserving structure
        lines = content.split('\n')
        normalized_lines = []
        
        for line in lines:
            # Clean each line but preserve empty lines for structure
            cleaned_line = ' '.join(line.split())  # Remove extra spaces
            normalized_lines.append(cleaned_line)
        
        # Join back and handle multiple consecutive empty lines
        normalized = '\n'.join(normalized_lines)
        
        # Replace multiple consecutive newlines with double newlines
        normalized = re.sub(r'\n\s*\n\s*\n+', '\n\n\n', normalized)
        
        return normalized.strip()
    
    def create_chunks_from_text(self, text: str, document_id: str = None, 
                               estimated_pages: int = None) -> List[DocumentChunk]:
        """
        Create chunks directly from text (synchronous version)
        
        Args:
            text: Raw text content
            document_id: Document identifier
            estimated_pages: Estimated page count
            
        Returns:
            List of DocumentChunk objects
        """
        
        if document_id is None:
            document_id = f"doc_{uuid.uuid4().hex[:8]}"
        
        # Estimate pages if not provided
        if estimated_pages is None:
            estimated_pages = max(1, len(text) // 2500)  # ~2500 chars per page
        
        # Create temporary document object
        document = StandardDocument(
            id=document_id,
            content=text,
            page_count=estimated_pages,
            word_count=len(text.split())
        )
        
        # Use async version but run synchronously
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.create_chunks(document))
                    return future.result()
            else:
                return loop.run_until_complete(self.create_chunks(document))
        except RuntimeError:
            # No event loop running, create new one
            return asyncio.run(self.create_chunks(document))
    
    def _needs_chunking(self, document: StandardDocument) -> bool:
        """Determine if document needs to be chunked"""
        return (
            document.page_count > self.max_pages_per_chunk or
            len(document.content) > self.max_chars_per_chunk
        )
    
    async def _create_single_chunk(self, document: StandardDocument) -> DocumentChunk:
        """Create single chunk for small documents"""
        
        return DocumentChunk(
            id=f"{document.id}_chunk_0",
            document_id=document.id,
            chunk_index=0,
            total_chunks=1,
            content=document.content,
            page_range=f"1-{document.page_count}",
            char_range=f"0-{len(document.content)}",
            word_count=document.word_count,
            metadata={
                'is_complete_document': True,
                'chunk_type': 'complete',
                'boundary_info': {
                    'start_boundary': 'document_start',
                    'end_boundary': 'document_end'
                },
                'quality_metrics': {
                    'completeness': 1.0,
                    'semantic_integrity': 1.0,
                    'context_preservation': 1.0,
                    'readability_score': 0.8
                }
            },
            created_at=datetime.now(timezone.utc)
        )
    
    async def _create_intelligent_chunks(self, document: StandardDocument) -> List[DocumentChunk]:
        """Create multiple chunks with intelligent boundaries"""
        
        content = document.content
        content_length = len(content)
        
        # Find all potential boundaries
        boundaries = self._find_all_boundaries(content)
        
        if not boundaries:
            # Fallback to simple chunking if no boundaries found
            return await self._create_simple_chunks(document)
        
        # Select optimal boundaries
        selected_boundaries = self._select_optimal_boundaries(boundaries, content_length)
        
        # Create chunks from selected boundaries
        chunks = []
        current_pos = 0
        
        for i, boundary in enumerate(selected_boundaries):
            # Calculate chunk boundaries with overlap
            start_pos = max(0, current_pos - (self.chunk_overlap_chars if i > 0 else 0))
            end_pos = min(content_length, boundary.position + (self.chunk_overlap_chars if i < len(selected_boundaries) - 1 else 0))
            
            chunk_content = content[start_pos:end_pos].strip()
            
            # Skip empty chunks
            if not chunk_content or len(chunk_content) < self.min_chars_per_chunk // 2:
                continue
            
            # Calculate page range (approximate)
            page_start = max(1, int((current_pos / content_length) * document.page_count) + 1)
            page_end = min(document.page_count, int((boundary.position / content_length) * document.page_count) + 1)
            
            # Ensure page_end is at least page_start
            if page_end < page_start:
                page_end = page_start
            
            # Create chunk
            chunk = self._create_chunk(
                document_id=document.id,
                chunk_index=i,
                total_chunks=len(selected_boundaries),
                content=chunk_content,
                page_range=f"{page_start}-{page_end}",
                char_range=f"{start_pos}-{end_pos}",
                boundary_info={
                    'start_boundary': 'document_start' if i == 0 else selected_boundaries[i-1].boundary_type,
                    'end_boundary': boundary.boundary_type,
                    'start_context': '' if i == 0 else selected_boundaries[i-1].context,
                    'end_context': boundary.context,
                    'boundary_score': boundary.score
                },
                chunk_type='intelligent'
            )
            
            chunks.append(chunk)
            current_pos = boundary.position
        
        return chunks
    
    def _find_all_boundaries(self, content: str) -> List[ChunkBoundary]:
        """Find all potential chunk boundaries"""
        
        boundaries = []
        
        # Process high priority patterns
        for pattern, score, boundary_type in self.high_priority_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE):
                boundary = ChunkBoundary(
                    position=match.start(),
                    score=score,
                    boundary_type=boundary_type,
                    context=self._extract_context(content, match.start(), match.end()),
                    quality=score
                )
                boundaries.append(boundary)
        
        # Process medium priority patterns
        for pattern, score, boundary_type in self.medium_priority_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                boundary = ChunkBoundary(
                    position=match.start(),
                    score=score,
                    boundary_type=boundary_type,
                    context=self._extract_context(content, match.start(), match.end()),
                    quality=score
                )
                boundaries.append(boundary)
        
        # Process low priority patterns only if needed
        if len(boundaries) < 3 and len(content) > self.max_chars_per_chunk * 2:
            for pattern, score, boundary_type in self.low_priority_patterns:
                for match in re.finditer(pattern, content):
                    boundary = ChunkBoundary(
                        position=match.start(),
                        score=score,
                        boundary_type=boundary_type,
                        context=self._extract_context(content, match.start(), match.end()),
                        quality=score
                    )
                    boundaries.append(boundary)
        
        # Remove duplicates and sort
        unique_boundaries = {}
        for boundary in boundaries:
            key = boundary.position
            if key not in unique_boundaries or boundary.score > unique_boundaries[key].score:
                unique_boundaries[key] = boundary
        
        sorted_boundaries = sorted(unique_boundaries.values(), key=lambda x: x.position)
        
        return sorted_boundaries
    
    def _extract_context(self, content: str, start: int, end: int) -> str:
        """Extract context around a boundary"""
        context_window = 50
        context_start = max(0, start - context_window)
        context_end = min(len(content), end + context_window)
        
        context = content[context_start:context_end]
        context = ' '.join(context.split())  # Normalize whitespace
        
        # Truncate if too long
        if len(context) > 100:
            context = context[:97] + '...'
        
        return context
    
    def _select_optimal_boundaries(self, boundaries: List[ChunkBoundary], content_length: int) -> List[ChunkBoundary]:
        """Select optimal boundaries for chunking"""
        
        if not boundaries:
            # Create artificial boundary at the end
            return [ChunkBoundary(
                position=content_length,
                score=1.0,
                boundary_type='document_end',
                context='[End of Document]',
                quality=1.0
            )]
        
        selected = []
        current_pos = 0
        
        for boundary in boundaries:
            chunk_size = boundary.position - current_pos
            
            # Select boundary if chunk would be of reasonable size
            if chunk_size >= self.min_chars_per_chunk:
                # Calculate combined score
                size_factor = min(1.0, chunk_size / self.max_chars_per_chunk)
                combined_score = (boundary.score * 0.7) + (size_factor * 0.3)
                
                # Accept if score is good or chunk is getting too large
                if combined_score >= 0.6 or chunk_size > self.max_chars_per_chunk * 1.2:
                    selected.append(boundary)
                    current_pos = boundary.position
        
        # Ensure we have a final boundary
        if not selected or selected[-1].position < content_length:
            selected.append(ChunkBoundary(
                position=content_length,
                score=1.0,
                boundary_type='document_end',
                context='[End of Document]',
                quality=1.0
            ))
        
        return selected
    
    async def _create_simple_chunks(self, document: StandardDocument) -> List[DocumentChunk]:
        """Create chunks using simple size-based splitting"""
        
        content = document.content
        content_length = len(content)
        
        chunks = []
        chunk_count = max(1, (content_length + self.max_chars_per_chunk - 1) // self.max_chars_per_chunk)
        
        for i in range(chunk_count):
            start_pos = i * self.max_chars_per_chunk
            end_pos = min((i + 1) * self.max_chars_per_chunk, content_length)
            
            # Try to find better boundary near the end
            if i < chunk_count - 1:  # Not the last chunk
                search_start = max(start_pos, end_pos - 300)
                search_end = min(content_length, end_pos + 100)
                
                # Look for sentence boundaries first
                for pattern in [r'[.!?]+\s+[A-Z]', r'[.!?]+\n', r'\n\s*\n']:
                    for match in re.finditer(pattern, content[search_start:search_end]):
                        better_end = search_start + match.start()
                        if abs(better_end - end_pos) < 200:
                            end_pos = better_end
                            break
                    if end_pos != min((i + 1) * self.max_chars_per_chunk, content_length):
                        break
            
            chunk_content = content[start_pos:end_pos].strip()
            
            if not chunk_content:
                continue
            
            # Calculate page range
            page_start = max(1, int((start_pos / content_length) * document.page_count) + 1)
            page_end = min(document.page_count, int((end_pos / content_length) * document.page_count) + 1)
            
            chunk = self._create_chunk(
                document_id=document.id,
                chunk_index=i,
                total_chunks=chunk_count,
                content=chunk_content,
                page_range=f"{page_start}-{page_end}",
                char_range=f"{start_pos}-{end_pos}",
                boundary_info={
                    'start_boundary': 'size_based',
                    'end_boundary': 'size_based'
                },
                chunk_type='simple'
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, document_id: str, chunk_index: int, total_chunks: int,
                     content: str, page_range: str, char_range: str,
                     boundary_info: Dict[str, Any], chunk_type: str) -> DocumentChunk:
        """Create a single chunk with metadata"""
        
        word_count = len(content.split())
        quality_metrics = self._analyze_chunk_quality(content, chunk_index, total_chunks)
        
        return DocumentChunk(
            id=f"{document_id}_chunk_{chunk_index}",
            document_id=document_id,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            content=content,
            page_range=page_range,
            char_range=char_range,
            word_count=word_count,
            metadata={
                'is_complete_document': total_chunks == 1,
                'chunk_type': chunk_type,
                'boundary_info': boundary_info,
                'quality_metrics': quality_metrics,
                'semantic_indicators': self._extract_semantic_indicators(content)
            },
            created_at=datetime.now(timezone.utc)
        )
    
    def _analyze_chunk_quality(self, content: str, chunk_index: int, total_chunks: int) -> Dict[str, Any]:
        """Analyze quality metrics for a chunk"""
        
        issues = []
        word_count = len(content.split())
        
        # Base quality score
        quality_score = 0.8
        
        # Check content length
        if word_count < 50:
            quality_score -= 0.2
            issues.append('very_short_content')
        elif word_count > 3000:
            quality_score -= 0.1
            issues.append('very_long_content')
        
        # Check for complete sentences
        sentences = re.split(r'[.!?]+', content.strip())
        complete_sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) >= 3]
        completeness = len(complete_sentences) / max(1, len(sentences)) if sentences else 0
        
        # Check for abrupt endings
        if not content.strip().endswith(('.', '!', '?', ';', ':')) and chunk_index < total_chunks - 1:
            quality_score -= 0.1
            issues.append('incomplete_ending')
        
        # Check for context indicators
        context_indicators = ['section', 'chapter', 'part', 'lesson', 'module', 'unit']
        has_context = any(indicator.lower() in content.lower() for indicator in context_indicators)
        
        semantic_integrity = 0.9 if has_context else 0.7
        context_preservation = 0.8 if has_context else 0.6
        
        return {
            'quality_score': max(0.0, min(1.0, quality_score)),
            'completeness': completeness,
            'semantic_integrity': semantic_integrity,
            'context_preservation': context_preservation,
            'readability_score': self._calculate_readability(content),
            'issues': issues,
            'word_count': word_count,
            'has_context_indicators': has_context
        }
    
    def _calculate_readability(self, content: str) -> float:
        """Calculate simple readability score"""
        
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.5
        
        words = content.split()
        avg_sentence_length = len(words) / len(sentences)
        
        # Simple readability based on sentence length
        if avg_sentence_length <= 15:
            return 0.9  # High readability
        elif avg_sentence_length <= 25:
            return 0.7  # Medium readability
        elif avg_sentence_length <= 35:
            return 0.5  # Low readability
        else:
            return 0.3  # Very low readability
    
    def _extract_semantic_indicators(self, content: str) -> Dict[str, Any]:
        """Extract semantic indicators from chunk content"""
        
        content_lower = content.lower()
        
        # Document structure indicators
        structure_indicators = ['chapter', 'section', 'part', 'article', 'lesson', 'module', 'unit', 'week']
        found_structures = [indicator for indicator in structure_indicators if indicator in content_lower]
        
        # Academic indicators
        academic_indicators = ['objective', 'learning', 'assignment', 'exercise', 'quiz', 'test', 'homework']
        found_academic = [indicator for indicator in academic_indicators if indicator in content_lower]
        
        # Legal indicators
        legal_indicators = ['shall', 'must', 'agreement', 'contract', 'clause', 'whereas', 'therefore']
        found_legal = [indicator for indicator in legal_indicators if indicator in content_lower]
        
        # Determine content type
        content_type = 'general'
        if found_academic:
            content_type = 'academic'
        elif found_legal:
            content_type = 'legal'
        elif found_structures:
            content_type = 'structured'
        
        # Check for lists and enumerations
        has_numbered_list = bool(re.search(r'\n\s*\d+\.', content))
        has_lettered_list = bool(re.search(r'\n\s*\([a-z]\)', content))
        has_bullet_list = bool(re.search(r'\n\s*[-•*]', content))
        
        return {
            'content_type': content_type,
            'structure_indicators': found_structures,
            'academic_indicators': found_academic[:5],  # Limit to avoid huge lists
            'legal_indicators': found_legal[:5],
            'has_lists': {
                'numbered': has_numbered_list,
                'lettered': has_lettered_list,
                'bulleted': has_bullet_list
            },
            'estimated_complexity': self._estimate_complexity(content_lower)
        }
    
    def _estimate_complexity(self, content_lower: str) -> str:
        """Estimate content complexity"""
        
        complex_indicators = ['however', 'nevertheless', 'furthermore', 'therefore', 'consequently']
        technical_indicators = ['algorithm', 'implementation', 'methodology', 'analysis', 'framework']
        
        complex_count = sum(1 for indicator in complex_indicators if indicator in content_lower)
        technical_count = sum(1 for indicator in technical_indicators if indicator in content_lower)
        
        if technical_count >= 3 or complex_count >= 4:
            return 'high'
        elif technical_count >= 1 or complex_count >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _post_process_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Post-process chunks to ensure quality"""
        
        if not chunks:
            return chunks
        
        processed_chunks = []
        
        # Merge very small chunks
        i = 0
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # Check if current chunk is too small and can be merged
            if (current_chunk.word_count < 50 and i < len(chunks) - 1 and 
                current_chunk.word_count + chunks[i + 1].word_count < self.max_chars_per_chunk):
                
                # Merge with next chunk
                next_chunk = chunks[i + 1]
                merged_content = current_chunk.content + "\n\n" + next_chunk.content
                
                merged_chunk = self._create_chunk(
                    document_id=current_chunk.document_id,
                    chunk_index=len(processed_chunks),
                    total_chunks=0,  # Will be updated later
                    content=merged_content,
                    page_range=f"{current_chunk.page_range.split('-')[0]}-{next_chunk.page_range.split('-')[-1]}",
                    char_range=f"{current_chunk.char_range.split('-')[0]}-{next_chunk.char_range.split('-')[-1]}",
                    boundary_info={
                        'start_boundary': current_chunk.metadata['boundary_info']['start_boundary'],
                        'end_boundary': next_chunk.metadata['boundary_info']['end_boundary'],
                        'merged_from': [current_chunk.id, next_chunk.id]
                    },
                    chunk_type='merged'
                )
                
                processed_chunks.append(merged_chunk)
                i += 2  # Skip next chunk as it's been merged
            else:
                current_chunk.chunk_index = len(processed_chunks)
                processed_chunks.append(current_chunk)
                i += 1
        
        # Update total chunks count and chunk IDs
        total_chunks = len(processed_chunks)
        for i, chunk in enumerate(processed_chunks):
            chunk.chunk_index = i
            chunk.total_chunks = total_chunks
            chunk.id = f"{chunk.document_id}_chunk_{i}"
        
        return processed_chunks
    
    def _create_emergency_chunk(self, document: StandardDocument, error_msg: str) -> DocumentChunk:
        """Create emergency chunk when all else fails"""
        
        # Truncate content if too long
        content = document.content
        if len(content) > self.max_chars_per_chunk:
            content = content[:self.max_chars_per_chunk]
            truncated = True
        else:
            truncated = False
        
        return DocumentChunk(
            id=f"{document.id}_emergency_chunk",
            document_id=document.id,
            chunk_index=0,
            total_chunks=1,
            content=content,
            page_range="1-1",
            char_range=f"0-{len(content)}",
            word_count=len(content.split()),
            metadata={
                'is_complete_document': not truncated,
                'chunk_type': 'emergency',
                'error_info': {
                    'error_message': error_msg,
                    'truncated': truncated
                },
                'quality_metrics': {
                    'quality_score': 0.3,
                    'completeness': 0.5 if not truncated else 0.3,
                    'semantic_integrity': 0.3,
                    'context_preservation': 0.3
                }
            },
            created_at=datetime.now(timezone.utc)
        )


def chunk_text_simple(text: str, max_chunk_size: int = 8000, document_id: str = None) -> List[Dict[str, Any]]:
    """
    Simple function to chunk text without complex dependencies
    
    Args:
        text: Text to chunk
        max_chunk_size: Maximum characters per chunk
        document_id: Document identifier
        
    Returns:
        List of simple chunk dictionaries
    """
    
    if document_id is None:
        document_id = f"doc_{uuid.uuid4().hex[:8]}"
    
    if len(text) <= max_chunk_size:
        return [{
            'id': f"{document_id}_chunk_0",
            'content': text,
            'chunk_index': 0,
            'total_chunks': 1,
            'word_count': len(text.split()),
            'char_count': len(text),
            'quality_score': 1.0
        }]
    
    chunks = []
    chunk_count = (len(text) + max_chunk_size - 1) // max_chunk_size
    
    for i in range(chunk_count):
        start_pos = i * max_chunk_size
        end_pos = min((i + 1) * max_chunk_size, len(text))
        
        # Try to find better boundary
        if i < chunk_count - 1:
            search_start = max(start_pos, end_pos - 200)
            search_end = min(len(text), end_pos + 100)
            
            # Look for sentence boundaries
            for pattern in [r'[.!?]+\s+', r'\n\s*\n']:
                for match in re.finditer(pattern, text[search_start:search_end]):
                    better_end = search_start + match.end()
                    if abs(better_end - end_pos) < 150:
                        end_pos = better_end
                        break
                if end_pos != min((i + 1) * max_chunk_size, len(text)):
                    break
        
        chunk_content = text[start_pos:end_pos].strip()
        
        if not chunk_content:
            continue
        
        chunks.append({
            'id': f"{document_id}_chunk_{i}",
            'content': chunk_content,
            'chunk_index': i,
            'total_chunks': chunk_count,
            'word_count': len(chunk_content.split()),
            'char_count': len(chunk_content),
            'char_range': f"{start_pos}-{end_pos}",
            'quality_score': 0.7
        })
    
    return chunks