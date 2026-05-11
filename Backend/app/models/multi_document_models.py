"""
Multi-Document Models for Advanced Features
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class DocumentRelationship(BaseModel):
    """Document relationship model"""
    source_document_id: str
    target_document_id: str
    relationship_type: str  # 'conflict', 'complement', 'reference'
    confidence_score: float = Field(ge=0.0, le=1.0)
    description: str

class CrossDocumentInsight(BaseModel):
    """Cross-document insight model"""
    insight_id: str
    document_ids: List[str]
    insight_type: str  # 'pattern', 'conflict', 'opportunity'
    summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: datetime

class MultiDocumentContext(BaseModel):
    """Context for multi-document operations"""
    session_id: str
    document_ids: List[str]
    relationships: List[DocumentRelationship] = []
    insights: List[CrossDocumentInsight] = []
    context_summary: Optional[str] = None

class DocumentCluster(BaseModel):
    """Document clustering model"""
    cluster_id: str
    document_ids: List[str]
    cluster_type: str  # 'similar', 'related', 'contradictory'
    similarity_score: float = Field(ge=0.0, le=1.0)
    representative_terms: List[str] = []

class MultiDocumentAnalysis(BaseModel):
    """Comprehensive multi-document analysis"""
    analysis_id: str
    session_id: str
    document_count: int
    relationships: List[DocumentRelationship]
    clusters: List[DocumentCluster]
    insights: List[CrossDocumentInsight]
    overall_risk_score: float = Field(ge=0.0, le=10.0)
    recommendations: List[str]
    created_at: datetime
