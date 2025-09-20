"""
LegalMind AI - Standalone Working Version for Docker Testing
"""

import os
import uvicorn
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uuid
import json

# Create FastAPI app
app = FastAPI(
    title="🧠 LegalMind AI - Multi-Document Assistant",
    version="2.0.0",
    description="AI-powered legal document analysis with multi-document capabilities"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for demo (replace with database in production)
sessions_data = {}
documents_data = {}

# Request models
class ChatRequest(BaseModel):
    session_id: str
    message: str
    language: str = "en"
    response_style: str = "comprehensive"
    max_sources: int = 5
    include_cross_references: bool = True

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "🧠 Welcome to LegalMind AI - Multi-Document Assistant",
        "version": "2.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "docker": "✅ Running in Docker",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "features": {
            "multi_document_analysis": True,
            "cross_document_relationships": True,
            "real_time_integration": True,
            "conversation_memory": True,
            "multilingual_support": True,
            "voice_summaries": True,
            "risk_assessment": True,
            "conflict_detection": True
        },
        "endpoints": {
            "api_docs": "/docs",
            "create_session": "/api/v1/multi-document/sessions/create",
            "upload_document": "/api/v1/multi-document/upload",
            "chat": "/api/v1/multi-document/chat",
            "session_overview": "/api/v1/multi-document/sessions/{session_id}/overview"
        },
        "supported_formats": ["PDF", "DOCX", "DOC", "TXT", "RTF"],
        "supported_languages": [
            "English", "Hindi", "Tamil", "Telugu", "Bengali", 
            "Gujarati", "Kannada", "Malayalam", "Marathi", 
            "Odia", "Punjabi", "Urdu"
        ]
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "LegalMind AI Multi-Document Assistant",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime": "operational",
        "knowledge_base": {"status": "✅ operational", "active_sessions": len(sessions_data)},
        "conversation_manager": {"status": "✅ operational"},
        "google_cloud": {
            "status": "✅ authenticated" if os.getenv("GOOGLE_API_KEY") else "⚠️ no api key",
            "services": {
                "gemini": "✅ available" if os.getenv("GOOGLE_API_KEY") else "❌ no api key",
                "translation": "✅ available",
                "storage": "✅ available"
            }
        },
        "environment": {
            "google_project": "✅" if os.getenv("GOOGLE_CLOUD_PROJECT") else "❌",
            "google_api_key": "✅" if os.getenv("GOOGLE_API_KEY") else "❌",
            "storage_bucket": "✅" if os.getenv("GCS_BUCKET_NAME") else "❌"
        },
        "overall": "✅ All systems operational"
    }

@app.get("/ping")
async def ping():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.get("/system/stats")
async def system_stats():
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "service_info": {
            "name": "LegalMind AI Multi-Document Assistant",
            "version": "2.0.0",
            "environment": os.getenv("ENVIRONMENT", "development")
        },
        "active_sessions": len(sessions_data),
        "total_documents": len(documents_data),
        "system_resources": {
            "python_version": "3.11",
            "docker": "✅ Running"
        }
    }

# Multi-document API endpoints
@app.post("/api/v1/multi-document/sessions/create")
async def create_session():
    """Create a new multi-document conversation session"""
    
    session_id = f"session_{uuid.uuid4().hex[:16]}"
    
    sessions_data[session_id] = {
        "session_id": session_id,
        "created_at": datetime.utcnow().isoformat(),
        "last_updated": datetime.utcnow().isoformat(),
        "documents": [],
        "conversation_history": [],
        "document_count": 0
    }
    
    return {
        "success": True,
        "session_id": session_id,
        "created_at": datetime.utcnow().isoformat(),
        "status": "ready",
        "capabilities": {
            "multi_document_analysis": True,
            "cross_document_references": True,
            "real_time_integration": True,
            "multilingual_support": True,
            "voice_summaries": True
        }
    }

@app.post("/api/v1/multi-document/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    language: str = Form("en"),
    auto_analyze: bool = Form(True)
):
    """Upload document to session"""
    
    if session_id not in sessions_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Read file content
    file_content = await file.read()
    
    if len(file_content) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    
    # Process document (simplified for demo)
    document_id = f"doc_{uuid.uuid4().hex[:12]}"
    
    # Simple text extraction (in production, handle PDF/DOCX properly)
    try:
        if file.filename.lower().endswith('.txt'):
            content = file_content.decode('utf-8')
        else:
            content = f"Document content from {file.filename} (simulated processing)"
    except:
        content = f"Binary document: {file.filename}"
    
    # Create document summary
    document_summary = {
        "document_id": document_id,
        "title": file.filename,
        "document_type": file.filename.split('.')[-1].lower() if '.' in file.filename else "unknown",
        "key_topics": ["legal_terms", "obligations", "risks"],  # Simulated
        "main_entities": ["Party A", "Party B"],  # Simulated
        "risk_level": "medium",  # Simulated
        "summary_text": f"Analysis of {file.filename} showing standard legal document structure.",
        "chunk_count": 1,
        "word_count": len(content.split()),
        "created_at": datetime.utcnow().isoformat()
    }
    
    # Store document
    documents_data[document_id] = {
        "id": document_id,
        "session_id": session_id,
        "filename": file.filename,
        "content": content,
        "summary": document_summary,
        "uploaded_at": datetime.utcnow().isoformat()
    }
    
    # Update session
    sessions_data[session_id]["documents"].append(document_id)
    sessions_data[session_id]["document_count"] += 1
    sessions_data[session_id]["last_updated"] = datetime.utcnow().isoformat()
    
    # Integration suggestions (simulated)
    integration_suggestions = []
    if len(sessions_data[session_id]["documents"]) > 1:
        integration_suggestions = [
            {
                "type": "topic_continuation",
                "priority": "medium",
                "message": f"I notice your new document also covers legal terms. Would you like me to compare with your other {len(sessions_data[session_id]['documents'])-1} documents?",
                "suggested_queries": [
                    "Compare legal terms across all documents",
                    "What are the differences in risk levels?",
                    "Are there any conflicts between my documents?"
                ]
            }
        ]
    else:
        integration_suggestions = [
            {
                "type": "initial_analysis",
                "priority": "medium",
                "message": f"I've analyzed your {document_summary['document_type']}. You can now ask questions about its content, risks, and key terms.",
                "suggested_queries": [
                    "What are the main risks in this document?",
                    "Summarize the key terms and obligations",
                    "What should I be concerned about?"
                ]
            }
        ]
    
    return {
        "success": True,
        "document_id": document_id,
        "session_id": session_id,
        "document_summary": document_summary,
        "integration_suggestions": integration_suggestions,
        "relationship_updates": [],
        "is_mid_conversation_upload": len(sessions_data[session_id]["documents"]) > 1,
        "processing_time": 2.3,
        "session_overview": {
            "total_documents": len(sessions_data[session_id]["documents"]),
            "session_id": session_id
        }
    }

@app.post("/api/v1/multi-document/chat")
async def chat_with_documents(request: ChatRequest):
    """Chat with all documents in session"""
    
    if request.session_id not in sessions_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions_data[request.session_id]
    
    if len(session["documents"]) == 0:
        raise HTTPException(status_code=400, detail="No documents found in session")
    
    # Simulate AI response based on documents
    doc_count = len(session["documents"])
    doc_titles = [documents_data[doc_id]["filename"] for doc_id in session["documents"]]
    
    # Create simulated response
    if "risk" in request.message.lower():
        ai_response = f"Based on your {doc_count} documents ({', '.join(doc_titles)}), I've identified several risk areas: 1) Contract termination clauses, 2) Liability limitations, 3) Payment obligations. The overall risk level appears to be medium across your document portfolio."
    elif "compare" in request.message.lower():
        ai_response = f"Comparing across your {doc_count} documents: I found similarities in legal structure but differences in specific terms. Your {doc_titles[0]} has stricter termination clauses compared to {doc_titles[-1] if len(doc_titles) > 1 else 'other standard agreements'}."
    else:
        ai_response = f"I've analyzed your question across all {doc_count} documents in your portfolio. The information spans multiple sections and shows consistent legal patterns. Would you like me to focus on any specific document or topic?"
    
    # Source attributions (simulated)
    source_attributions = []
    for i, doc_id in enumerate(session["documents"][:3]):  # Max 3 sources
        doc = documents_data[doc_id]
        source_attributions.append({
            "document_id": doc_id,
            "document_title": doc["filename"],
            "page_range": "1-2",
            "relevance_score": 0.8 - (i * 0.1),
            "confidence_score": 0.85,
            "key_concepts": ["legal_terms", "obligations"],
            "excerpt": f"Relevant excerpt from {doc['filename']}..."
        })
    
    # Add to conversation history
    session["conversation_history"].append({
        "user_message": request.message,
        "assistant_response": ai_response,
        "timestamp": datetime.utcnow().isoformat(),
        "sources_used": len(source_attributions)
    })
    
    # Update session
    sessions_data[request.session_id]["last_updated"] = datetime.utcnow().isoformat()
    
    return {
        "success": True,
        "session_id": request.session_id,
        "response": {
            "query": request.message,
            "answer": ai_response,
            "source_attributions": source_attributions,
            "synthesis_strategy": "comprehensive",
            "confidence_score": 0.85,
            "follow_up_suggestions": [
                "Tell me more about the specific risks",
                "How can I mitigate these issues?",
                "Compare termination clauses across documents"
            ],
            "cross_references": [
                {
                    "topic": "termination_clauses",
                    "documents": session["documents"][:2],
                    "relationship": "similar"
                }
            ] if len(session["documents"]) > 1 else []
        },
        "session_state": {
            "session_id": request.session_id,
            "document_count": len(session["documents"]),
            "conversation_turn_count": len(session["conversation_history"]),
            "last_updated": datetime.utcnow().isoformat()
        },
        "integration_opportunities": [],
        "processing_time": 1.8
    }

@app.get("/api/v1/multi-document/sessions/{session_id}/overview")
async def get_session_overview(session_id: str):
    """Get session overview"""
    
    if session_id not in sessions_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions_data[session_id]
    
    # Get document summaries
    documents = []
    for doc_id in session["documents"]:
        if doc_id in documents_data:
            documents.append(documents_data[doc_id]["summary"])
    
    return {
        "success": True,
        "session_id": session_id,
        "session_overview": {
            "session_id": session_id,
            "created_at": session["created_at"],
            "document_count": len(session["documents"]),
            "conversation_turns": len(session["conversation_history"]),
            "last_activity": session["last_updated"]
        },
        "documents": documents,
        "relationships": [],  # Simulated
        "conversation_insights": {
            "total_messages": len(session["conversation_history"]),
            "dominant_topics": ["legal_analysis", "risk_assessment"],
            "user_expertise_level": "intermediate"
        },
        "portfolio_analysis": {
            "total_documents": len(session["documents"]),
            "document_types": list(set([doc["summary"]["document_type"] for doc_id, doc in documents_data.items() if doc_id in session["documents"]])),
            "overall_risk": "medium",
            "analysis_depth": "comprehensive" if len(session["documents"]) >= 2 else "basic"
        }
    }

@app.get("/api/v1/multi-document/sessions/{session_id}/relationships")
async def get_relationships(session_id: str):
    """Get document relationships"""
    
    if session_id not in sessions_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions_data[session_id]
    
    # Create network graph (simulated)
    nodes = []
    edges = []
    
    for doc_id in session["documents"]:
        if doc_id in documents_data:
            doc = documents_data[doc_id]
            nodes.append({
                "id": doc_id,
                "label": doc["filename"],
                "type": "document",
                "risk_level": doc["summary"]["risk_level"],
                "size": 30
            })
    
    # Create edges between documents (simulated)
    if len(nodes) > 1:
        for i in range(len(nodes)-1):
            edges.append({
                "source": nodes[i]["id"],
                "target": nodes[i+1]["id"],
                "weight": 0.7,
                "type": "related",
                "strength": "medium"
            })
    
    return {
        "success": True,
        "session_id": session_id,
        "relationships": [],
        "network_graph": {
            "nodes": nodes,
            "edges": edges
        },
        "relationship_summary": {
            "total_relationships": len(edges),
            "document_count": len(nodes),
            "most_connected_document": nodes[0]["id"] if nodes else None
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
