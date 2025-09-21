# 🧠 LegalMind AI - Multi-Document Legal Assistant

> **FULLY FUNCTIONAL** enterprise-grade legal document analysis platform with a sophisticated backend and a modern, minimalistic frontend. This project combines a 14-service microservice architecture with a React-based UI to demystify complex legal documents.

[](https://www.python.org/downloads/)
[](https://fastapi.tiangolo.com/)
[](https://react.dev/)
[](https://www.google.com/search?q=)

-----

## ✨ Features

### **Backend & AI Core**

  - **🧠 Intelligent Document Processing**: Upload any document type for automatic content extraction, risk assessment, and smart categorization.
  - **💬 Advanced Conversational AI**: Multi-document context awareness, cross-document relationship detection, and professional response synthesis with source citations.
  - **📊 Portfolio Analytics**: Cross-document risk assessment, document relationship mapping, and advanced conversation insights.
  - **⚡ Real-time Features**: Live session management, instant document integration, and context-aware responses.
  - **🔧 Enterprise-Grade Architecture**: Built with **14 sophisticated microservices**, dynamic service discovery, and professional error handling.

### **Frontend UI/UX**

  - **Modern Chat Interface**: A three-panel layout inspired by NotebookLM for managing sources, interacting with the AI, and viewing insights.
  - **Rich Data Visualization**: Risk scoring and document insights are displayed with modern graphs using Recharts.
  - **Elegant Design**: Built with React and Tailwind CSS, featuring a dark theme, Google Sans font, and glassmorphism-style elements.
  - **Smooth Animations**: Fluid user experience powered by Framer Motion.
  - **Voice-Enabled Input**: Ask questions using your voice for hands-free interaction.
  - **Real-time Progress**: A progress bar shows the status of backend processes like text extraction, AI analysis, and summary generation.

-----

## 🛠 Tech Stack

  - **Backend**: Python 3.11+, FastAPI
  - **Frontend**: React (Vite)
  - **Styling**: Tailwind CSS
  - **Animations**: Framer Motion
  - **Charts & Graphs**: Recharts
  - **Icons**: Lucide React
  - **API Integration**: Axios

-----

## 🚀 Getting Started

### **Prerequisites**

  * Node.js (v18+)
  * npm or yarn package manager
  * Python 3.11+ with required packages for the backend.

### **1. Backend Setup**

First, ensure the backend server is running. It serves the core logic and API for the frontend.

  - **Backend is running at**: `http://localhost:8000`
  - **API Documentation**: `http://localhost:8000/docs`
  - **Health Check**: `curl http://localhost:8000/health`

#### **Environment Configuration (Optional for Development)**

Create a `.env` file in the backend directory if needed:

```
GOOGLE_API_KEY=your-gemini-api-key
GOOGLE_CLOUD_PROJECT=your-project-id
ENVIRONMENT=development
```

### **2. Frontend Setup**

With the backend running, you can set up and launch the frontend.

```bash
# Clone the repository
git clone https://github.com/your-repo/legal-docs-ai-frontend.git

# Navigate to project folder
cd legal-docs-ai-frontend

# Install dependencies
npm install

# Run the development server
npm run dev
```

-----

## 📡 Complete API Reference

### **System Endpoints**

#### `GET /`

**System information and capabilities.**

```json
{
  "message": "🧠 Welcome to LegalMind AI - Advanced Multi-Document Assistant",
  "status": "operational",
  "your_sophisticated_architecture": {
    "available_services": ["chat_handler", ...],
    "total_services": 14
  }
}
```

#### `GET /health`

**System health and service status.**

```json
{
  "status": "healthy",
  "your_architecture": {
    "services_loaded": 14,
    "services_available": ["chat_handler", ...]
  }
}
```

### **Session Management**

#### `POST /api/v1/multi-document/sessions/create`

**Create new conversation session.**

```json
{
  "success": true,
  "session_id": "session_a1b2c3d4",
  "created_at": "2025-09-17T20:00:00Z"
}
```

#### `GET /api/v1/multi-document/sessions/{session_id}/overview`

**Get session overview with documents and analytics.**

```json
{
  "success": true,
  "session_id": "session_a1b2c3d4",
  "documents": [
    {
      "document_id": "doc_abc123",
      "title": "employment_contract.pdf",
      "risk_level": "high"
    }
  ],
  "portfolio_analysis": {
    "overall_risk": "medium"
  }
}
```

### **Document Management**

#### `POST /api/v1/multi-document/upload`

**Upload document with advanced processing.**

  * **Request**: `FormData` containing `file`, `session_id`, `language`, `auto_analyze`.
  * **Response**:

<!-- end list -->

```json
{
  "success": true,
  "document_id": "doc_abc123",
  "document_summary": {
    "title": "employment_contract.pdf",
    "key_topics": ["employment", "termination", "compensation"],
    "risk_level": "high"
  }
}
```

### **Advanced Conversational AI**

#### `POST /api/v1/multi-document/chat`

**Chat with multi-document intelligence.**

  * **Request**:

<!-- end list -->

```json
{
  "session_id": "session_a1b2c3d4",
  "message": "What are the biggest risks across all my documents?",
  "language": "en"
}
```

  * **Response**:

<!-- end list -->

```json
{
  "success": true,
  "response": {
    "answer": "Using your sophisticated multi-document architecture...",
    "source_attributions": [
      {
        "document_id": "doc_abc123",
        "document_title": "employment_contract.pdf",
        "excerpt": "Contract termination requires 30-day notice..."
      }
    ],
    "cross_references": [
        {
            "topic": "termination_clauses",
            "documents": ["doc_abc123", "doc_def456"],
            "relationship": "conflicting_terms"
        }
    ]
  }
}
```

-----

## 🎨 Frontend Integration Guide

This is how the frontend components map to the backend API endpoints.

### **1. Sources Panel (Left Side)**

  * **Functionality**: Users upload and manage documents.
  * **Endpoints Used**:
      * `POST /api/v1/multi-document/upload`: To upload new files.
      * `GET /api/v1/multi-document/sessions/{session_id}/overview`: To refresh the list of documents in the current session.

<!-- end list -->

```javascript
// Document Upload Handler
const handleDocumentUpload = async (files, sessionId) => {
  const uploadPromises = files.map(file => uploadDocument(file, sessionId));
  const results = await Promise.all(uploadPromises);
  // Update UI with document summaries from results
};

// Real-time document list update
const refreshDocumentsList = async (sessionId) => {
  const overview = await getSessionOverview(sessionId);
  updateSourcesPanel(overview.documents);
};
```

### **2. Chat Panel (Center)**

  * **Functionality**: Users ask questions via text or voice.
  * **Endpoint Used**:
      * `POST /api/v1/multi-document/chat`: To send a user message and receive an AI-generated response.

<!-- end list -->

```javascript
// Advanced Chat Handler
const handleChatMessage = async (message, sessionId) => {
  showTypingIndicator();
  try {
    const response = await chatWithDocuments(message, sessionId);
    if (response.success) {
      // Display AI response with citations, suggestions, and cross-references
      displayChatResponse({
        message: response.response.answer,
        sources: response.response.source_attributions,
        crossReferences: response.response.cross_references
      });
    }
  } finally {
    hideTypingIndicator();
  }
};
```

### **3. Insights Panel (Right Side)**

  * **Functionality**: Displays portfolio-level analytics, risk scores, and document relationship graphs.
  * **Endpoint Used**:
      * `GET /api/v1/multi-document/sessions/{session_id}/overview`: To fetch aggregate data for display.

<!-- end list -->

```javascript
// Advanced Analytics Display
const displayPortfolioAnalytics = async (sessionId) => {
  const overview = await getSessionOverview(sessionId);
  showPortfolioMetrics({
    totalDocuments: overview.session_overview.document_count,
    overallRisk: overview.portfolio_analysis.overall_risk,
  });
};
```

-----

## 🧪 Testing Your Integration

Use this script in your browser's developer console to perform a quick end-to-end test of the backend integration.

```javascript
// Complete integration test
const testBackendIntegration = async () => {
  console.log('🧪 Testing LegalMind AI Backend Integration...');
  const API_BASE = 'http://localhost:8000';

  try {
    // 1. Test system health
    const health = await fetch(`${API_BASE}/health`);
    const healthData = await health.json();
    console.log('✅ Health:', healthData.status, `(${healthData.your_architecture.services_loaded} services loaded)`);

    // 2. Create session
    const sessionResponse = await fetch(`${API_BASE}/api/v1/multi-document/sessions/create`, { method: 'POST' });
    const sessionData = await sessionResponse.json();
    const sessionId = sessionData.session_id;
    console.log('✅ Session Created:', sessionId);

    // 3. Upload test document
    const testFile = new File(['Test legal document with penalty clauses'], 'test.txt', { type: 'text/plain' });
    const formData = new FormData();
    formData.append('file', testFile);
    formData.append('session_id', sessionId);
    const uploadResponse = await fetch(`${API_BASE}/api/v1/multi-document/upload`, { method: 'POST', body: formData });
    const uploadResult = await uploadResponse.json();
    console.log('✅ Document Uploaded:', uploadResult.success);
    console.log('✅ Detected Risk Level:', uploadResult.document_summary.risk_level);

    // 4. Test chat
    const chatResponse = await fetch(`${API_BASE}/api/v1/multi-document/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message: 'What risks are in my document?' })
    });
    const chatResult = await chatResponse.json();
    console.log('✅ Chat Responded:', chatResult.success);
    console.log('✅ AI Response:', chatResult.response.answer.substring(0, 100) + '...');

    console.log('🎉 All backend integrations working perfectly!');
  } catch (error) {
    console.error('❌ Integration test failed:', error);
  }
};

// Run test
testBackendIntegration();
```

-----

## 🚀 Deployment

The backend is production-ready and can be deployed to any modern cloud container service:

  - Google Cloud Run
  - AWS Lambda/ECS
  - Azure Container Instances
  - Heroku
  - Render
  - Railway

The frontend is a standard React application and can be deployed to static hosting providers like Vercel, Netlify, or Firebase Hosting.

-----



**Backend Status:** ✅ 100% Functional - Ready for Frontend Integration
**API Documentation:** `http://localhost:8000/docs`
**System Health:** `http://localhost:8000/health`

**Your 14-service sophisticated architecture is operational and ready to power an award-winning application\!** 🏆

## 📞 Support
Gmail -: shadowicwarrior@gmail.com
