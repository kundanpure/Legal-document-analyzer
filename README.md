# LegalMind AI

## Overview
LegalMind AI is a comprehensive, enterprise-grade multi-document legal assistant designed to demystify complex legal documents. Built on a robust, service-oriented architecture, the platform combines a high-performance backend with a modern, responsive frontend to deliver unparalleled accuracy and insights in legal document analysis.

## Core Capabilities
- **Intelligent Document Processing:** Automatically extracts content, identifies clauses, and assesses risk across various document types.
- **Advanced Conversational AI:** Features context-aware, multi-document chat functionality with accurate source citations and professional response synthesis.
- **Dynamic Insights Generation:** Generates concise document summaries, text-to-speech audio overviews, and structured PDF reports.
- **Cross-Document Analytics:** Maps relationships between multiple documents and provides aggregated risk assessments.
- **Enterprise-Grade Infrastructure:** Powered by 14 specialized microservices handling tasks ranging from chunking and real-time processing to secure storage management.

## System Architecture

The platform is logically partitioned into two primary components:

### 1. Frontend (React / Vite)
The user interface is built as a single-page application focused on providing a seamless, desktop-class experience in the browser. 
- **Technologies:** React 18, TypeScript, Vite, Tailwind CSS.
- **Features:** Split-screen authentication, interactive dashboard, multi-document chat interface with contextual sidebars, and real-time rendering of document insights.
- **State Management:** TanStack React Query handles server state caching and synchronization, ensuring a responsive interface without redundant network requests.
- **Documentation:** For detailed setup and development instructions, refer to `Frontend/README.md`.

### 2. Backend (FastAPI / Python)
The intelligence engine is built on a high-performance Python framework, heavily utilizing asynchronous programming to manage concurrent processing workloads.
- **Technologies:** Python 3.10+, FastAPI, Google Cloud Platform (Document AI, Storage, Vertex AI/Gemini, Text-to-Speech).
- **Features:** Asynchronous document parsing, semantic chunking, secure signed-URL generation for direct-to-cloud uploads, and a highly modular service registry.
- **Data Persistence:** Currently leverages a synchronized local JSON data store (`legalmind_db.json`), designed to easily transition to a robust NoSQL/SQL database in production.
- **Documentation:** For detailed setup and development instructions, refer to `Backend/README.md`.

## Setup and Installation

### Prerequisites
- Node.js v18 or higher
- Python 3.10 or higher
- A Google Cloud Platform account with a provisioned Service Account containing required permissions.

### Backend Initialization
1. Navigate to the `Backend` directory.
2. Copy `.env.example` to `.env` and populate your Google Cloud credentials and API keys.
3. Install dependencies using `pip install -r requirements.txt`.
4. Run the server using `python main.py` or `python3 main.py`.

### Frontend Initialization
1. Navigate to the `Frontend` directory.
2. Copy `.env.example` to `.env` and populate your Firebase credentials and API URL.
3. Install dependencies using `npm install`.
4. Start the development server using `npm run dev`.

## Security Framework
- **Data Isolation:** User data and chat histories are isolated using X-User-ID headers and strict backend validation.
- **Direct Cloud Uploads:** The system utilizes Google Cloud Storage V4 Signed URLs to facilitate secure, temporary upload channels directly from the client to the cloud, bypassing the backend to minimize bandwidth overhead.
- **Uniform Bucket-Level Access:** The backend securely proxies authenticated requests to GCP to retrieve documents and generated audio files, maintaining strict compliance with bucket-level security policies.

## Development and Contribution
Ensure you maintain the strict separation of concerns between the frontend presentation logic and the backend processing pipelines. Always use the generated `.env.example` files as a baseline for configuring new environments. Code modifications must adhere strictly to the established architectural patterns and typing standards.
