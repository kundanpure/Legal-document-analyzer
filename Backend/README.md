# Backend Architecture: LegalMind AI

## Overview
The LegalMind AI backend serves as the critical intelligence processing engine. It manages heavy computational workloads including text extraction from legal documents via Google Cloud Document AI, conversational intelligence via Google Gemini, text-to-speech audio generation via Google Cloud TTS, and comprehensive cloud storage management via Google Cloud Storage (GCS).

## Technical Stack
- **Framework:** FastAPI (Python 3.10+)
- **Concurrency:** Asynchronous programming (`asyncio`, `aiohttp`, `aiofiles`)
- **Cloud Infrastructure:** Google Cloud Platform (GCP)
- **AI Models:** Google Gemini 2.5 Flash / Pro via Vertex AI / Generative AI SDK
- **Data Persistence:** In-memory local JSON-based state persistence (`legalmind_db.json`)
- **Storage:** Google Cloud Storage (with dynamic secure Signed URL and direct file proxy processing)
- **Authentication:** Middleware-based header verification (`X-User-ID`) isolating user-specific datasets.

## System Architecture

The backend operates on a service-oriented architecture, ensuring maximum modularity and scalability. All primary intelligence services are lazily loaded to ensure quick server startups and isolation of dependencies.

### Core Services
1. **Document Processor:** Interacts with Google Cloud Document AI to perform Optical Character Recognition (OCR) and layout parsing on complex legal documents.
2. **Document Chunker:** Intelligently splits massive legal texts into optimized contexts, retaining paragraph boundaries and semantic meaning.
3. **Gemini Analyzer:** Core wrapper around the Gemini Generative AI models. Uses custom system instructions tuned for legal precision to prevent hallucination and provide accurate legal analysis.
4. **Voice Generator:** Uses Google Cloud Text-to-Speech to generate audio summaries (MP3 format) of lengthy legal documents for quick auditory consumption.
5. **Report Generator:** Utilizes `reportlab` to compile structured visual PDF reports containing risk analysis metrics and summaries.
6. **Storage Manager:** Handles securely interfacing with Google Cloud Storage. Generates V4 signed URLs for client-side direct uploads and bypasses Uniform Bucket-Level Access limitations by dynamically fetching content using internal service account credentials.

## Local Setup

### 1. Prerequisites
Ensure you have Python 3.10 or higher installed. You must also have a valid Google Cloud Service Account JSON key file with permissions for Cloud Storage, Document AI, and Vertex AI.

### 2. Environment Variables
Copy `.env.example` to `.env` and fill in your details:
- `GOOGLE_CLOUD_PROJECT`: Your GCP Project ID.
- `GOOGLE_API_KEY`: Your Gemini API Key.
- `GCS_BUCKET_NAME`: Name of the bucket designated for application storage.
- `DOCUMENT_AI_LOCATION` & `DOCUMENT_AI_PROCESSOR_ID`: Location and specific ID of your provisioned Document AI processor.
- `GOOGLE_APPLICATION_CREDENTIALS`: Absolute or relative path to your downloaded GCP JSON key file.

### 3. Installation
Install the necessary requirements. We strongly recommend using a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 4. Running the Server
Execute the main script to start the FastAPI server via Uvicorn:
```bash
python main.py
```
The server will initialize all services and be available at `http://localhost:8000`. The interactive API documentation is automatically accessible at `http://localhost:8000/docs`.

## Data Management
The system currently uses an optimized local JSON file database (`legalmind_db.json`) mapped dynamically to internal memory. This allows rapid prototyping without dedicated external database infrastructure. The system leverages background tasks (`asyncio.create_task`) to synchronize in-memory state with disk storage persistently.

## Security Considerations
- **CORS:** Cross-Origin Resource Sharing is currently configured permissively to allow frontend integration. Tighten these rules in production.
- **Signed URLs:** Client direct uploads heavily utilize short-lived V4 signed URLs to prevent public write access to the storage bucket.
- **IAM:** Your GCP service account should employ the principle of least privilege.
