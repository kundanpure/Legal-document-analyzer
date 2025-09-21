"""
LegalMind AI - Frontend Integration Ready API
ALL ENDPOINTS MATCHING FRONTEND REQUIREMENTS
"""

import uvicorn
import asyncio
import json
import base64
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, timezone, timedelta
import os
import sys
import tempfile
import aiohttp
import importlib
import traceback
import uuid
from io import BytesIO
import mimetypes

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configuration
class Settings:
    def __init__(self):
        self.GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "docanalyzer-470219")
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
        self.GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "docanalyzer-470219-storage")
        self.SECRET_KEY = os.getenv("SECRET_KEY", "default-secret-key")
        self.ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

from dotenv import load_dotenv
load_dotenv()  # ensure .env is read

# Logger setup
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

settings = Settings()

# Service imports (dynamic)
available_services = {}
service_imports = {
    'chat_handler': 'app.services.chat_handler',
    'document_processor': 'app.services.document_processor',
    'gemini_analyzer': 'app.services.gemini_analyzer',
    'multi_document_chat': 'app.services.multi_document_chat',
    'storage_manager': 'app.services.storage_manager',
    'translation_service': 'app.services.translation_service',
    'report_generator': 'app.services.report_generator',
    'voice_generator': 'app.services.voice_generator'
}

def _download_gcs_as_bytes(gcs_uri: str) -> bytes:
    """
    Download a GCS object to memory using explicit service-account creds.
    gcs_uri examples:
      - gs://bucket/path/to/file.pdf
      - path/to/file.pdf   (treated under settings.GCS_BUCKET_NAME)
    """
    from google.cloud import storage
    from google.oauth2 import service_account

    bucket_name = settings.GCS_BUCKET_NAME
    if gcs_uri.startswith("gs://"):
        without = gcs_uri[5:]
        bucket_name, _, object_key = without.partition("/")
        if not bucket_name or not object_key:
            raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    else:
        object_key = gcs_uri.lstrip("/")

    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path or not os.path.exists(creds_path):
        raise RuntimeError("Missing GOOGLE_APPLICATION_CREDENTIALS")
    creds = service_account.Credentials.from_service_account_file(creds_path)

    client = storage.Client(project=settings.GOOGLE_CLOUD_PROJECT, credentials=creds)
    blob = client.bucket(bucket_name).blob(object_key)
    if not blob.exists(client):
        raise FileNotFoundError(f"GCS object not found: gs://{bucket_name}/{object_key}")

    return blob.download_as_bytes()


def try_import_service(service_name: str, module_path: str):
    """Try to import a service dynamically"""
    try:
        module = importlib.import_module(module_path)
        possible_classes = [
            service_name.replace('_', '').title(),
            ''.join(word.capitalize() for word in service_name.split('_')),
            service_name.title().replace('_', ''),
            f"{service_name.title().replace('_', '')}Service"
        ]
        for class_name in possible_classes:
            if hasattr(module, class_name):
                cls = getattr(module, class_name)
                logger.info(f"✅ Successfully imported {service_name}: {class_name}")
                return cls
        logger.info(f"📦 Imported module {service_name}")
        return module
    except ImportError as e:
        logger.warning(f"⚠️ Could not import {service_name}: {e}")
        return None
    except Exception as e:
        logger.warning(f"⚠️ Error importing {service_name}: {e}")
        return None

logger.info("🔍 Discovering services...")
for service_name, module_path in service_imports.items():
    service = try_import_service(service_name, module_path)
    if service:
        available_services[service_name] = service
logger.info(f"✅ Successfully imported {len(available_services)} services!")

# Models
class AuthTokenRequest(BaseModel):
    token: str

class SignedUrlRequest(BaseModel):
    filename: str
    content_type: str
    file_size: Optional[int] = None

class UploadNotificationRequest(BaseModel):
    file_id: str
    gcs_path: str
    original_filename: str
    file_size: int
    content_type: str

class ProcessingStartRequest(BaseModel):
    file_id: str
    processing_type: str = "full_analysis"
    options: Optional[Dict[str, Any]] = None

class InsightRequest(BaseModel):
    file_id: str
    type: str  # summary, audio, report
    options: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    file_id: Optional[str] = None
    conversation_id: Optional[str] = None
    stream: bool = False

class ExportConversationRequest(BaseModel):
    conversation_id: str
    format: str = "pdf"  # pdf, json, txt

# Global stores (in-memory)
files_data: Dict[str, Dict[str, Any]] = {}
jobs_data: Dict[str, Dict[str, Any]] = {}
insights_data: Dict[str, Dict[str, Any]] = {}
conversations_data: Dict[str, Dict[str, Any]] = {}
signed_urls_data: Dict[str, Any] = {}

def get_utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

def generate_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting LegalMind AI API Server...")
    for directory in ["uploads", "reports", "audio", "temp", "exports"]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ LegalMind AI API Server started with {len(available_services)} services!")
    yield
    logger.info("🛑 Shutting down LegalMind AI API Server...")

# FastAPI app
app = FastAPI(
    title="🧠 LegalMind AI - Frontend Integration API",
    description="""
    **Complete API for LegalMind AI Frontend Integration**
    
    ## 📋 Essential Endpoints
    - **File Upload Management** - Signed URLs, upload notifications, file listing
    - **Processing Pipeline** - Job status tracking, processing triggers
    - **AI Insights** - Summary, audio, report generation
    - **Chat Integration** - Sync/async chat with document context
    - **Export & Download** - Conversation exports, file downloads
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8080",
        "https://nimble-kringle-55eacc.netlify.app",
        "https://legal-ai-xi.vercel.app",   # <-- add this
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


try:
    app.mount("/static", StaticFiles(directory="uploads"), name="static")
except Exception:
    pass

# ================================
# AUTH (optional)
# ================================
@app.post("/api/auth/verify-token")
async def verify_token(request: AuthTokenRequest):
    if request.token and len(request.token) > 10:
        return {
            "valid": True,
            "user_id": "user_123",
            "permissions": ["read", "write", "admin"],
            "expires_at": get_utc_timestamp()
        }
    return {"valid": False, "error": "Invalid token"}

# ================================
# UPLOADS
# ================================
from google.cloud import storage
from google.oauth2 import service_account

@app.post("/api/uploads/get-signed-url")
async def get_signed_url(request: SignedUrlRequest):
    """
    Return a real V4 signed URL for uploading directly to GCS via PUT.
    """
    try:
        file_id = generate_id("file")
        filename = request.filename
        object_key = f"uploads/{file_id}/{filename}"
        content_type = request.content_type or "application/pdf"

        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not creds_path or not os.path.exists(creds_path):
            raise HTTPException(
                status_code=500,
                detail=f"GOOGLE_APPLICATION_CREDENTIALS not found or invalid: {creds_path}"
            )
        creds = service_account.Credentials.from_service_account_file(creds_path)

        storage_client = storage.Client(
            project=settings.GOOGLE_CLOUD_PROJECT,
            credentials=creds
        )
        bucket = storage_client.bucket(settings.GCS_BUCKET_NAME)
        blob = bucket.blob(object_key)

        signed_url = blob.generate_signed_url(
            method="PUT",
            version="v4",
            expiration=timedelta(minutes=10),
            content_type=content_type,
        )

        return {
            "file_id": file_id,
            "signed_url": signed_url,
            "expires_at": get_utc_timestamp(),
            "upload_fields": {"key": object_key, "Content-Type": content_type},
            "gcs_path": f"gs://{settings.GCS_BUCKET_NAME}/{object_key}",
        }

    except Exception as e:
        logger.error(f"Signed URL generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate signed URL: {e}")

@app.post("/api/uploads/notify-uploaded")
async def notify_uploaded(req: UploadNotificationRequest):
    files_data[req.file_id] = {
        "file_id": req.file_id,
        "original_filename": req.original_filename,
        "file_size": req.file_size,
        "content_type": req.content_type,
        "gcs_path": req.gcs_path,
        "uploaded_at": get_utc_timestamp(),
        "processing_status": "queued",
        "insights": {},
        "metadata": {},
    }

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    jobs_data[job_id] = {
        "id": job_id,
        "file_id": req.file_id,
        "status": "queued",
        "error": None,
        "created_at": get_utc_timestamp(),
        "updated_at": get_utc_timestamp(),
        "type": "real_document_analysis",
    }
    files_data[req.file_id]["job_id"] = job_id

    asyncio.create_task(process_file_job(req.file_id, job_id))

    return {
        "success": True,
        "file_id": req.file_id,
        "processing_started": True,
        "job_id": job_id,
    }

# ================================
# PROCESSING PIPELINE
# ================================
async def process_file_job(file_id: str, job_id: str):
    logger.info(f"[JOB] start file_id={file_id} job_id={job_id}")
    try:
        jobs_data[job_id]["status"] = "processing"
        jobs_data[job_id]["updated_at"] = get_utc_timestamp()
        files_data[file_id]["processing_status"] = "processing"

        # 1) Download from GCS
        local_path = await download_file_from_gcs_real(files_data[file_id]["gcs_path"])
        logger.info(f"[PROC] downloaded to {local_path}")

        # 2) Extract with Document AI
        try:
            from app.services.document_processor import RealDocumentProcessor
        except ImportError:
            raise Exception("RealDocumentProcessor could not be imported from app.services.document_processor")

        processor = RealDocumentProcessor()
        extraction = await processor.extract_text_from_pdf(local_path)
        logger.info(f"[PROC] docai ok. chars={len(extraction.get('full_text',''))}")

        # 3) Chunk
        try:
            from app.services.document_chunker import EnhancedChunker, StandardDocument
        except ImportError:
            raise Exception("EnhancedChunker could not be imported from app.services.document_chunker")

        chunker = EnhancedChunker(max_pages_per_chunk=15, max_chars_per_chunk=8000)
        chunks = await chunker.create_chunks(
            StandardDocument(
                id=file_id,
                content=extraction["full_text"],
                page_count=extraction["statistics"]["total_pages"],
                word_count=len(extraction["full_text"].split()),
            )
        )
        logger.info(f"[PROC] chunks={len(chunks)}")

        # 4) Save results
        files_data[file_id]["extraction"] = extraction
        files_data[file_id]["chunks"] = [c.__dict__ for c in chunks]
        # minimal analysis so downstream endpoints don't fail
        files_data[file_id]["analysis"] = {
            "full_text": extraction.get("full_text", ""),
            "summary": "",
            "document_type": "document",
            "overall_risk_score": 0,
            "key_risks": [],
        }
        files_data[file_id]["processing_status"] = "completed"
        files_data[file_id]["updated_at"] = get_utc_timestamp()

        jobs_data[job_id]["status"] = "completed"
        jobs_data[job_id]["updated_at"] = get_utc_timestamp()
        logger.info(f"[JOB] done file_id={file_id} job_id={job_id}")

    except Exception as e:
        logger.exception(f"[JOB] failed file_id={file_id} job_id={job_id}: {e}")
        files_data[file_id]["processing_status"] = "failed"
        files_data[file_id]["error"] = str(e)
        files_data[file_id]["updated_at"] = get_utc_timestamp()
        jobs_data[job_id]["status"] = "failed"
        jobs_data[job_id]["error"] = str(e)
        jobs_data[job_id]["updated_at"] = get_utc_timestamp()

async def download_file_from_gcs_real(gcs_path: str) -> str:
    """Download a file from GCS using the SAME service account as Document AI."""
    try:
        from google.cloud import storage
        from google.oauth2 import service_account
        import tempfile

        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not creds_path or not os.path.exists(creds_path):
            raise Exception(f"GOOGLE_APPLICATION_CREDENTIALS not found or invalid: {creds_path}")
        creds = service_account.Credentials.from_service_account_file(creds_path)

        client = storage.Client(
            project=settings.GOOGLE_CLOUD_PROJECT,
            credentials=creds
        )

        bucket_name = settings.GCS_BUCKET_NAME
        if gcs_path.startswith("gs://"):
            without_scheme = gcs_path[5:]
            bucket_in_path, _, object_path = without_scheme.partition("/")
            if not bucket_in_path:
                raise Exception(f"Invalid GCS path (missing bucket): {gcs_path}")
            bucket_name = bucket_in_path
            blob_name = object_path
        else:
            blob_name = gcs_path.lstrip("/")

        if not blob_name:
            raise Exception(f"Could not determine object name from '{gcs_path}'")

        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if not blob.exists(client):
            raise Exception(f"GCS object not found: gs://{bucket_name}/{blob_name}")

        suffix = Path(blob_name).suffix or ".pdf"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.close()

        blob.download_to_filename(tmp.name)

        logger.info(f"Downloaded gs://{bucket_name}/{blob_name} -> {tmp.name}")
        return tmp.name

    except Exception as e:
        logger.error(f"GCS download failed: {e}")
        raise Exception(f"Failed to download from GCS: {e}")

# ================================
# UTILS
# ================================
def clean_text_for_gemini(text: str) -> str:
    if not text:
        return ""
    cleaned = ' '.join(text.split())
    cleaned = cleaned.replace('\x00', '').replace('\ufeff', '')
    if len(cleaned) > 30000:
        cleaned = cleaned[:30000] + "...[truncated]"
    import re
    cleaned = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '[EMAIL]', cleaned)
    cleaned = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', cleaned)
    cleaned = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', cleaned)
    return cleaned.strip()

# ================================
# DEBUG/LIST/FILE META
# ================================
@app.get("/api/debug/file/{file_id}")
async def debug_file(file_id: str):
    f = files_data.get(file_id)
    if not f:
        raise HTTPException(404, "File not found")
    j = jobs_data.get(f.get("job_id")) if f.get("job_id") else None
    return {"file": f, "job": j}

@app.get("/api/debug/job/{job_id}")
async def debug_job(job_id: str):
    j = jobs_data.get(job_id)
    if not j:
        raise HTTPException(404, "Job not found")
    return j

@app.get("/api/uploads")
async def list_user_files(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None, description="Filter by status"),
    file_type: Optional[str] = Query(None, description="Filter by file type")
):
    try:
        all_files = list(files_data.values())
        if status:
            all_files = [f for f in all_files if f.get("processing_status") == status]
        if file_type:
            all_files = [f for f in all_files if f.get("metadata", {}).get("file_type") == file_type]
        all_files.sort(key=lambda x: x.get("uploaded_at", ""), reverse=True)
        paginated_files = all_files[offset:offset + limit]

        files_list = []
        for fd in paginated_files:
            files_list.append({
                "file_id": fd["file_id"],
                "filename": fd["original_filename"],
                "file_size": fd["file_size"],
                "content_type": fd["content_type"],
                "uploaded_at": fd["uploaded_at"],
                "processing_status": fd.get("processing_status", "pending"),
                "file_type": fd.get("metadata", {}).get("file_type", "unknown"),
                "insights_available": {
                    "summary": fd.get("insights", {}).get("summary") is not None,
                    "audio": fd.get("insights", {}).get("audio") is not None,
                    "report": fd.get("insights", {}).get("report") is not None
                }
            })

        return {
            "files": files_list,
            "pagination": {
                "total": len(all_files),
                "limit": limit,
                "offset": offset,
                "has_more": len(all_files) > offset + limit
            },
            "filters_applied": {
                "status": status,
                "file_type": file_type
            }
        }
    except Exception as e:
        logger.error(f"List files error: {e}")
        raise HTTPException(status_code=500, detail="Failed to list files")

@app.get("/api/uploads/{file_id}")
async def get_file_metadata(file_id: str):
    try:
        if file_id not in files_data:
            raise HTTPException(status_code=404, detail="File not found")
        fd = files_data[file_id]
        return {
            "file_id": file_id,
            "filename": fd["original_filename"],
            "file_size": fd["file_size"],
            "content_type": fd["content_type"],
            "uploaded_at": fd["uploaded_at"],
            "processing_status": fd.get("processing_status", "pending"),
            "processed_at": fd.get("processed_at"),
            "gcs_path": fd["gcs_path"],
            "metadata": fd.get("metadata", {}),
            "analysis": fd.get("analysis", {}),
            "insights": {
                "summary_available": fd.get("insights", {}).get("summary") is not None,
                "audio_available": fd.get("insights", {}).get("audio") is not None,
                "report_available": fd.get("insights", {}).get("report") is not None
            },
            "job_id": fd.get("job_id"),
            "error": fd.get("error")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get file metadata error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get file metadata")

@app.get("/api/uploads/{file_id}/download-url")
async def get_file_download_url(file_id: str):
    try:
        if file_id not in files_data:
            raise HTTPException(status_code=404, detail="File not found")

        gcs_path = files_data[file_id]["gcs_path"]  # e.g. gs://bucket/key
        if not gcs_path.startswith("gs://"):
            bucket_name = settings.GCS_BUCKET_NAME
            object_key = gcs_path.lstrip("/")
        else:
            without = gcs_path[5:]
            bucket_name, _, object_key = without.partition("/")
            if not bucket_name or not object_key:
                raise HTTPException(status_code=500, detail=f"Invalid gcs_path: {gcs_path}")

        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not creds_path or not os.path.exists(creds_path):
            raise HTTPException(status_code=500, detail="Missing GOOGLE_APPLICATION_CREDENTIALS")
        creds = service_account.Credentials.from_service_account_file(creds_path)

        client = storage.Client(project=settings.GOOGLE_CLOUD_PROJECT, credentials=creds)
        blob = client.bucket(bucket_name).blob(object_key)

        download_url = blob.generate_signed_url(
            method="GET",
            version="v4",
            expiration=timedelta(minutes=15),
        )

        fd = files_data[file_id]
        return {
            "file_id": file_id,
            "download_url": download_url,
            "expires_at": get_utc_timestamp(),
            "filename": fd["original_filename"],
            "file_size": fd["file_size"],
            "content_type": fd["content_type"],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download URL error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate download URL")

# ================================
# MANUAL START (optional)
# ================================
@app.post("/api/process/start")
async def start_processing(request: ProcessingStartRequest, background_tasks: BackgroundTasks):
    try:
        if request.file_id not in files_data:
            raise HTTPException(status_code=404, detail="File not found")

        job_id = generate_id("job")
        jobs_data[job_id] = {
            "job_id": job_id,
            "file_id": request.file_id,
            "type": request.processing_type,
            "status": "queued",
            "options": request.options or {},
            "created_at": get_utc_timestamp()
        }

        background_tasks.add_task(process_file_job, request.file_id, job_id)

        return {
            "job_id": job_id,
            "file_id": request.file_id,
            "status": "queued",
            "processing_type": request.processing_type
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Start processing error: {e}")
        raise HTTPException(status_code=500, detail="Failed to start processing")

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    try:
        if job_id not in jobs_data:
            raise HTTPException(status_code=404, detail="Job not found")
        job = jobs_data[job_id]
        return {
            "job_id": job_id,
            "file_id": job["file_id"],
            "type": job["type"],
            "status": job["status"],
            "progress": job.get("progress", 0),
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
            "estimated_completion": job.get("estimated_completion"),
            "steps": job.get("steps", []),
            "error": job.get("error"),
            "result": job.get("result")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get job status error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get job status")

# ================================
# INSIGHTS
# ================================
@app.post("/api/insights/{file_id}/summary")
async def request_summary_generation(file_id: str, request: Optional[InsightRequest] = None, background_tasks: BackgroundTasks = None):
    try:
        if file_id not in files_data:
            raise HTTPException(status_code=404, detail="File not found")
        file_data = files_data[file_id]
        if file_data.get("processing_status") != "completed":
            raise HTTPException(status_code=400, detail="File processing not completed")

        extraction = file_data.get("extraction")
        if not extraction or not extraction.get("full_text"):
            raise HTTPException(status_code=400, detail="Extracted text not available")

        document_text = extraction["full_text"]

        if 'gemini_analyzer' not in available_services:
            raise HTTPException(status_code=500, detail="AI analyzer service not available")
        analyzer_class = available_services['gemini_analyzer']
        analyzer = analyzer_class()
        summary_result = await analyzer.analyze_document_comprehensive(
            text=document_text,
            query="Generate a comprehensive summary",
            language="en",
            filename=file_data['original_filename']
        )

        summary_id = generate_id("summary")
        summary_data = {
            "summary_id": summary_id,
            "file_id": file_id,
            "status": "completed",
            "created_at": get_utc_timestamp(),
            "summary_text": summary_result.get('summary', ''),
            "key_points": summary_result.get('key_risks', [])[:5],
            "summary_url": f"/api/insights/{file_id}/summary/download/{summary_id}",
            "word_count": len(summary_result.get('summary', '').split()),
            "confidence_score": 0.85
        }

        files_data[file_id].setdefault("insights", {})
        files_data[file_id]["insights"]["summary"] = summary_data
        return summary_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summary generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")

@app.post("/api/insights/{file_id}/audio")
async def request_audio_generation(file_id: str, request: Optional[InsightRequest] = None):
    try:
        if file_id not in files_data:
            raise HTTPException(status_code=404, detail="File not found")
        file_data = files_data[file_id]
        if file_data.get("processing_status") != "completed":
            raise HTTPException(status_code=400, detail="File processing not completed")

        if file_data.get("insights", {}).get("audio"):
            return file_data["insights"]["audio"]

        if 'voice_generator' not in available_services:
            raise HTTPException(status_code=500, detail="Voice generator service not available")

        summary_data = file_data.get("insights", {}).get("summary")
        if not summary_data:
            document_analysis = file_data.get('analysis', {})
            text_for_audio = document_analysis.get('summary', 'Document analysis completed.')
        else:
            text_for_audio = summary_data.get('summary_text', 'Document analysis completed.')

        voice_gen_class = available_services['voice_generator']
        voice_generator = voice_gen_class()

        options = request.options if request else {}
        voice_result = await voice_generator.generate_voice_summary(
            text_content=text_for_audio,
            language=options.get("language", "en"),
            voice_type=options.get("voice_type", "female"),
            speed=options.get("speed", 1.0)
        )

        # Prefer explicit GCS path if the service returns it; else use audio_url if it's gs://
        gcs_path = voice_result.get('gcs_path')
        audio_url = voice_result.get('audio_url', '')
        if not gcs_path and audio_url.startswith("gs://"):
            gcs_path = audio_url

        audio_id = generate_id("audio")
        audio_data = {
            "audio_id": audio_id,
            "file_id": file_id,
            "status": "completed",
            "created_at": get_utc_timestamp(),
            "audio_url": f"/api/insights/{file_id}/audio/download/{audio_id}",  # API download endpoint for frontend
            "duration": voice_result.get('duration', '1:30'),
            "duration_seconds": voice_result.get('duration_seconds', 90),
            "voice_type": voice_result.get('voice_type', 'female'),
            "language": voice_result.get('language', 'en'),
            "speed": options.get("speed", 1.0),
            "transcript": voice_result.get('transcript', text_for_audio),
            "file_size": voice_result.get('file_size', 1024000),
            "mime_type": voice_result.get('mime_type', 'audio/mpeg'),
            # Real sources to stream from:
            "gcs_path": gcs_path,                           # e.g. gs://bucket/voice_summary_xxx.mp3
            "real_audio_url": audio_url if audio_url.startswith("http") else None  # https, if provided
        }

        files_data[file_id].setdefault("insights", {})
        files_data[file_id]["insights"]["audio"] = audio_data
        return audio_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate audio: {str(e)}")


@app.post("/api/insights/{file_id}/report")
async def request_report_generation(file_id: str, request: Optional[InsightRequest] = None):
    try:
        if file_id not in files_data:
            raise HTTPException(status_code=404, detail="File not found")
        file_data = files_data[file_id]
        if file_data.get("processing_status") != "completed":
            raise HTTPException(status_code=400, detail="File processing not completed")

        if file_data.get("insights", {}).get("report"):
            return file_data["insights"]["report"]

        if 'report_generator' not in available_services:
            raise HTTPException(status_code=500, detail="Report generator service not available")

        document_analysis = file_data.get('analysis', {})
        if not document_analysis:
            raise HTTPException(status_code=400, detail="Document analysis not available")

        report_gen_class = available_services['report_generator']
        report_generator = report_gen_class()

        options = request.options if request else {}

        report_document_data = {
            'document_id': file_id,
            'title': file_data['original_filename'],
            'document_type': document_analysis.get('document_type', 'legal_document'),
            'summary': document_analysis.get('summary', ''),
            'overall_risk_score': document_analysis.get('overall_risk_score', 5.0),
            'key_risks': document_analysis.get('key_risks', []),
            'risk_categories': document_analysis.get('risk_categories', {}),
            'flagged_clauses': document_analysis.get('flagged_clauses', []),
            'recommendations': document_analysis.get('recommendations', []),
            'user_obligations': document_analysis.get('user_obligations', []),
            'user_rights': document_analysis.get('user_rights', []),
            'financial_implications': document_analysis.get('financial_implications', {}),
            'fairness_score': document_analysis.get('fairness_score', 5.0),
            'metadata': {
                'word_count': len(document_analysis.get('full_text', '').split()),
                'analysis_date': get_utc_timestamp()
            }
        }

        report_result = await report_generator.generate_comprehensive_report(
            document_data=report_document_data,
            report_type=options.get("type", "comprehensive"),
            language=options.get("language", "en"),
            include_charts=options.get("include_charts", True)
        )

        report_id = generate_id("report")
        fmt = str(report_result.get('report_data', {}).get('format', 'pdf')).lower()

        # Capture sources to stream later
        gcs_path = report_result.get('gcs_path')
        download_url = report_result.get('download_url')  # often a signed HTTPS URL
        if not gcs_path:
            # Some generators store by file_id; if you know the StorageManager key, you can reconstruct:
            # gcs_path = f"gs://{settings.GCS_BUCKET_NAME}/reports/{file_id}.pdf"
            # But prefer what the service returns:
            pass

        report_data = {
            "report_id": report_id,
            "file_id": file_id,
            "status": "completed",
            "created_at": get_utc_timestamp(),
            "report_url": f"/api/insights/{file_id}/report/download/{report_id}",  # API download endpoint for frontend
            "report_type": options.get("type", "comprehensive"),
            "format": fmt,
            "page_count": report_result.get('report_data', {}).get('page_count', 5),
            "file_size": report_result.get('pdf_size', 1024000),
            "sections": report_result.get('report_data', {}).get('sections', []),
            # Real sources to stream from:
            "gcs_path": gcs_path if (gcs_path and gcs_path.startswith("gs://")) else None,
            "real_download_url": download_url if (download_url and download_url.startswith("http")) else None
        }

        files_data[file_id].setdefault("insights", {})
        files_data[file_id]["insights"]["report"] = report_data
        return report_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


@app.get("/api/insights/{file_id}")
async def get_insights_status(file_id: str):
    try:
        if file_id not in files_data:
            raise HTTPException(status_code=404, detail="File not found")
        file_data = files_data.get(file_id)
        if not file_data:
            raise HTTPException(status_code=404, detail="File data not found")

        insights = file_data.get("insights", {})
        summary_insight = insights.get("summary")
        audio_insight = insights.get("audio")
        report_insight = insights.get("report")

        return {
            "file_id": file_id,
            "insights": {
                "summary": {
                    "available": summary_insight is not None,
                    "summary_id": summary_insight.get("summary_id") if summary_insight else None,
                    "url": summary_insight.get("summary_url") if summary_insight else None,
                    "created_at": summary_insight.get("created_at") if summary_insight else None,
                    "word_count": summary_insight.get("word_count") if summary_insight else None
                },
                "audio": {
                    "available": audio_insight is not None,
                    "audio_id": audio_insight.get("audio_id") if audio_insight else None,
                    "url": audio_insight.get("audio_url") if audio_insight else None,
                    "created_at": audio_insight.get("created_at") if audio_insight else None,
                    "duration": audio_insight.get("duration") if audio_insight else None
                },
                "report": {
                    "available": report_insight is not None,
                    "report_id": report_insight.get("report_id") if report_insight else None,
                    "url": report_insight.get("report_url") if report_insight else None,
                    "created_at": report_insight.get("created_at") if report_insight else None,
                    "page_count": report_insight.get("page_count") if report_insight else None
                }
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get insights error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get insights status")

# Downloads
@app.get("/api/insights/{file_id}/summary/download/{summary_id}")
async def download_summary(file_id: str, summary_id: str):
    try:
        if file_id not in files_data:
            raise HTTPException(status_code=404, detail="File not found")
        
        summary_data = files_data[file_id].get("insights", {}).get("summary")
        if not summary_data or summary_data["summary_id"] != summary_id:
            raise HTTPException(status_code=404, detail="Summary not found")

        content = f"""LEGALMIND AI - DOCUMENT SUMMARY

File: {files_data[file_id]['original_filename']}
Generated: {summary_data['created_at']}

EXECUTIVE SUMMARY
{summary_data.get('summary_text','')}

KEY POINTS
{chr(10).join([f"• {p}" for p in summary_data.get('key_points', [])])}

CONFIDENCE SCORE: {summary_data.get('confidence_score')}
WORD COUNT: {summary_data.get('word_count')}
"""
        return StreamingResponse(
            BytesIO(content.encode("utf-8")),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename=summary_{summary_id}.txt"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download summary error: {e}")
        raise HTTPException(status_code=500, detail="Failed to download summary")

@app.get("/api/insights/{file_id}/audio/download/{audio_id}")
async def download_audio(file_id: str, audio_id: str):
    try:
        if file_id not in files_data:
            raise HTTPException(status_code=404, detail="File not found")

        audio_data = files_data[file_id].get("insights", {}).get("audio")
        if not audio_data or audio_data["audio_id"] != audio_id:
            raise HTTPException(status_code=404, detail="Audio not found")

        mime = audio_data.get("mime_type", "audio/mpeg")
        filename = f"audio_{audio_id}.mp3" if mime == "audio/mpeg" else f"audio_{audio_id}"

        gcs_path = audio_data.get("gcs_path")
        https_url = audio_data.get("real_audio_url")

        if gcs_path and gcs_path.startswith("gs://"):
            data = await asyncio.to_thread(_download_gcs_as_bytes, gcs_path)
            return StreamingResponse(BytesIO(data), media_type=mime,
                                     headers={"Content-Disposition": f"attachment; filename={filename}"})

        if https_url and https_url.startswith("http"):
            async with aiohttp.ClientSession() as session:
                async with session.get(https_url) as resp:
                    if resp.status != 200:
                        raise HTTPException(status_code=502, detail="Upstream audio fetch failed")
                    data = await resp.read()
                    content_type = resp.headers.get("Content-Type", mime)
                    return StreamingResponse(BytesIO(data), media_type=content_type,
                                             headers={"Content-Disposition": f"attachment; filename={filename}"})

        # If we get here, we don't have a real source to fetch.
        raise HTTPException(status_code=500, detail="Audio artifact location missing (no GCS or HTTPS URL)")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download audio error: {e}")
        raise HTTPException(status_code=500, detail="Failed to download audio")


@app.get("/api/insights/{file_id}/report/download/{report_id}")
async def download_report(file_id: str, report_id: str):
    try:
        if file_id not in files_data:
            raise HTTPException(status_code=404, detail="File not found")

        report_data = files_data[file_id].get("insights", {}).get("report")
        if not report_data or report_data["report_id"] != report_id:
            raise HTTPException(status_code=404, detail="Report not found")

        fmt = (report_data.get("format") or "pdf").lower()
        filename = f"report_{report_id}.pdf" if fmt == "pdf" else f"report_{report_id}.{fmt}"

        gcs_path = report_data.get("gcs_path")
        https_url = report_data.get("real_download_url")

        if gcs_path and gcs_path.startswith("gs://"):
            data = await asyncio.to_thread(_download_gcs_as_bytes, gcs_path)
            return StreamingResponse(BytesIO(data),
                                     media_type="application/pdf" if fmt == "pdf" else "application/octet-stream",
                                     headers={"Content-Disposition": f"attachment; filename={filename}"})

        if https_url and https_url.startswith("http"):
            async with aiohttp.ClientSession() as session:
                async with session.get(https_url) as resp:
                    if resp.status != 200:
                        raise HTTPException(status_code=502, detail="Upstream report fetch failed")
                    data = await resp.read()
                    content_type = resp.headers.get("Content-Type", "application/pdf")
                    return StreamingResponse(BytesIO(data), media_type=content_type,
                                             headers={"Content-Disposition": f"attachment; filename={filename}"})

        # If we get here, we don't have a real source to fetch.
        # Return a small diagnostic text so it's obvious in the UI.
        diag = f"""Report not found at source.
gcs_path: {gcs_path}
real_download_url: {https_url}
"""
        return StreamingResponse(BytesIO(diag.encode("utf-8")),
                                 media_type="text/plain; charset=utf-8",
                                 headers={"Content-Disposition": f"attachment; filename={filename.replace('.pdf','.txt')}"})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download report error: {e}")
        raise HTTPException(status_code=500, detail="Failed to download report")


# ================================
# CHAT
# ================================
# --- REPLACE the whole /api/chat route with this ---
@app.post("/api/chat")
async def sync_chat(request: ChatRequest):
    try:
        conversation_id = request.conversation_id or generate_id("conv")

        if conversation_id not in conversations_data:
            conversations_data[conversation_id] = {
                "conversation_id": conversation_id,
                "created_at": get_utc_timestamp(),
                "messages": [],
                "file_id": request.file_id,
                "context": {}
            }

        conversation = conversations_data[conversation_id]

        # ---- Build file_context with actual document text ----
        file_context: Dict[str, Any] = {}
        if request.file_id:
            if request.file_id not in files_data:
                raise HTTPException(status_code=400, detail="Unknown file_id")

            fd = files_data[request.file_id]

            analysis = fd.get("analysis", {}) or {}
            chunks = fd.get("chunks", []) or []

            # Prefer chunk body; fallback only to analysis.full_text that we produced earlier
            def _chunk_texts():
                out = []
                for c in chunks:
                    txt = c.get("content") or c.get("text") or ""
                    if txt:
                        out.append(txt)
                return out

            combined = "\n\n".join(_chunk_texts())
            if not combined:
                combined = analysis.get("full_text", "") or ""

            if not combined.strip():
                raise HTTPException(status_code=409, detail="No extracted text available for chat. Ensure DocAI and chunking finished successfully.")

            safe_text = clean_text_for_gemini(combined)

            file_context = {
                "filename": fd.get("original_filename"),
                "file_type": fd.get("metadata", {}).get("file_type"),
                "analysis": analysis,
                "document_text": safe_text,
                "chunks_meta": [
                    {
                        "index": i,
                        "page_start": c.get("page_start"),
                        "page_end": c.get("page_end"),
                        "char_count": len((c.get("content") or c.get("text") or "")),
                    }
                    for i, c in enumerate(chunks)
                ],
            }

        # ---- STRICT: require chat_handler service ----
        if 'chat_handler' not in available_services:
            raise HTTPException(status_code=500, detail="chat_handler service not loaded")

        response_text = await generate_chat_response(
            request.message, file_context, conversation["messages"]
        )

        user_message = {
            "message_id": generate_id("msg"),
            "role": "user",
            "content": request.message,
            "timestamp": get_utc_timestamp(),
            "file_id": request.file_id
        }
        assistant_message = {
            "message_id": generate_id("msg"),
            "role": "assistant",
            "content": response_text,
            "timestamp": get_utc_timestamp(),
            "confidence": 0.89,
            "sources": [file_context] if file_context else []
        }

        conversation["messages"].extend([user_message, assistant_message])
        conversation["last_activity"] = get_utc_timestamp()

        return {
            "conversation_id": conversation_id,
            "message_id": assistant_message["message_id"],
            "response": response_text,
            "confidence": 0.89,
            "sources": [file_context] if file_context else [],
            "suggestions": [
                "Can you explain this in more detail?",
                "What are the main risks?",
                "How does this compare to standard contracts?",
                "What should I be careful about?"
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {e}")


# --- REPLACE the whole generate_chat_response helper with this ---
async def generate_chat_response(message: str, file_context: Dict, conversation_history: List) -> str:
    try:
        if 'chat_handler' not in available_services:
            raise RuntimeError("chat_handler service not loaded")

        chat_handler_class = available_services['chat_handler']
        chat_handler = chat_handler_class()

        # Strict requirement: if a file_id was provided earlier, we expect document_text here
        if file_context and not file_context.get("document_text"):
            raise RuntimeError("Missing document_text in file_context")

        result = await chat_handler.process_message(
            message=message,
            context=file_context or {},
            conversation_history=conversation_history or []
        )

        resp = result.get('response')
        if not resp or not resp.strip():
            raise RuntimeError("Empty response from chat handler")
        return resp

    except Exception as e:
        logger.error(f"Chat response error: {e}")
        # No fallback; make it obvious in UI/logs
        return f"Error: {e}"



@app.get("/api/chat/stream")
async def stream_chat():
    return {"message": "Streaming chat not implemented yet", "use": "/api/chat for sync chat"}

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            response = {
                "type": "response",
                "message": f"Echo: {message_data.get('message', '')}",
                "timestamp": get_utc_timestamp()
            }
            await websocket.send_text(json.dumps(response))
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

# ================================
# EXPORTS
# ================================
@app.post("/api/export/conversation")
async def export_conversation(request: ExportConversationRequest):
    try:
        if request.conversation_id not in conversations_data:
            raise HTTPException(status_code=404, detail="Conversation not found")

        conversation = conversations_data[request.conversation_id]
        export_id = generate_id("export")

        if request.format == "pdf":
            content = f"""LEGALMIND AI - CONVERSATION EXPORT

Conversation ID: {request.conversation_id}
Created: {conversation['created_at']}
Exported: {get_utc_timestamp()}
Messages: {len(conversation['messages'])}

{'=' * 50}

CONVERSATION TRANSCRIPT
"""
            for msg in conversation["messages"]:
                role = "USER" if msg["role"] == "user" else "ASSISTANT"
                content += f"\n[{msg['timestamp']}] {role}: {msg['content']}\n"
            content += f"\n{'=' * 50}\nExported by LegalMind AI"
        elif request.format == "json":
            content = json.dumps(conversation, indent=2)
        else:
            content = "LegalMind AI Conversation Export\n\n"
            for msg in conversation["messages"]:
                role = "You" if msg["role"] == "user" else "AI Assistant"
                content += f"{role}: {msg['content']}\n\n"

        export_data = {
            "export_id": export_id,
            "conversation_id": request.conversation_id,
            "format": request.format,
            "created_at": get_utc_timestamp(),
            "file_size": len(content.encode()),
            "download_url": f"/api/export/{export_id}/download"
        }

        conversations_data[request.conversation_id].setdefault("exports", {})
        conversations_data[request.conversation_id]["exports"][export_id] = {
            "data": content,
            "metadata": export_data
        }

        return {
            "export_id": export_id,
            "conversation_id": request.conversation_id,
            "format": request.format,
            "file_size": export_data["file_size"],
            "download_url": export_data["download_url"],
            "created_at": export_data["created_at"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export conversation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to export conversation")

@app.get("/api/export/{export_id}/download")
async def download_export(export_id: str):
    try:
        for conversation in conversations_data.values():
            exports = conversation.get("exports", {})
            if export_id in exports:
                export_info = exports[export_id]
                content = export_info["data"]
                metadata = export_info["metadata"]
                format_type = metadata["format"]
                if format_type == "pdf":
                    media_type = "application/pdf"
                elif format_type == "json":
                    media_type = "application/json"
                else:
                    media_type = "text/plain"
                return StreamingResponse(
                    BytesIO(content.encode()),
                    media_type=media_type,
                    headers={"Content-Disposition": f"attachment; filename=conversation_{export_id}.{format_type}"}
                )
        raise HTTPException(status_code=404, detail="Export not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download export error: {e}")
        raise HTTPException(status_code=500, detail="Failed to download export")

# ================================
# INTERNAL JOB STATUS
# ================================
@app.post("/internal/job-status")
async def update_job_status(request: Request):
    try:
        body = await request.json()
        job_id = body.get("job_id")
        status = body.get("status")
        progress = body.get("progress", 0)
        if job_id in jobs_data:
            jobs_data[job_id]["status"] = status
            jobs_data[job_id]["progress"] = progress
            jobs_data[job_id]["updated_at"] = get_utc_timestamp()
            if status == "completed":
                jobs_data[job_id]["completed_at"] = get_utc_timestamp()
            elif status == "failed":
                jobs_data[job_id]["error"] = body.get("error", "Processing failed")
        return {"status": "updated", "job_id": job_id}
    except Exception as e:
        logger.error(f"Job status update error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update job status")

# ================================
# HEALTH & ROOT
# ================================
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "LegalMind AI API",
        "version": "2.0.0",
        "timestamp": get_utc_timestamp(),
        "services": {
            "loaded": len(available_services),
            "available": list(available_services.keys())
        },
        "storage": {
            "files": len(files_data),
            "jobs": len(jobs_data),
            "conversations": len(conversations_data)
        }
    }

@app.get("/")
async def root():
    return {
        "service": "LegalMind AI - Frontend Integration API",
        "version": "2.0.0",
        "status": "operational",
        "documentation": "/docs",
        "health": "/health",
        "endpoints": {
            "essential": [
                "POST /api/uploads/get-signed-url",
                "POST /api/uploads/notify-uploaded",
                "GET /api/uploads",
                "GET /api/jobs/{job_id}",
                "POST /api/insights/{file_id}/summary",
                "POST /api/chat",
                "POST /api/export/conversation"
            ],
            "optional": [
                "POST /api/auth/verify-token",
                "POST /api/process/start",
                "GET /api/chat/stream",
                "POST /internal/job-status"
            ]
        }
    }

# ================================
# GLOBAL ERROR HANDLER
# ================================
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error in {request.url}: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": get_utc_timestamp(),
            "path": str(request.url)
        }
    )

if __name__ == "__main__":
    logger.info("🚀 Starting LegalMind AI API Server...")
    logger.info("📚 API Documentation: http://localhost:8000/docs")
    logger.info(f"🛠️ Services loaded: {len(available_services)}")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
