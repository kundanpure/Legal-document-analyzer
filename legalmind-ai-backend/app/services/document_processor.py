"""
REAL Document AI integration with page-batching (avoids 15-page limit)
"""

import asyncio
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List
import logging

# Google Cloud Document AI imports
from google.cloud import documentai
from google.api_core import exceptions as google_exceptions
from google.oauth2 import service_account

# Try to read page count locally (preferred). Works with pypdf or PyPDF2.
def _count_pdf_pages(file_path: str) -> int:
    try:
        try:
            from pypdf import PdfReader  # modern
        except Exception:
            from PyPDF2 import PdfReader  # fallback
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            return len(reader.pages)
    except Exception:
        # As a last resort, return a large number to force batching; real count
        # will still be safe because we select pages explicitly per request.
        return 9999

from config.settings import get_settings
from config.logging import get_logger
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)
settings = get_settings()

class DocumentProcessingError(Exception):
    """Exception raised for document processing errors"""
    pass


class RealDocumentProcessor:
    """REAL Document AI processor with batching to respect page limits"""

    def __init__(self):
        self.settings = settings
        self.logger = logger

        self.client = None
        self.processor_name = None

        try:
            # ---- Credentials (must exist) ----
            credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if not credentials_path or not os.path.exists(credentials_path):
                raise DocumentProcessingError(
                    f"GOOGLE_APPLICATION_CREDENTIALS not found or invalid: {credentials_path}"
                )
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            sa_email = getattr(credentials, "service_account_email", "unknown")

            # ---- Required envs ----
            self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
            self.location = os.getenv('DOCUMENT_AI_LOCATION', 'us')
            self.processor_id = os.getenv('DOCUMENT_AI_PROCESSOR_ID')

            if not self.project_id:
                raise DocumentProcessingError("Missing env GOOGLE_CLOUD_PROJECT")
            if not self.processor_id:
                raise DocumentProcessingError("Missing env DOCUMENT_AI_PROCESSOR_ID")
            if not self.location:
                raise DocumentProcessingError("Missing env DOCUMENT_AI_LOCATION")

            # ---- Regional endpoint (CRITICAL) ----
            api_endpoint = f"{self.location}-documentai.googleapis.com"
            self.client = documentai.DocumentProcessorServiceClient(
                credentials=credentials,
                client_options={"api_endpoint": api_endpoint},
            )

            # ---- Build processor resource name ----
            self.processor_name = self.client.processor_path(
                self.project_id, self.location, self.processor_id
            )

            # ---- Verify access/exists ----
            try:
                processor_info = self.client.get_processor(name=self.processor_name)
                self.logger.info("✅ Document AI processor verified: %s",
                                 getattr(processor_info, "display_name", self.processor_id))
            except google_exceptions.PermissionDenied as e:
                raise DocumentProcessingError(
                    f"Permission denied accessing processor {self.processor_id}. "
                    f"Grant roles/documentai.viewer and roles/documentai.apiUser to {sa_email} "
                    f"in project {self.project_id}. Underlying error: {e}"
                )
            except google_exceptions.NotFound as e:
                raise DocumentProcessingError(
                    f"Processor {self.processor_id} not found in project {self.project_id} at location {self.location}. "
                    f"Underlying error: {e}"
                )

            self.logger.info("RealDocumentProcessor initialized successfully")
            self.logger.info("GOOGLE_APPLICATION_CREDENTIALS: %s", credentials_path)
            self.logger.info("Using service account: %s", sa_email)
            self.logger.info(
                "DocAI target -> project=%s location=%s processor=%s endpoint=%s",
                self.project_id, self.location, self.processor_id, api_endpoint
            )

        except Exception as e:
            self.logger.error("Document AI initialization failed: %s", e)
            raise DocumentProcessingError(f"Failed to initialize Document AI: {e}")

    # -----------------------------
    # Public API
    # -----------------------------
    async def extract_text_from_pdf(self, file_path: str) -> Dict[str, Any]:
        """REAL text extraction using Google Document AI, with page-batching."""
        try:
            self.logger.info(f"Starting REAL Document AI extraction: {file_path}")

            if not os.path.exists(file_path):
                raise DocumentProcessingError(f"File not found: {file_path}")

            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise DocumentProcessingError("File is empty")

            if file_size > 20 * 1024 * 1024:  # 20MB basic safety; adjust if your processor allows larger
                raise DocumentProcessingError("File too large for Document AI processing")

            with open(file_path, "rb") as f:
                file_content = f.read()

            if not file_content.startswith(b"%PDF-"):
                raise DocumentProcessingError("File is not a valid PDF")

            # Count pages locally so we can batch safely
            total_pages = _count_pdf_pages(file_path)
            if total_pages <= 0:
                total_pages = 1

            # Batch size to respect the 15-page limit in non-imageless mode
            batch_size = 15

            if total_pages <= batch_size:
                # Single call
                resp = await self._process_with_document_ai(file_content)
                self.logger.info("✅ Document AI extraction completed successfully")
                # Add an accurate page count
                resp["statistics"]["total_pages"] = total_pages
                return resp
            else:
                # Multiple calls, selecting individual page ranges per request
                self.logger.info(
                    "Document has %s pages; processing in %s-page batches",
                    total_pages, batch_size
                )
                merged = await self._process_in_batches(file_content, total_pages, batch_size)
                self.logger.info("✅ Document AI extraction (batched) completed successfully")
                return merged

        except DocumentProcessingError:
            raise
        except Exception as e:
            raise DocumentProcessingError(f"Extraction failed: {str(e)}")

    # -----------------------------
    # Internal helpers
    # -----------------------------
    async def _process_with_document_ai(self, file_content: bytes,
                                        pages: List[int] = None) -> Dict[str, Any]:
        """
        Process document using Google Document AI.
        If 'pages' is provided, only those pages will be processed (1-based indices).
        """
        try:
            raw_document = documentai.RawDocument(
                content=file_content,
                mime_type="application/pdf",
            )

            # Optional page filtering via ProcessOptions
            process_options = None
            if pages:
                process_options = documentai.ProcessOptions(
                    individual_page_selector=documentai.ProcessOptions.IndividualPageSelector(
                        pages=pages
                    )
                )

            request = documentai.ProcessRequest(
                name=self.processor_name,
                raw_document=raw_document,
                process_options=process_options,
            )

            self.logger.info(
                "Sending document to Google Document AI...%s",
                (f" (pages={pages[0]}–{pages[-1]})" if pages else "")
            )

            response = await asyncio.to_thread(self.client.process_document, request=request)

            if not response or not response.document:
                raise DocumentProcessingError("Document AI returned empty response")

            document = response.document
            if not document.text:
                raise DocumentProcessingError("Document AI extracted no text from PDF")

            return self._extract_comprehensive_data(document)

        except google_exceptions.PermissionDenied as e:
            raise DocumentProcessingError(f"Permission denied: {e}")
        except google_exceptions.InvalidArgument as e:
            # surfacing original error message is helpful
            raise DocumentProcessingError(f"Invalid request: {e}")
        except google_exceptions.ResourceExhausted as e:
            raise DocumentProcessingError(f"Quota exceeded: {e}")
        except Exception as e:
            raise DocumentProcessingError(f"Document AI processing failed: {e}")

    async def _process_in_batches(self, file_content: bytes, total_pages: int, batch_size: int) -> Dict[str, Any]:
        """Process the file in multiple requests (1..15, 16..30, ...) and merge outputs."""
        batches: List[Dict[str, Any]] = []
        page = 1
        while page <= total_pages:
            end = min(page + batch_size - 1, total_pages)
            pages = list(range(page, end + 1))
            part = await self._process_with_document_ai(file_content, pages=pages)
            batches.append(part)
            page = end + 1

        # Merge batches
        return self._merge_batches(batches, total_pages)

    def _merge_batches(self, parts: List[Dict[str, Any]], total_pages: int) -> Dict[str, Any]:
        """Merge multiple per-range extraction dicts into a single dict."""
        merged_texts: List[str] = []
        merged_pages: List[Dict[str, Any]] = []
        merged_tables: List[Dict[str, Any]] = []
        merged_forms: List[Dict[str, Any]] = []

        char_offset = 0
        for part in parts:
            text = part.get("full_text", "")
            # Keep a clean separator between batches
            if merged_texts and text:
                merged_texts.append("\n\n")
            merged_texts.append(text)

            # pages
            for p in part.get("pages", []):
                merged_pages.append(p)

            # tables/forms (just append; page numbers are already 1-based from DocAI)
            merged_tables.extend(part.get("tables", []))
            merged_forms.extend(part.get("form_fields", []))

            char_offset += len(text)

        full_text = "".join(merged_texts)

        statistics = {
            "total_pages": total_pages,
            "total_characters": len(full_text),
            "total_words": len(full_text.split()),
            "total_paragraphs": sum(p.get("paragraphs_count", 0) for p in merged_pages),
            "total_tables": len(merged_tables),
            "total_form_fields": len(merged_forms),
            "confidence_score": 0.95,
            "processing_time": None,
            "language_detected": self._detect_language(full_text),
        }

        return {
            "full_text": full_text,
            "raw_text": full_text,  # keeping same for consistency
            "pages": merged_pages,
            "tables": merged_tables,
            "form_fields": merged_forms,
            "statistics": statistics,
            "extraction_metadata": {
                "processor_version": "Google Cloud Document AI",
                "processor_id": self.processor_id,
                "mime_type": "application/pdf",
                "extraction_method": "document_ai_ocr_batched",
            },
        }

    # -----------------------------
    # Data shaping helpers (unchanged)
    # -----------------------------
    def _extract_comprehensive_data(self, document) -> Dict[str, Any]:
        """Extract comprehensive data from Document AI response"""

        full_text = self._clean_extracted_text(document.text)

        pages_data = []
        for i, page in enumerate(document.pages):
            page_info = {
                'page_number': i + 1,
                'dimensions': {
                    'width': page.dimension.width if page.dimension else 0,
                    'height': page.dimension.height if page.dimension else 0,
                    'unit': page.dimension.unit if page.dimension else 'pixels'
                },
                'blocks_count': len(page.blocks) if page.blocks else 0,
                'paragraphs_count': len(page.paragraphs) if page.paragraphs else 0,
                'lines_count': len(page.lines) if page.lines else 0,
                'tokens_count': len(page.tokens) if page.tokens else 0
            }
            pages_data.append(page_info)

        tables_data = []
        for page_idx, page in enumerate(document.pages):
            if page.tables:
                for table_idx, table in enumerate(page.tables):
                    table_info = self._extract_table_data(table, document.text, page_idx + 1, table_idx)
                    tables_data.append(table_info)

        form_fields = []
        for page_idx, page in enumerate(document.pages):
            if page.form_fields:
                for field in page.form_fields:
                    field_info = self._extract_form_field(field, document.text, page_idx + 1)
                    form_fields.append(field_info)

        statistics = {
            'total_pages': len(document.pages),
            'total_characters': len(full_text),
            'total_words': len(full_text.split()),
            'total_paragraphs': sum(len(page.paragraphs) if page.paragraphs else 0 for page in document.pages),
            'total_tables': len(tables_data),
            'total_form_fields': len(form_fields),
            'confidence_score': 0.95,
            'processing_time': None,
            'language_detected': self._detect_language(full_text)
        }

        return {
            'full_text': full_text,
            'raw_text': document.text,
            'pages': pages_data,
            'tables': tables_data,
            'form_fields': form_fields,
            'statistics': statistics,
            'extraction_metadata': {
                'processor_version': 'Google Cloud Document AI',
                'processor_id': self.processor_id,
                'mime_type': 'application/pdf',
                'extraction_method': 'document_ai_ocr'
            }
        }

    def _clean_extracted_text(self, raw_text: str) -> str:
        if not raw_text:
            return ""
        cleaned = raw_text.replace('\x00', '').replace('\ufeff', '')
        cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')
        lines = cleaned.split('\n')
        processed = [' '.join(line.split()) for line in lines]
        # collapse >2 blank lines
        out = []
        blank = 0
        for line in processed:
            if line.strip():
                blank = 0
                out.append(line)
            else:
                blank += 1
                if blank <= 2:
                    out.append("")
        return '\n'.join(out).strip()

    def _extract_table_data(self, table, full_text: str, page_number: int, table_index: int) -> Dict[str, Any]:
        header_rows = []
        if table.header_rows:
            for row in table.header_rows:
                header_cells = []
                for cell in row.cells:
                    cell_text = self._get_text_from_layout(cell.layout, full_text)
                    header_cells.append(cell_text.strip())
                header_rows.append(header_cells)

        body_rows = []
        if table.body_rows:
            for row in table.body_rows:
                row_cells = []
                for cell in row.cells:
                    cell_text = self._get_text_from_layout(cell.layout, full_text)
                    row_cells.append(cell_text.strip())
                body_rows.append(row_cells)

        return {
            'page_number': page_number,
            'table_index': table_index,
            'header_rows': header_rows,
            'body_rows': body_rows,
            'total_rows': len(header_rows) + len(body_rows),
            'total_columns': max(len(row) for row in (header_rows + body_rows)) if (header_rows + body_rows) else 0,
            'confidence': getattr(table, 'confidence', 0.9)
        }

    def _extract_form_field(self, field, full_text: str, page_number: int) -> Dict[str, Any]:
        field_name = ""
        field_value = ""
        field_confidence = 0.0

        if field.field_name and field.field_name.text_anchor:
            field_name = self._get_text_from_layout(field.field_name, full_text).strip()

        if field.field_value and field.field_value.text_anchor:
            field_value = self._get_text_from_layout(field.field_value, full_text).strip()
            field_confidence = getattr(field.field_value, 'confidence', 0.0)

        return {
            'page_number': page_number,
            'field_name': field_name,
            'field_value': field_value,
            'confidence': field_confidence,
            'field_type': self._classify_field_type(field_name, field_value)
        }

    def _get_text_from_layout(self, layout, full_text: str) -> str:
        if not layout or not layout.text_anchor:
            return ""
        text_segments = []
        for segment in layout.text_anchor.text_segments:
            start_idx = int(segment.start_index) if hasattr(segment, 'start_index') and segment.start_index else 0
            end_idx = int(segment.end_index) if hasattr(segment, 'end_index') and segment.end_index else len(full_text)
            if start_idx < len(full_text) and end_idx <= len(full_text) and start_idx < end_idx:
                text_segments.append(full_text[start_idx:end_idx])
        return "".join(text_segments)

    def _classify_field_type(self, field_name: str, field_value: str) -> str:
        name_lower = field_name.lower()
        if any(k in name_lower for k in ['email', '@']):
            return 'email'
        elif any(k in name_lower for k in ['phone', 'tel', 'mobile']):
            return 'phone'
        elif any(k in name_lower for k in ['date', 'time']):
            return 'date'
        elif any(k in name_lower for k in ['amount', 'price', 'cost', '$']):
            return 'currency'
        elif field_value.replace('.', '').replace(',', '').isdigit():
            return 'number'
        else:
            return 'text'

    def _detect_language(self, text: str) -> str:
        english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        words = text.lower().split()
        english_count = sum(1 for w in words if w in english_words)
        if len(words) > 0 and english_count / len(words) > 0.05:
            return 'en'
        else:
            return 'unknown'
