"""
Google Cloud Storage manager for document handling
"""
import os
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from google.cloud import storage
from shared.auth import get_google_credentials
from shared.constants import GCS_BUCKET_DOCUMENTS, GCS_BUCKET_RESULTS, GCS_BUCKET_TEMP
import json


class GCSManager:
    def __init__(self):
        try:
            credentials, self.project_id = get_google_credentials()
            self.client = storage.Client(credentials=credentials, project=self.project_id)
            self.documents_bucket = self.client.bucket(GCS_BUCKET_DOCUMENTS)
            self.results_bucket = self.client.bucket(GCS_BUCKET_RESULTS)
            self.temp_bucket = self.client.bucket(GCS_BUCKET_TEMP)
        except Exception as e:
            print(f"Warning: GCS not configured: {e}")
            self.client = None
    
    def upload_document(self, file_content: bytes, filename: str, user_id: str = None) -> str:
        """Upload document to GCS and return GCS path"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            
            if user_id:
                gcs_path = f"users/{user_id}/{timestamp}_{unique_id}_{filename}"
            else:
                gcs_path = f"documents/{timestamp}_{unique_id}_{filename}"
            
            blob = self.documents_bucket.blob(gcs_path)
            blob.upload_from_string(file_content)
            
            # Add metadata
            metadata = {
                'uploaded_at': datetime.now().isoformat(),
                'original_filename': filename,
                'user_id': user_id or 'anonymous'
            }
            blob.metadata = metadata
            blob.patch()
            
            return f"gs://{GCS_BUCKET_DOCUMENTS}/{gcs_path}"
            
        except Exception as e:
            raise Exception(f"Failed to upload document to GCS: {str(e)}")
    
    def download_document(self, gcs_path: str) -> bytes:
        """Download document from GCS"""
        try:
            # Parse GCS path
            if gcs_path.startswith('gs://'):
                bucket_name = gcs_path.split('/')[2]
                blob_path = '/'.join(gcs_path.split('/')[3:])
            else:
                bucket_name = GCS_BUCKET_DOCUMENTS
                blob_path = gcs_path
            
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            return blob.download_as_bytes()
            
        except Exception as e:
            raise Exception(f"Failed to download document from GCS: {str(e)}")
    
    def upload_chunk(self, chunk_content: bytes, original_path: str, chunk_index: int) -> str:
        """Upload document chunk to temp bucket"""
        try:
            base_path = original_path.replace(f"gs://{GCS_BUCKET_DOCUMENTS}/", "")
            chunk_path = f"chunks/{base_path}_chunk_{chunk_index}.pdf"
            
            blob = self.temp_bucket.blob(chunk_path)
            blob.upload_from_string(chunk_content)
            
            return f"gs://{GCS_BUCKET_TEMP}/{chunk_path}"
            
        except Exception as e:
            raise Exception(f"Failed to upload chunk to GCS: {str(e)}")
    
    def save_analysis_results(self, results: Dict[str, Any], task_id: str) -> str:
        """Save analysis results to GCS"""
        try:
            results_path = f"analysis/{task_id}.json"
            blob = self.results_bucket.blob(results_path)
            
            results_with_metadata = {
                'task_id': task_id,
                'timestamp': datetime.now().isoformat(),
                'results': results
            }
            
            blob.upload_from_string(json.dumps(results_with_metadata, indent=2))
            return f"gs://{GCS_BUCKET_RESULTS}/{results_path}"
            
        except Exception as e:
            raise Exception(f"Failed to save results to GCS: {str(e)}")
    
    def load_analysis_results(self, task_id: str) -> Dict[str, Any]:
        """Load analysis results from GCS"""
        try:
            results_path = f"analysis/{task_id}.json"
            blob = self.results_bucket.blob(results_path)
            
            if not blob.exists():
                return None
            
            content = blob.download_as_text()
            return json.loads(content)
            
        except Exception as e:
            raise Exception(f"Failed to load results from GCS: {str(e)}")
    
    def cleanup_temp_files(self, gcs_paths: List[str]):
        """Clean up temporary files"""
        try:
            for path in gcs_paths:
                if path.startswith(f"gs://{GCS_BUCKET_TEMP}/"):
                    blob_path = path.replace(f"gs://{GCS_BUCKET_TEMP}/", "")
                    blob = self.temp_bucket.blob(blob_path)
                    if blob.exists():
                        blob.delete()
        except Exception as e:
            print(f"Warning: Failed to cleanup temp files: {e}")
