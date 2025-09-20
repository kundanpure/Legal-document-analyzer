"""
Google Cloud Storage integration for file management
Enhanced with fallbacks and comprehensive error handling
"""

import asyncio
import uuid
import os
from dotenv import load_dotenv
load_dotenv() 
import tempfile
import shutil
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
from pathlib import Path

from config.logging import get_logger

logger = get_logger(__name__)

# Try to import Google Cloud Storage with fallback
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
    logger.info("✅ Google Cloud Storage available - full cloud storage enabled")
except ImportError:
    GCS_AVAILABLE = False
    logger.warning("⚠️ Google Cloud Storage not available - using local storage fallback")

# Try to import aiofiles with fallback
try:
    import aiofiles
    AIOFILES_AVAILABLE = True
    logger.info("✅ aiofiles available - async file operations enabled")
except ImportError:
    AIOFILES_AVAILABLE = False
    logger.warning("⚠️ aiofiles not available - using sync file operations")

# Try to import settings with fallback
try:
    from config.settings import get_settings
    settings = get_settings()
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False
    logger.warning("⚠️ Settings not available - using default configuration")
    # Mock settings
    class MockSettings:
        GCS_BUCKET_NAME = "default-bucket"
        LOCAL_STORAGE_PATH = "./storage"
    settings = MockSettings()

# Custom exceptions
class StorageError(Exception):
    """Exception raised for storage-related errors"""
    pass

class StorageManager:
    """Professional storage management with Google Cloud Storage and local fallback"""
    
    def __init__(self):
        self.logger = logger
        self.use_gcs = GCS_AVAILABLE
        
        # Initialize storage based on availability
        if self.use_gcs:
            try:
                self.client = storage.Client()
                self.bucket_name = getattr(settings, 'GCS_BUCKET_NAME', 'default-bucket')
                self.bucket = self.client.bucket(self.bucket_name)
                logger.info(f"✅ GCS initialized with bucket: {self.bucket_name}")
            except Exception as e:
                logger.warning(f"⚠️ GCS initialization failed: {e}, falling back to local storage")
                self.use_gcs = False
        
        # Local storage configuration
        if not self.use_gcs:
            self.local_storage_path = Path(getattr(settings, 'LOCAL_STORAGE_PATH', './storage'))
            self.local_storage_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Local storage initialized at: {self.local_storage_path}")
        
        # Storage paths
        self.paths = {
            'documents': 'documents/',
            'audio': 'audio/',
            'reports': 'reports/',
            'temp': 'temp/'
        }
        
        # Create local directories if using local storage
        if not self.use_gcs:
            for category, path in self.paths.items():
                local_path = self.local_storage_path / path.rstrip('/')
                local_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"StorageManager initialized - Mode: {'GCS' if self.use_gcs else 'Local'}")

    async def upload_document(
        self, 
        file_content: bytes, 
        filename: str,
        document_id: str,
        content_type: str = "application/pdf"
    ) -> str:
        """
        Upload document to storage (GCS or local)
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            document_id: Unique document identifier
            content_type: MIME type of file
            
        Returns:
            Storage URI of uploaded file
        """
        
        try:
            self.logger.info(f"Uploading document: {filename} (ID: {document_id})")
            
            if self.use_gcs:
                return await self._upload_document_gcs(file_content, filename, document_id, content_type)
            else:
                return await self._upload_document_local(file_content, filename, document_id, content_type)
                
        except Exception as e:
            self.logger.error(f"Error uploading document: {str(e)}")
            raise StorageError(f"Failed to upload document: {str(e)}")

    async def _upload_document_gcs(self, file_content: bytes, filename: str, document_id: str, content_type: str) -> str:
        """Upload document to Google Cloud Storage"""
        
        # Generate storage path
        file_extension = Path(filename).suffix
        blob_name = f"{self.paths['documents']}{document_id}{file_extension}"
        
        # Create blob
        blob = self.bucket.blob(blob_name)
        
        # Set metadata
        blob.metadata = {
            'document_id': document_id,
            'original_filename': filename,
            'upload_date': datetime.now(timezone.utc).isoformat(),
            'content_type': content_type
        }
        
        # Upload file
        await asyncio.to_thread(
            blob.upload_from_string,
            file_content,
            content_type=content_type
        )
        
        # Set cache control
        blob.cache_control = 'private, max-age=3600'
        await asyncio.to_thread(blob.patch)
        
        gcs_uri = f"gs://{self.bucket_name}/{blob_name}"
        
        self.logger.info(f"Document uploaded to GCS: {gcs_uri}")
        return gcs_uri

    async def _upload_document_local(self, file_content: bytes, filename: str, document_id: str, content_type: str) -> str:
        """Upload document to local storage"""
        
        # Generate file path
        file_extension = Path(filename).suffix
        local_filename = f"{document_id}{file_extension}"
        file_path = self.local_storage_path / self.paths['documents'].rstrip('/') / local_filename
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        if AIOFILES_AVAILABLE:
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(file_content)
        else:
            with open(file_path, 'wb') as f:
                f.write(file_content)
        
        # Create metadata file
        metadata = {
            'document_id': document_id,
            'original_filename': filename,
            'upload_date': datetime.now(timezone.utc).isoformat(),
            'content_type': content_type,
            'file_size': len(file_content)
        }
        
        metadata_path = file_path.with_suffix(file_path.suffix + '.meta')
        if AIOFILES_AVAILABLE:
            async with aiofiles.open(metadata_path, 'w') as f:
                await f.write(str(metadata))
        else:
            with open(metadata_path, 'w') as f:
                f.write(str(metadata))
        
        local_uri = f"file://{file_path.absolute()}"
        
        self.logger.info(f"Document uploaded to local storage: {local_uri}")
        return local_uri

    async def save_audio_file(
        self, 
        audio_data: bytes, 
        filename: str,
        voice_id: str = None
    ) -> str:
        """
        Save audio file to storage
        
        Args:
            audio_data: Audio data as bytes
            filename: Audio filename
            voice_id: Optional voice identifier
            
        Returns:
            Public URL of audio file
        """
        
        try:
            self.logger.info(f"Saving audio file: {filename}")
            
            if self.use_gcs:
                return await self._save_audio_gcs(audio_data, filename, voice_id)
            else:
                return await self._save_audio_local(audio_data, filename, voice_id)
                
        except Exception as e:
            self.logger.error(f"Error saving audio file: {str(e)}")
            raise StorageError(f"Failed to save audio file: {str(e)}")

    async def _save_audio_gcs(self, audio_data: bytes, filename: str, voice_id: str = None) -> str:
        """Save audio file to GCS"""
        
        # Generate storage path
        blob_name = f"{self.paths['audio']}{filename}"
        
        # Create blob
        blob = self.bucket.blob(blob_name)
        
        # Set metadata
        blob.metadata = {
            'voice_id': voice_id or str(uuid.uuid4()),
            'created_date': datetime.now(timezone.utc).isoformat(),
            'file_size': str(len(audio_data))
        }
        
        # Upload audio
        await asyncio.to_thread(
            blob.upload_from_string,
            audio_data,
            content_type='audio/mpeg'
        )
        
        # Make publicly accessible with expiration
        await asyncio.to_thread(blob.make_public)
        
        # Set expiration (7 days)
        blob.custom_time = datetime.now(timezone.utc) + timedelta(days=7)
        await asyncio.to_thread(blob.patch)
        
        self.logger.info(f"Audio file saved to GCS: {filename}")
        return blob.public_url

    async def _save_audio_local(self, audio_data: bytes, filename: str, voice_id: str = None) -> str:
        """Save audio file to local storage"""
        
        # Generate file path
        file_path = self.local_storage_path / self.paths['audio'].rstrip('/') / filename
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        if AIOFILES_AVAILABLE:
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(audio_data)
        else:
            with open(file_path, 'wb') as f:
                f.write(audio_data)
        
        # Create metadata
        metadata = {
            'voice_id': voice_id or str(uuid.uuid4()),
            'created_date': datetime.now(timezone.utc).isoformat(),
            'file_size': str(len(audio_data)),
            'expires_at': (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
        }
        
        metadata_path = file_path.with_suffix(file_path.suffix + '.meta')
        if AIOFILES_AVAILABLE:
            async with aiofiles.open(metadata_path, 'w') as f:
                await f.write(str(metadata))
        else:
            with open(metadata_path, 'w') as f:
                f.write(str(metadata))
        
        # Return file URL (in production, this would be served by a web server)
        local_url = f"file://{file_path.absolute()}"
        
        self.logger.info(f"Audio file saved locally: {local_url}")
        return local_url

    async def save_report_pdf(
        self, 
        pdf_data: bytes, 
        report_id: str,
        filename: str = None
    ) -> str:
        """
        Save generated PDF report to storage
        
        Args:
            pdf_data: PDF data as bytes
            report_id: Report identifier
            filename: Optional custom filename
            
        Returns:
            Download URL for PDF
        """
        
        try:
            self.logger.info(f"Saving PDF report: {report_id}")
            
            if self.use_gcs:
                return await self._save_report_gcs(pdf_data, report_id, filename)
            else:
                return await self._save_report_local(pdf_data, report_id, filename)
                
        except Exception as e:
            self.logger.error(f"Error saving PDF report: {str(e)}")
            raise StorageError(f"Failed to save PDF report: {str(e)}")

    async def _save_report_gcs(self, pdf_data: bytes, report_id: str, filename: str = None) -> str:
        """Save PDF report to GCS"""
        
        # Generate filename if not provided
        if not filename:
            filename = f"legal_analysis_{report_id}.pdf"
        
        # Generate storage path
        blob_name = f"{self.paths['reports']}{filename}"
        
        # Create blob
        blob = self.bucket.blob(blob_name)
        
        # Set metadata
        blob.metadata = {
            'report_id': report_id,
            'created_date': datetime.now(timezone.utc).isoformat(),
            'file_size': str(len(pdf_data))
        }
        
        # Upload PDF
        await asyncio.to_thread(
            blob.upload_from_string,
            pdf_data,
            content_type='application/pdf'
        )
        
        # Generate signed URL for download (valid for 1 hour)
        download_url = await asyncio.to_thread(
            blob.generate_signed_url,
            expiration=datetime.now(timezone.utc) + timedelta(hours=1),
            method='GET'
        )
        
        self.logger.info(f"PDF report saved to GCS: {report_id}")
        return download_url

    async def _save_report_local(self, pdf_data: bytes, report_id: str, filename: str = None) -> str:
        """Save PDF report to local storage"""
        
        # Generate filename if not provided
        if not filename:
            filename = f"legal_analysis_{report_id}.pdf"
        
        # Generate file path
        file_path = self.local_storage_path / self.paths['reports'].rstrip('/') / filename
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        if AIOFILES_AVAILABLE:
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(pdf_data)
        else:
            with open(file_path, 'wb') as f:
                f.write(pdf_data)
        
        # Create metadata
        metadata = {
            'report_id': report_id,
            'created_date': datetime.now(timezone.utc).isoformat(),
            'file_size': str(len(pdf_data)),
            'expires_at': (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        }
        
        metadata_path = file_path.with_suffix('.meta')
        if AIOFILES_AVAILABLE:
            async with aiofiles.open(metadata_path, 'w') as f:
                await f.write(str(metadata))
        else:
            with open(metadata_path, 'w') as f:
                f.write(str(metadata))
        
        # Return download URL
        download_url = f"file://{file_path.absolute()}"
        
        self.logger.info(f"PDF report saved locally: {download_url}")
        return download_url

    async def get_document(self, document_id: str) -> Optional[bytes]:
        """
        Retrieve document from storage
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document content as bytes or None if not found
        """
        
        try:
            if self.use_gcs:
                return await self._get_document_gcs(document_id)
            else:
                return await self._get_document_local(document_id)
                
        except Exception as e:
            self.logger.error(f"Error retrieving document: {str(e)}")
            return None

    async def _get_document_gcs(self, document_id: str) -> Optional[bytes]:
        """Get document from GCS"""
        
        # Find blob by document_id
        blobs = self.client.list_blobs(
            self.bucket, 
            prefix=self.paths['documents']
        )
        
        for blob in blobs:
            if blob.metadata and blob.metadata.get('document_id') == document_id:
                content = await asyncio.to_thread(blob.download_as_bytes)
                return content
        
        return None

    async def _get_document_local(self, document_id: str) -> Optional[bytes]:
        """Get document from local storage"""
        
        # Search for files with matching document_id
        docs_dir = self.local_storage_path / self.paths['documents'].rstrip('/')
        
        if not docs_dir.exists():
            return None
        
        for file_path in docs_dir.iterdir():
            if file_path.suffix == '.meta':
                continue
                
            # Check if this file matches the document_id
            if document_id in file_path.stem:
                if AIOFILES_AVAILABLE:
                    async with aiofiles.open(file_path, 'rb') as f:
                        return await f.read()
                else:
                    with open(file_path, 'rb') as f:
                        return f.read()
        
        return None

    async def delete_document(self, document_id: str) -> bool:
        """
        Delete document from storage
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if deleted successfully, False otherwise
        """
        
        try:
            if self.use_gcs:
                return await self._delete_document_gcs(document_id)
            else:
                return await self._delete_document_local(document_id)
                
        except Exception as e:
            self.logger.error(f"Error deleting document: {str(e)}")
            return False

    async def _delete_document_gcs(self, document_id: str) -> bool:
        """Delete document from GCS"""
        
        # Find and delete blob
        blobs = self.client.list_blobs(
            self.bucket,
            prefix=self.paths['documents']
        )
        
        for blob in blobs:
            if blob.metadata and blob.metadata.get('document_id') == document_id:
                await asyncio.to_thread(blob.delete)
                self.logger.info(f"Document deleted from GCS: {document_id}")
                return True
        
        return False

    async def _delete_document_local(self, document_id: str) -> bool:
        """Delete document from local storage"""
        
        docs_dir = self.local_storage_path / self.paths['documents'].rstrip('/')
        
        if not docs_dir.exists():
            return False
        
        deleted = False
        for file_path in docs_dir.iterdir():
            if document_id in file_path.stem:
                try:
                    file_path.unlink()  # Delete file
                    
                    # Also delete metadata file if exists
                    meta_path = file_path.with_suffix(file_path.suffix + '.meta')
                    if meta_path.exists():
                        meta_path.unlink()
                    
                    deleted = True
                    self.logger.info(f"Document deleted locally: {document_id}")
                except Exception as e:
                    self.logger.warning(f"Error deleting file {file_path}: {e}")
        
        return deleted

    async def cleanup_expired_files(self) -> Dict[str, int]:
        """
        Clean up expired files from storage
        
        Returns:
            Dictionary with cleanup statistics
        """
        
        try:
            self.logger.info("Starting cleanup of expired files")
            
            if self.use_gcs:
                return await self._cleanup_expired_gcs()
            else:
                return await self._cleanup_expired_local()
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            return {'error': str(e)}

    async def _cleanup_expired_gcs(self) -> Dict[str, int]:
        """Cleanup expired files from GCS"""
        
        cleanup_stats = {
            'audio_files': 0,
            'temp_files': 0,
            'reports': 0,
            'total': 0
        }
        
        # Clean up audio files older than 7 days
        audio_blobs = self.client.list_blobs(
            self.bucket,
            prefix=self.paths['audio']
        )
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=7)
        
        for blob in audio_blobs:
            if blob.time_created.replace(tzinfo=timezone.utc) < cutoff_date:
                await asyncio.to_thread(blob.delete)
                cleanup_stats['audio_files'] += 1
        
        # Clean up temp files older than 1 day
        temp_blobs = self.client.list_blobs(
            self.bucket,
            prefix=self.paths['temp']
        )
        
        temp_cutoff = datetime.now(timezone.utc) - timedelta(days=1)
        
        for blob in temp_blobs:
            if blob.time_created.replace(tzinfo=timezone.utc) < temp_cutoff:
                await asyncio.to_thread(blob.delete)
                cleanup_stats['temp_files'] += 1
        
        # Clean up reports older than 30 days
        report_blobs = self.client.list_blobs(
            self.bucket,
            prefix=self.paths['reports']
        )
        
        report_cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        
        for blob in report_blobs:
            if blob.time_created.replace(tzinfo=timezone.utc) < report_cutoff:
                await asyncio.to_thread(blob.delete)
                cleanup_stats['reports'] += 1
        
        cleanup_stats['total'] = sum(cleanup_stats.values()) - cleanup_stats['total']
        
        self.logger.info(f"GCS cleanup completed: {cleanup_stats}")
        return cleanup_stats

    async def _cleanup_expired_local(self) -> Dict[str, int]:
        """Cleanup expired files from local storage"""
        
        cleanup_stats = {
            'audio_files': 0,
            'temp_files': 0,
            'reports': 0,
            'total': 0
        }
        
        current_time = datetime.now(timezone.utc)
        
        # Define cutoff times
        cutoffs = {
            'audio': current_time - timedelta(days=7),
            'temp': current_time - timedelta(days=1),
            'reports': current_time - timedelta(days=30)
        }
        
        # Clean up each category
        for category, cutoff_time in cutoffs.items():
            if category == 'reports':
                category_key = 'reports'
            else:
                category_key = f'{category}_files'
                
            category_dir = self.local_storage_path / self.paths[category].rstrip('/')
            
            if not category_dir.exists():
                continue
            
            for file_path in category_dir.iterdir():
                try:
                    # Check file modification time
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
                    
                    if file_time < cutoff_time:
                        file_path.unlink()
                        cleanup_stats[category_key] += 1
                        
                        # Also delete metadata file if exists
                        meta_path = file_path.with_suffix(file_path.suffix + '.meta')
                        if meta_path.exists():
                            meta_path.unlink()
                            
                except Exception as e:
                    self.logger.warning(f"Error cleaning up file {file_path}: {e}")
        
        cleanup_stats['total'] = sum(v for k, v in cleanup_stats.items() if k != 'total')
        
        self.logger.info(f"Local cleanup completed: {cleanup_stats}")
        return cleanup_stats

    async def get_storage_usage(self) -> Dict[str, Any]:
        """
        Get storage usage statistics
        
        Returns:
            Storage usage information
        """
        
        try:
            if self.use_gcs:
                return await self._get_storage_usage_gcs()
            else:
                return await self._get_storage_usage_local()
                
        except Exception as e:
            self.logger.error(f"Error getting storage usage: {str(e)}")
            return {'error': str(e)}

    async def _get_storage_usage_gcs(self) -> Dict[str, Any]:
        """Get storage usage from GCS"""
        
        stats = {
            'total_files': 0,
            'total_size_bytes': 0,
            'by_category': {
                'documents': {'count': 0, 'size': 0},
                'audio': {'count': 0, 'size': 0},
                'reports': {'count': 0, 'size': 0},
                'temp': {'count': 0, 'size': 0}
            }
        }
        
        # Iterate through all blobs
        blobs = self.client.list_blobs(self.bucket)
        
        for blob in blobs:
            stats['total_files'] += 1
            stats['total_size_bytes'] += blob.size or 0
            
            # Categorize by path
            for category, path in self.paths.items():
                if blob.name.startswith(path):
                    stats['by_category'][category]['count'] += 1
                    stats['by_category'][category]['size'] += blob.size or 0
                    break
        
        # Convert sizes to human readable format
        stats['total_size_mb'] = round(stats['total_size_bytes'] / (1024 * 1024), 2)
        
        for category in stats['by_category']:
            size_bytes = stats['by_category'][category]['size']
            stats['by_category'][category]['size_mb'] = round(size_bytes / (1024 * 1024), 2)
        
        stats['storage_type'] = 'Google Cloud Storage'
        return stats

    async def _get_storage_usage_local(self) -> Dict[str, Any]:
        """Get storage usage from local storage"""
        
        stats = {
            'total_files': 0,
            'total_size_bytes': 0,
            'by_category': {
                'documents': {'count': 0, 'size': 0},
                'audio': {'count': 0, 'size': 0},
                'reports': {'count': 0, 'size': 0},
                'temp': {'count': 0, 'size': 0}
            }
        }
        
        # Iterate through categories
        for category, path in self.paths.items():
            category_dir = self.local_storage_path / path.rstrip('/')
            
            if not category_dir.exists():
                continue
                
            for file_path in category_dir.iterdir():
                if file_path.suffix == '.meta':
                    continue  # Skip metadata files
                    
                try:
                    file_size = file_path.stat().st_size
                    
                    stats['total_files'] += 1
                    stats['total_size_bytes'] += file_size
                    stats['by_category'][category]['count'] += 1
                    stats['by_category'][category]['size'] += file_size
                    
                except Exception as e:
                    self.logger.warning(f"Error getting file size for {file_path}: {e}")
        
        # Convert sizes to human readable format
        stats['total_size_mb'] = round(stats['total_size_bytes'] / (1024 * 1024), 2)
        
        for category in stats['by_category']:
            size_bytes = stats['by_category'][category]['size']
            stats['by_category'][category]['size_mb'] = round(size_bytes / (1024 * 1024), 2)
        
        stats['storage_type'] = 'Local Storage'
        stats['storage_path'] = str(self.local_storage_path.absolute())
        return stats

    async def create_backup(self, backup_location: str = None) -> bool:
        """
        Create backup of important files
        
        Args:
            backup_location: Backup location (bucket name for GCS, path for local)
            
        Returns:
            True if backup successful
        """
        
        try:
            if self.use_gcs:
                return await self._create_backup_gcs(backup_location)
            else:
                return await self._create_backup_local(backup_location)
                
        except Exception as e:
            self.logger.error(f"Error creating backup: {str(e)}")
            return False

    async def _create_backup_gcs(self, backup_bucket: str = None) -> bool:
        """Create backup in GCS"""
        
        if not backup_bucket:
            backup_bucket = f"{self.bucket_name}-backup"
        
        self.logger.info(f"Creating GCS backup to: {backup_bucket}")
        
        # Create backup bucket if not exists
        backup_bucket_obj = self.client.bucket(backup_bucket)
        
        try:
            await asyncio.to_thread(backup_bucket_obj.create)
        except Exception:
            pass  # Bucket might already exist
        
        # Copy important files (documents)
        blobs = self.client.list_blobs(
            self.bucket,
            prefix=self.paths['documents']
        )
        
        backup_count = 0
        for blob in blobs:
            # Copy to backup bucket
            backup_blob = backup_bucket_obj.blob(blob.name)
            await asyncio.to_thread(
                backup_blob.upload_from_string,
                await asyncio.to_thread(blob.download_as_bytes),
                content_type=blob.content_type
            )
            backup_count += 1
        
        self.logger.info(f"GCS backup completed: {backup_count} files backed up")
        return True

    async def _create_backup_local(self, backup_path: str = None) -> bool:
        """Create local backup"""
        
        if not backup_path:
            backup_path = str(self.local_storage_path.parent / f"{self.local_storage_path.name}_backup")
        
        backup_dir = Path(backup_path)
        
        self.logger.info(f"Creating local backup to: {backup_dir}")
        
        try:
            # Copy entire storage directory
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            
            shutil.copytree(self.local_storage_path, backup_dir)
            
            self.logger.info(f"Local backup completed to: {backup_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Local backup failed: {e}")
            return False

    def get_public_url(self, blob_name: str) -> str:
        """Get public URL for a blob (GCS only)"""
        
        if self.use_gcs:
            blob = self.bucket.blob(blob_name)
            return blob.public_url
        else:
            # For local storage, return file path
            return f"file://{self.local_storage_path / blob_name}"

    def get_signed_url(
        self, 
        blob_name: str, 
        expiration_hours: int = 1,
        method: str = 'GET'
    ) -> str:
        """
        Get signed URL for secure access (GCS only)
        
        Args:
            blob_name: Name of the blob
            expiration_hours: URL expiration in hours
            method: HTTP method (GET, POST, etc.)
            
        Returns:
            Signed URL or file path for local storage
        """
        
        if self.use_gcs:
            blob = self.bucket.blob(blob_name)
            
            return blob.generate_signed_url(
                expiration=datetime.now(timezone.utc) + timedelta(hours=expiration_hours),
                method=method
            )
        else:
            # For local storage, return file path with expiration note
            file_path = self.local_storage_path / blob_name
            return f"file://{file_path}?expires_in={expiration_hours}h"

    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage configuration and capabilities"""
        
        return {
            'storage_type': 'Google Cloud Storage' if self.use_gcs else 'Local Storage',
            'bucket_name': getattr(self, 'bucket_name', None),
            'local_path': str(self.local_storage_path) if not self.use_gcs else None,
            'capabilities': {
                'cloud_storage': self.use_gcs,
                'local_storage': not self.use_gcs,
                'async_operations': AIOFILES_AVAILABLE,
                'signed_urls': self.use_gcs,
                'public_urls': self.use_gcs,
                'automatic_expiration': self.use_gcs,
                'backup_support': True,
                'cleanup_support': True
            },
            'storage_paths': self.paths,
            'dependencies': {
                'google_cloud_storage': GCS_AVAILABLE,
                'aiofiles': AIOFILES_AVAILABLE,
                'settings': SETTINGS_AVAILABLE
            }
        }


# Utility functions
async def create_simple_storage_manager() -> StorageManager:
    """Create a simple storage manager for testing"""
    return StorageManager()

def get_file_extension(filename: str) -> str:
    """Get file extension from filename"""
    return Path(filename).suffix.lower()

def is_supported_document_type(filename: str) -> bool:
    """Check if document type is supported"""
    supported_extensions = {'.pdf', '.docx', '.doc', '.txt', '.rtf'}
    return get_file_extension(filename) in supported_extensions

def is_supported_audio_type(filename: str) -> bool:
    """Check if audio type is supported"""
    supported_extensions = {'.mp3', '.wav', '.m4a', '.ogg'}
    return get_file_extension(filename) in supported_extensions

def generate_unique_filename(original_filename: str, prefix: str = None) -> str:
    """Generate unique filename with timestamp and UUID"""
    
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())[:8]
    
    file_path = Path(original_filename)
    stem = file_path.stem
    suffix = file_path.suffix
    
    if prefix:
        return f"{prefix}_{stem}_{timestamp}_{unique_id}{suffix}"
    else:
        return f"{stem}_{timestamp}_{unique_id}{suffix}"

async def cleanup_temp_files(storage_manager: StorageManager) -> int:
    """Cleanup temporary files"""
    try:
        stats = await storage_manager.cleanup_expired_files()
        return stats.get('temp_files', 0)
    except Exception as e:
        logger.error(f"Error cleaning temp files: {e}")
        return 0
