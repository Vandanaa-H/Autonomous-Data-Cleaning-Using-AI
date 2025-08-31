"""
Cloud Storage Integration for AI Data Cleaning System
Supports Google Cloud Storage and AWS S3
"""

import os
import logging
from typing import Optional, Union
from pathlib import Path
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)

# Cloud storage configuration
CLOUD_STORAGE_ENABLED = os.getenv("CLOUD_STORAGE_ENABLED", "false").lower() == "true"
CLOUD_PROVIDER = os.getenv("CLOUD_PROVIDER", "gcs")  # gcs or s3
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "dataclean-storage")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "dataclean-storage")

# Try to import cloud storage libraries
try:
    if CLOUD_PROVIDER == "gcs":
        from google.cloud import storage as gcs
        GCS_AVAILABLE = True
    else:
        GCS_AVAILABLE = False
except ImportError:
    GCS_AVAILABLE = False
    logger.warning("Google Cloud Storage client not available")

try:
    if CLOUD_PROVIDER == "s3":
        import boto3
        from botocore.exceptions import ClientError
        S3_AVAILABLE = True
    else:
        S3_AVAILABLE = False
except ImportError:
    S3_AVAILABLE = False
    logger.warning("AWS S3 client not available")

class CloudStorageManager:
    """Unified cloud storage manager"""
    
    def __init__(self):
        self.provider = CLOUD_PROVIDER
        self.enabled = CLOUD_STORAGE_ENABLED
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize cloud storage client"""
        if not self.enabled:
            logger.info("Cloud storage disabled")
            return
            
        try:
            if self.provider == "gcs" and GCS_AVAILABLE:
                self.client = gcs.Client()
                self.bucket_name = GCS_BUCKET_NAME
                logger.info("Google Cloud Storage client initialized")
            elif self.provider == "s3" and S3_AVAILABLE:
                self.client = boto3.client('s3')
                self.bucket_name = S3_BUCKET_NAME
                logger.info("AWS S3 client initialized")
            else:
                logger.warning(f"Cloud provider {self.provider} not available")
                self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize cloud storage client: {e}")
            self.enabled = False
    
    def upload_file(self, local_path: str, cloud_path: str, 
                   content_type: str = None) -> Optional[str]:
        """
        Upload file to cloud storage
        Returns cloud URL on success, None on failure
        """
        if not self.enabled or not self.client:
            logger.info("Cloud storage not available, skipping upload")
            return None
            
        try:
            if self.provider == "gcs":
                return self._upload_to_gcs(local_path, cloud_path, content_type)
            elif self.provider == "s3":
                return self._upload_to_s3(local_path, cloud_path, content_type)
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to cloud storage: {e}")
            return None
    
    def _upload_to_gcs(self, local_path: str, cloud_path: str, 
                      content_type: str = None) -> str:
        """Upload file to Google Cloud Storage"""
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(cloud_path)
        
        if content_type:
            blob.content_type = content_type
            
        blob.upload_from_filename(local_path)
        
        # Generate signed URL for access
        url = blob.generate_signed_url(
            expiration=datetime.utcnow().replace(year=datetime.utcnow().year + 1)
        )
        
        logger.info(f"File uploaded to GCS: {cloud_path}")
        return url
    
    def _upload_to_s3(self, local_path: str, cloud_path: str, 
                     content_type: str = None) -> str:
        """Upload file to AWS S3"""
        extra_args = {}
        if content_type:
            extra_args['ContentType'] = content_type
            
        self.client.upload_file(local_path, self.bucket_name, cloud_path, 
                               ExtraArgs=extra_args)
        
        # Generate presigned URL
        url = self.client.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.bucket_name, 'Key': cloud_path},
            ExpiresIn=3600 * 24 * 365  # 1 year
        )
        
        logger.info(f"File uploaded to S3: {cloud_path}")
        return url
    
    def download_file(self, cloud_path: str, local_path: str) -> bool:
        """Download file from cloud storage"""
        if not self.enabled or not self.client:
            return False
            
        try:
            if self.provider == "gcs":
                return self._download_from_gcs(cloud_path, local_path)
            elif self.provider == "s3":
                return self._download_from_s3(cloud_path, local_path)
        except Exception as e:
            logger.error(f"Failed to download {cloud_path} from cloud storage: {e}")
            return False
    
    def _download_from_gcs(self, cloud_path: str, local_path: str) -> bool:
        """Download file from Google Cloud Storage"""
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(cloud_path)
        blob.download_to_filename(local_path)
        logger.info(f"File downloaded from GCS: {cloud_path}")
        return True
    
    def _download_from_s3(self, cloud_path: str, local_path: str) -> bool:
        """Download file from AWS S3"""
        self.client.download_file(self.bucket_name, cloud_path, local_path)
        logger.info(f"File downloaded from S3: {cloud_path}")
        return True
    
    def delete_file(self, cloud_path: str) -> bool:
        """Delete file from cloud storage"""
        if not self.enabled or not self.client:
            return False
            
        try:
            if self.provider == "gcs":
                bucket = self.client.bucket(self.bucket_name)
                blob = bucket.blob(cloud_path)
                blob.delete()
            elif self.provider == "s3":
                self.client.delete_object(Bucket=self.bucket_name, Key=cloud_path)
            
            logger.info(f"File deleted from cloud storage: {cloud_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {cloud_path} from cloud storage: {e}")
            return False
    
    def list_files(self, prefix: str = "") -> list:
        """List files in cloud storage"""
        if not self.enabled or not self.client:
            return []
            
        try:
            if self.provider == "gcs":
                bucket = self.client.bucket(self.bucket_name)
                blobs = bucket.list_blobs(prefix=prefix)
                return [blob.name for blob in blobs]
            elif self.provider == "s3":
                response = self.client.list_objects_v2(
                    Bucket=self.bucket_name, 
                    Prefix=prefix
                )
                return [obj['Key'] for obj in response.get('Contents', [])]
        except Exception as e:
            logger.error(f"Failed to list files from cloud storage: {e}")
            return []

class FileStorageManager:
    """Unified file storage manager (local + cloud)"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.cloud_storage = CloudStorageManager()
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def save_file(self, file_data: bytes, filename: str, 
                  subfolder: str = "", sync_to_cloud: bool = True) -> dict:
        """
        Save file locally and optionally to cloud storage
        Returns file info dict
        """
        # Create subfolder path
        if subfolder:
            folder_path = self.base_path / subfolder
            folder_path.mkdir(exist_ok=True)
        else:
            folder_path = self.base_path
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{timestamp}{ext}"
        
        # Save locally
        local_path = folder_path / unique_filename
        with open(local_path, "wb") as f:
            f.write(file_data)
        
        # Calculate file hash
        file_hash = self.calculate_file_hash(str(local_path))
        
        # Prepare file info
        file_info = {
            "local_path": str(local_path),
            "filename": unique_filename,
            "original_filename": filename,
            "size": len(file_data),
            "hash": file_hash,
            "created_at": datetime.utcnow().isoformat(),
            "cloud_url": None
        }
        
        # Upload to cloud if enabled
        if sync_to_cloud and self.cloud_storage.enabled:
            cloud_path = f"{subfolder}/{unique_filename}" if subfolder else unique_filename
            cloud_url = self.cloud_storage.upload_file(
                str(local_path), 
                cloud_path,
                self._get_content_type(filename)
            )
            file_info["cloud_url"] = cloud_url
        
        logger.info(f"File saved: {filename} -> {unique_filename}")
        return file_info
    
    def get_file(self, file_path: str, try_cloud: bool = True) -> Optional[bytes]:
        """Get file content from local storage or cloud"""
        local_path = Path(file_path)
        
        # Try local first
        if local_path.exists():
            with open(local_path, "rb") as f:
                return f.read()
        
        # Try cloud if local not found
        if try_cloud and self.cloud_storage.enabled:
            try:
                # Download from cloud to temp location
                temp_path = self.base_path / "temp" / local_path.name
                temp_path.parent.mkdir(exist_ok=True)
                
                cloud_path = str(local_path.relative_to(self.base_path))
                if self.cloud_storage.download_file(cloud_path, str(temp_path)):
                    with open(temp_path, "rb") as f:
                        data = f.read()
                    temp_path.unlink()  # Clean up temp file
                    return data
            except Exception as e:
                logger.error(f"Failed to retrieve file from cloud: {e}")
        
        return None
    
    def delete_file(self, file_path: str, delete_from_cloud: bool = True) -> bool:
        """Delete file from local and optionally cloud storage"""
        local_path = Path(file_path)
        success = True
        
        # Delete local file
        if local_path.exists():
            try:
                local_path.unlink()
                logger.info(f"Local file deleted: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete local file: {e}")
                success = False
        
        # Delete from cloud
        if delete_from_cloud and self.cloud_storage.enabled:
            try:
                cloud_path = str(local_path.relative_to(self.base_path))
                cloud_success = self.cloud_storage.delete_file(cloud_path)
                success = success and cloud_success
            except Exception as e:
                logger.error(f"Failed to delete file from cloud: {e}")
                success = False
        
        return success
    
    def _get_content_type(self, filename: str) -> str:
        """Get content type based on file extension"""
        ext = os.path.splitext(filename)[1].lower()
        content_types = {
            '.csv': 'text/csv',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.xls': 'application/vnd.ms-excel',
            '.json': 'application/json',
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.txt': 'text/plain'
        }
        return content_types.get(ext, 'application/octet-stream')

# Global storage manager instance
storage_manager = FileStorageManager("data")
