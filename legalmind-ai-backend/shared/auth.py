"""
Authentication utilities for Google services
"""
import os
import json
from google.oauth2 import service_account
from google.auth import default


def get_google_credentials():
    """
    Get Google Cloud credentials and project ID
    """
    try:
        # Try service account key file first
        service_account_key = os.getenv('GOOGLE_SERVICE_ACCOUNT_KEY')
        if service_account_key:
            if os.path.isfile(service_account_key):
                credentials = service_account.Credentials.from_service_account_file(service_account_key)
            else:
                # Try parsing as JSON string
                key_data = json.loads(service_account_key)
                credentials = service_account.Credentials.from_service_account_info(key_data)
            project_id = credentials.project_id
        else:
            # Use default credentials (for GCP environments)
            credentials, project_id = default()
            
        return credentials, project_id
        
    except Exception as e:
        print(f"Warning: Could not load Google credentials: {e}")
        return None, os.getenv('GOOGLE_CLOUD_PROJECT', 'your-project-id')


def get_gemini_api_key():
    """Get Gemini API key"""
    return os.getenv('GOOGLE_API_KEY', 'your-gemini-api-key')
