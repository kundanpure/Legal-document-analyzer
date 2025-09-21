# set_gcs_cors.py
import os, json
from google.cloud import storage
from google.oauth2 import service_account

BUCKET = os.getenv("GCS_BUCKET_NAME") or "docanalyzer-470219-storage"
ORIGINS = [
  "http://localhost:5174",
  "http://localhost:3000",
  "http://localhost:8080",
  "https://nimble-kringle-55eacc.netlify.app",
  "https://legal-ai-xi.vercel.app",
]

cors = [{
  "origin": ORIGINS,
  "method": ["GET","PUT","POST","HEAD","DELETE","OPTIONS"],
  "responseHeader": ["Content-Type","Authorization","x-goog-meta-*","x-goog-resumable"],
  "maxAgeSeconds": 3600
}]


creds = service_account.Credentials.from_service_account_file(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
client = storage.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"), credentials=creds)
bucket = client.bucket(BUCKET)
bucket.cors = cors
bucket.patch()
print("CORS set:", json.dumps(cors, indent=2))
