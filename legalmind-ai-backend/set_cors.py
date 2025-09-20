# set_cors.py
import os, json, sys, argparse
from pathlib import Path

# Try to load .env if present (works even outside the app)
try:
    from dotenv import load_dotenv
    here = Path(__file__).parent
    # look in current dir, then parent (common layouts)
    for p in [here / ".env", here.parent / ".env"]:
        if p.exists():
            load_dotenv(p)
            break
    else:
        load_dotenv()  # fallback
except Exception:
    pass

from google.cloud import storage
from google.oauth2 import service_account

DEFAULT_CORS = [
    {
        "origin": ["http://localhost:5173", "http://localhost:3000", "http://localhost:8080"],
        "method": ["PUT", "GET", "HEAD", "OPTIONS"],
        "responseHeader": ["Content-Type", "Content-Length", "x-goog-resumable", "x-goog-meta-*"],
        "maxAgeSeconds": 3600,
    }
]

def main():
    ap = argparse.ArgumentParser(description="Set GCS CORS on a bucket")
    ap.add_argument("--bucket", default=os.environ.get("GCS_BUCKET_NAME", "docanalyzer-470219-storage"))
    ap.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT", "docanalyzer-470219"))
    ap.add_argument("--key", help="Path to service account JSON. Falls back to env var GOOGLE_APPLICATION_CREDENTIALS.")
    ap.add_argument("--cors", help="Path to a CORS JSON file (optional). If not given, uses DEFAULT_CORS.")
    args = ap.parse_args()

    key_path = args.key or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not key_path:
        sys.exit("GOOGLE_APPLICATION_CREDENTIALS not set and --key not provided.")
    key_path = str(Path(key_path))  # normalize

    if not Path(key_path).exists():
        sys.exit(f"Service account key not found at: {key_path}")

    cors_rules = DEFAULT_CORS
    if args.cors:
        cors_file = Path(args.cors)
        if not cors_file.exists():
            sys.exit(f"CORS file not found: {cors_file}")
        cors_rules = json.loads(cors_file.read_text(encoding="utf-8"))

    print(f"Using service account key: {key_path}")
    print(f"Target project: {args.project}")
    print(f"Bucket: {args.bucket}")

    creds = service_account.Credentials.from_service_account_file(key_path)
    client = storage.Client(project=args.project, credentials=creds)
    bucket = client.bucket(args.bucket)

    print("Current CORS:", json.dumps(bucket.cors or [], indent=2))
    bucket.cors = cors_rules
    bucket.patch()
    print("Updated CORS:", json.dumps(bucket.cors or [], indent=2))
    print("✅ CORS saved.")

if __name__ == "__main__":
    main()
