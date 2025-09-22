#set_gcs_cors.py
import os, sys, json, argparse
from typing import List
from google.cloud import storage
from google.oauth2 import service_account

def _csv(v: str | None) -> List[str]:
    if not v:
        return []
    return [x.strip() for x in v.split(",") if x.strip()]

def set_bucket_cors(
    bucket_name: str,
    origins: list[str],
    methods: list[str],
    headers: list[str],
    max_age: int,
    project: str | None = None,
    credentials_path: str | None = None
) -> None:
    # Prefer ADC if no explicit file path
    if credentials_path:
        creds = service_account.Credentials.from_service_account_file(credentials_path)
        client = storage.Client(project=project, credentials=creds)
    else:
        client = storage.Client(project=project)

    bucket = client.bucket(bucket_name)
    desired = [{
        "origin": origins,
        "method": [m.upper() for m in methods],
        "responseHeader": headers,
        "maxAgeSeconds": int(max_age),
    }]

    current = bucket.cors or []

    def _norm(r: dict) -> dict:
        return {
            "origin": sorted(r.get("origin", [])),
            "method": sorted([m.upper() for m in r.get("method", [])]),
            "responseHeader": sorted(r.get("responseHeader", [])),
            "maxAgeSeconds": int(r.get("maxAgeSeconds", 0)),
        }

    if [_norm(r) for r in current] == [_norm(r) for r in desired]:
        print(f"[OK] CORS already up-to-date on '{bucket_name}'.")
        print(json.dumps(desired, indent=2))
        return

    bucket.cors = desired
    bucket.patch()
    print(f"[DONE] Updated CORS on '{bucket_name}':")
    print(json.dumps(desired, indent=2))

def main() -> int:
    p = argparse.ArgumentParser(description="Set CORS on a GCS bucket")
    p.add_argument("--bucket", default=os.getenv("GCS_BUCKET_NAME"))
    p.add_argument("--origins", default=os.getenv("GCS_CORS_ORIGINS"))
    p.add_argument("--methods", default=os.getenv("GCS_CORS_METHODS", "GET,PUT,POST,HEAD,DELETE,OPTIONS"))
    p.add_argument("--headers", default=os.getenv("GCS_CORS_HEADERS", "Content-Type,Authorization,x-goog-meta-*,x-goog-resumable"))
    p.add_argument("--max-age", type=int, default=int(os.getenv("GCS_CORS_MAX_AGE", "3600")))
    p.add_argument("--project", default=os.getenv("GOOGLE_CLOUD_PROJECT"))
    p.add_argument("--creds", default=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))  # optional
    args = p.parse_args()

    if not args.bucket:
        print("ERROR: GCS bucket not provided (--bucket or GCS_BUCKET_NAME)", file=sys.stderr)
        return 2
    origins = _csv(args.origins) or [
        "http://localhost:5174",
        "http://localhost:3000",
        "http://localhost:8080",
        "https://nimble-kringle-55eacc.netlify.app",
        "https://legal-ai-xi.vercel.app",
    ]

    try:
        set_bucket_cors(
            bucket_name=args.bucket,
            origins=origins,
            methods=_csv(args.methods),
            headers=_csv(args.headers),
            max_age=args.max_age,
            project=args.project,
            credentials_path=args.creds
        )
        return 0
    except Exception as e:
        print(f"ERROR: failed to set CORS: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
