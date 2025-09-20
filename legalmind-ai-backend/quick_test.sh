#!/bin/bash

echo "��� LEGALMIND AI - QUICK SERVICE TESTING"
echo "======================================"

BASE="http://localhost:8000"

echo ""
echo "1️⃣ Health Check:"
curl -s $BASE/health

echo ""
echo ""
echo "2️⃣ Get Signed URL:"
curl -s -X POST $BASE/api/uploads/get-signed-url \
  -H "Content-Type: application/json" \
  -d '{"filename":"test.pdf","content_type":"application/pdf","file_size":1000000}'

echo ""
echo ""
echo "3️⃣ Upload Notification:"
curl -s -X POST $BASE/api/uploads/notify-uploaded \
  -H "Content-Type: application/json" \
  -d '{"file_id":"file_test123","gcs_path":"uploads/test.pdf","original_filename":"test.pdf","file_size":1000000,"content_type":"application/pdf"}'

echo ""
echo ""
echo "4️⃣ AI Chat:"
curl -s -X POST $BASE/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Can you help analyze contracts?","file_id":"file_test123"}'

echo ""
echo ""
echo "5️⃣ Generate Summary:"
curl -s -X POST $BASE/api/insights/file_test123/summary

echo ""
echo ""
echo "6️⃣ Generate Voice:"
curl -s -X POST $BASE/api/insights/file_test123/audio \
  -H "Content-Type: application/json" \
  -d '{"options":{"voice_type":"female","language":"en"}}'

echo ""
echo ""
echo "��� ALL TESTS COMPLETE!"
echo "Visit http://localhost:8000/docs for interactive testing"
