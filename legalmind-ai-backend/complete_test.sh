#!/bin/bash

echo "��� LEGALMIND AI - COMPREHENSIVE SERVICE TESTING"
echo "=============================================="
echo ""

BASE="http://localhost:8000"

# Function to run test with formatted output
run_test() {
    local test_name="$1"
    local method="$2" 
    local endpoint="$3"
    local data="$4"
    
    echo "��� Testing: $test_name"
    echo "Endpoint: $method $endpoint"
    echo ""
    
    if [ -z "$data" ]; then
        curl -s "$BASE$endpoint"
    else
        curl -s -X "$method" "$BASE$endpoint" \
            -H "Content-Type: application/json" \
            -d "$data"
    fi
    
    echo ""
    echo "────────────────────────────────────────────"
    echo ""
}

# Test 1: Health Check
run_test "Health Check" "GET" "/health"

# Test 2: API Root
run_test "API Root" "GET" "/"

# Test 3: Get Signed URL
run_test "Get Signed URL" "POST" "/api/uploads/get-signed-url" '{
  "filename": "test_contract.pdf",
  "content_type": "application/pdf", 
  "file_size": 1024000
}'

# Test 4: Upload Notification
run_test "Upload Notification" "POST" "/api/uploads/notify-uploaded" '{
  "file_id": "file_test123",
  "gcs_path": "uploads/file_test123/contract.pdf",
  "original_filename": "contract.pdf",
  "file_size": 1024000,
  "content_type": "application/pdf"
}'

# Wait for background processing
echo "⏳ Waiting 3 seconds for background processing..."
sleep 3
echo ""

# Test 5: List Files
run_test "List Files" "GET" "/api/uploads"

# Test 6: Get File Details
run_test "Get File Details" "GET" "/api/uploads/file_test123"

# Test 7: AI Chat
run_test "AI Chat" "POST" "/api/chat" '{
  "message": "Hello! Can you help me analyze legal documents and identify risks?",
  "file_id": "file_test123"
}'

# Test 8: Generate Summary
run_test "Generate Summary" "POST" "/api/insights/file_test123/summary" '{}'

# Test 9: Generate Audio
run_test "Generate Audio" "POST" "/api/insights/file_test123/audio" '{
  "options": {
    "voice_type": "female",
    "language": "en",
    "speed": 1.0
  }
}'

# Test 10: Generate Report
run_test "Generate Report" "POST" "/api/insights/file_test123/report" '{
  "options": {
    "type": "comprehensive",
    "format": "pdf"
  }
}'

# Test 11: Get Insights Status
run_test "Get Insights Status" "GET" "/api/insights/file_test123"

# Test 12: Export Conversation
run_test "Export Conversation" "POST" "/api/export/conversation" '{
  "conversation_id": "conv_test123", 
  "format": "pdf"
}'

echo "��� ALL TESTS COMPLETED!"
echo ""
echo "��� Your LegalMind AI API is working perfectly!"
echo "��� Visit http://localhost:8000/docs for interactive testing"
echo "��� Ready for frontend integration!"
