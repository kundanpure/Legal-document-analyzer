#!/bin/bash

echo "🐳 LegalMind AI Docker Testing Script"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker is running${NC}"

# Check if required files exist
required_files=("Dockerfile" "docker-compose.yml" "requirements.txt" ".env")
for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo -e "${RED}❌ Missing required file: $file${NC}"
        exit 1
    fi
done

echo -e "${GREEN}✅ All required files present${NC}"

# Build and start containers
echo -e "${YELLOW}🔨 Building Docker containers...${NC}"
docker-compose build

echo -e "${YELLOW}🚀 Starting containers...${NC}"
docker-compose up -d

# Wait for services to be ready
echo -e "${YELLOW}⏳ Waiting for services to start...${NC}"
sleep 30

# Check if containers are running
if docker-compose ps | grep -q "Up"; then
    echo -e "${GREEN}✅ Containers are running${NC}"
else
    echo -e "${RED}❌ Containers failed to start${NC}"
    docker-compose logs
    exit 1
fi

# Test health endpoint
echo -e "${YELLOW}🔍 Testing health endpoint...${NC}"
health_response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)

if [[ "$health_response" == "200" ]]; then
    echo -e "${GREEN}✅ Health check passed${NC}"
else
    echo -e "${RED}❌ Health check failed (Status: $health_response)${NC}"
    docker-compose logs legalmind-api
fi

# Run comprehensive API tests
echo -e "${YELLOW}🧪 Running comprehensive API tests...${NC}"
if [[ -f "test_endpoints.py" ]]; then
    python3 test_endpoints.py
else
    echo -e "${YELLOW}⚠️ test_endpoints.py not found, running basic tests...${NC}"
    
    # Basic endpoint tests
    endpoints=(
        "/"
        "/health"
        "/system/stats"
        "/ping"
    )
    
    for endpoint in "${endpoints[@]}"; do
        response=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:8000$endpoint")
        if [[ "$response" == "200" ]]; then
            echo -e "${GREEN}✅ $endpoint - Status: $response${NC}"
        else
            echo -e "${RED}❌ $endpoint - Status: $response${NC}"
        fi
    done
fi

# Show container logs
echo -e "${YELLOW}📋 Recent container logs:${NC}"
docker-compose logs --tail=10 legalmind-api

# Show running containers
echo -e "${YELLOW}📊 Container status:${NC}"
docker-compose ps

echo -e "${GREEN}🎉 Docker testing completed!${NC}"
echo -e "${YELLOW}💡 Access your API at: http://localhost:8000${NC}"
echo -e "${YELLOW}📚 API Documentation: http://localhost:8000/docs${NC}"
echo -e "${YELLOW}🔍 ReDoc Documentation: http://localhost:8000/redoc${NC}"

# Keep containers running
echo -e "${YELLOW}⏳ Containers will keep running. Use 'docker-compose down' to stop.${NC}"
