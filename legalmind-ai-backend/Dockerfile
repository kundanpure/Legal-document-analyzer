FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p static logs uploads

EXPOSE 8000

# Use the standalone main file
CMD ["python", "-m", "uvicorn", "standalone_main:app", "--host", "0.0.0.0", "--port", "8000"]
