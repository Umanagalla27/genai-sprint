# Use a lightweight Python base image
FROM python:3.11-slim

# Set environment variables: disable buffering and bytecode generation
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies if required, then install Python requirements
COPY requirements.docker.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.docker.txt

# Copy source application
COPY src/ ./src/

# Expose FastAPI default port
EXPOSE 8000

# Run uvicorn server bound to 0.0.0.0 for container networking
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
