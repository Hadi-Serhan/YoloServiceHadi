# Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*


# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r torch-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Default command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
