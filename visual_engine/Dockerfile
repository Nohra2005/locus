FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for image processing
# Switched 'libgl1-mesa-glx' to 'libgl1' for newer Debian versions
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install torch first (large ~200MB download — cached in its own layer)
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.6.0+cpu \
    torchvision==0.21.0+cpu

# Install ultralytics separately (pulls scipy ~38MB, cache this layer)
RUN pip install --no-cache-dir ultralytics

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Start the Visual Engine on port 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]