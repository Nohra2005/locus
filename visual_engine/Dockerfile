FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for image processing
# Switched 'libgl1-mesa-glx' to 'libgl1' for newer Debian versions
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Start the Visual Engine on port 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]