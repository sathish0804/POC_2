FROM python:3.11-slim

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies for OpenCV, EasyOCR, Tesseract, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first for better layer caching
COPY requirements.txt ./

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Pre-install CPU PyTorch wheels to avoid source builds
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

# Install a Linux-compatible mediapipe version (0.10.18 has wheels)
RUN pip install --no-cache-dir mediapipe==0.10.18

# Install remaining requirements, excluding heavy/unused items that cause build issues
# (torch/torchvision already provided above; tensorflow/jax/keras not used by this app)
RUN set -eux; \
    grep -viE '^(torch|torchvision|tensorflow|jax|jaxlib|keras|mediapipe)(==.*)?$' requirements.txt > requirements.runtime.txt; \
    pip install --no-cache-dir -r requirements.runtime.txt

# Copy the rest of the project
COPY . .

# Serve with Gunicorn
EXPOSE 8000

CMD ["gunicorn", "-c", "gunicorn.conf.py", "wsgi:app"]


