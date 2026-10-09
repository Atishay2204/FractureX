# Use the official lightweight Python image.
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code and model weights
COPY api.py .
COPY best_binary_fracture.pt .
COPY best_verified_light.pt .

# Expose port 8000
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
