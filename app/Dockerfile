# Use a slim, official Python base image
FROM --platform=linux/amd64 python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install system dependencies needed for libraries like PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --- Model Caching ---
# Download the sentence-transformer model during the build process.
# This ensures the model is included in the image for offline runtime use.
# The model will be cached in the standard Hugging Face cache directory inside the image.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of the application files
COPY process_documents.py .
COPY challenge1b_output_schema.json .

# Define the command to run the application
# It expects inputs to be mounted at /app/input and outputs to be written to /app/output
CMD ["python", "process_documents.py"]