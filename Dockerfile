# Nu1lm Microplastics Expert
# Docker image for running Nu1lm anywhere

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt .
COPY setup.py .
COPY README.md .
COPY nu1lm_microplastics.py .
COPY analyzers/ analyzers/
COPY knowledge/ knowledge/
COPY scripts/ scripts/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for models and data
RUN mkdir -p models data training_data

# Download the base model (optional - can be done at runtime)
# RUN python scripts/download_model.py --model nu1lm-nano

# Expose port for API mode (if added later)
EXPOSE 8000

# Default command - interactive mode
CMD ["python", "nu1lm_microplastics.py"]
