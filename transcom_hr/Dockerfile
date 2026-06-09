# Use a lightweight official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for scientific libraries (scikit-learn, numpy, faiss)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libatlas-base-dev \
    python3-dev \
    gfortran \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the project code into the container
COPY . /app/

# Expose port 8000
EXPOSE 8000

# Run migrations and start development server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
