# Start with a base image with Python
FROM python:3.10-slim

# Set environment variables
ENV POETRY_VERSION=1.4.0 \
    POETRY_VIRTUALENVS_CREATE=false  
    # Installs in global site-packages

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Set the working directory
WORKDIR /app

# Copy only the dependency files first for caching
COPY pyproject.toml poetry.lock* ./

# Install dependencies only (no code yet for cache efficiency)
RUN poetry install --no-root --no-interaction --no-ansi

# Copy the rest of the application code
COPY . .