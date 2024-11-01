FROM python:3.11.9-slim

# Set environment variables
ENV POETRY_VERSION=1.4.0 \
    POETRY_VIRTUALENVS_CREATE=false
    # Installs in global site-packages

# Install Git and Poetry
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/* && \
    pip install "poetry==$POETRY_VERSION"

# Set the working directory
WORKDIR /app

# Copy only the dependency files first for caching
COPY pyproject.toml poetry.lock* ./

# Install dependencies only (no code yet for cache efficiency)
RUN poetry install --no-root --no-interaction --no-ansi

# Copy the rest of the application code
COPY . .
