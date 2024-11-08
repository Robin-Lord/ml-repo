FROM python:3.11.9-slim

# Set environment variables
ENV POETRY_VERSION=1.4.0 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONPATH="/app/src:${PYTHONPATH}"
    # Add to path so jupyter can import libraries
    # Installs in global site-packages


    # Install Git and Poetry
    RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/* && \
    pip install "poetry==$POETRY_VERSION"

    # Set the working directory
    WORKDIR /app

    # Copy dependency files and install
    COPY pyproject.toml poetry.lock* ./
    RUN poetry install --no-root

    # Keep container running
    CMD ["tail", "-f", "/dev/null"]
