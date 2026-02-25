FROM python:3.13-slim

# Install Git and required libraries
RUN apt-get update && apt-get install --no-install-recommends -y \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade --no-cache-dir pip

# Clone the GitHub repository
ARG GITHUB_TOKEN
RUN git clone -b main --single-branch https://${GITHUB_TOKEN}@github.com/hasanakblut/tez-repo.git /home/codes

# Set Python path
ENV PYTHONPATH="/home/codes"

# Set working directory
WORKDIR /home/codes

# Install Python dependencies (torch is pulled from PyTorch CUDA wheels via --find-links in requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge
