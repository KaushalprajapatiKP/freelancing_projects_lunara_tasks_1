FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
    "pip==24.0" \
    "setuptools==69.0.2" \
    "wheel==0.42.0" \
    "numpy==1.26.4" \
    "pandas==2.2.0" \
    "scikit-learn==1.3.2" \
    "lightgbm==4.1.0" \
    "joblib==1.3.2" \
    "xgboost==2.1.4"

WORKDIR /app

COPY ./data /app/data

