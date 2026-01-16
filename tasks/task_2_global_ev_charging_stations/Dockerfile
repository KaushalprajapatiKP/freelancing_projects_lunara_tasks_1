FROM apex_arena:base

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    build-essential \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.1.4 \
    scikit-learn==1.3.2 \
    scipy==1.11.4 \
    xgboost==2.0.3 \
    lightgbm==4.1.0 \
    imbalanced-learn==0.11.0 \
    joblib==1.3.2

COPY ./data /workdir/data
COPY ./tests/ /tests/
COPY ./solution.sh /tests/solution.sh
COPY ./grader.py /tests/grader.py

RUN chown -R model:model /workdir/data && chmod -R 700 /workdir/data

