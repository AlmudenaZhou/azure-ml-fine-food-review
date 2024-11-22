FROM python:3.11-slim

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# COPY src/training_pipeline.py training_pipeline.py
# COPY src/ src/
