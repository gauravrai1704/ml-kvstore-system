FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model at build time
RUN python -c "from transformers import DistilBertTokenizer, DistilBertForSequenceClassification; \
    DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english'); \
    DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')"

# Copy application
COPY ml_inference.py .

EXPOSE 8000

CMD ["python", "ml_inference.py"]