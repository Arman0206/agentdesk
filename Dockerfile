FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# non-root user (Hugging Face Spaces and most hosts require it)
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# Demo defaults: offline mock model, so a public URL can never burn an API quota.
ENV LLM_PROVIDER=mock EMBEDDING_PROVIDER=hash PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
