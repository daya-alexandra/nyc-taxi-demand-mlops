FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt setup.py ./
COPY src src

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY models models

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]