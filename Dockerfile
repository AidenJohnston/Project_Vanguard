FROM python:3.11-slim

# Install OS‑level deps if you need any (none needed here)
WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app.py .

# Let Render inject $PORT
ENV PORT=5000

# Use gevent worker to support streaming
CMD ["gunicorn", "app:app", "-k", "gevent", "-b", "0.0.0.0:5000", "--timeout", "120"]
