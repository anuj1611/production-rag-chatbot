#!/bin/bash

set -e  

echo "🚀 Starting Docker containers..."
docker compose up -d

echo "⏳ Waiting for MySQL to be ready..."
until docker exec chatbot-mysql mysqladmin ping -h localhost -u myuser -pmypassword --silent; do
	sleep 2
done

echo "📦 Applying schema to MySQL..."
docker exec -i chatbot-mysql mysql -u myuser -pmypassword mydb < schema.sql

echo "📊 Running data ingestion script..."
python ingest_new_data.py

echo "Running workers"
python -m arq worker.WorkerSettings &
WORKER_PID=$!
trap "kill $WORKER_PID" EXIT

echo "🔥 Starting FastAPI server with 5 workers..."
python -m uvicorn main:app --host 0.0.0.0 --port 8080 --workers 5
