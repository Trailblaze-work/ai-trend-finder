#!/bin/sh
set -e

# Start Ollama in the background
ollama serve &

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
until ollama list >/dev/null 2>&1; do
  sleep 0.5
done
echo "Ollama is ready."

# Run the trend aggregator with any arguments passed to the container
exec python3 /app/free_trend_aggregator.py "$@"
