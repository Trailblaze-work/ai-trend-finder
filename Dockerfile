FROM ollama/ollama

# Install Python and httpx
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    pip3 install --no-cache-dir --break-system-packages httpx && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy application
WORKDIR /app
COPY free_trend_aggregator.py .
COPY entrypoint.sh .

# Pull the model at build time so it's baked into the image
RUN ollama serve & \
    sleep 3 && \
    ollama pull qwen2.5:3b && \
    kill %1 && wait || true

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--slack"]
