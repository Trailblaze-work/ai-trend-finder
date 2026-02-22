# AI Trend Finder

Finds trending AI-at-work topics from free sources (HackerNews, TechMeme, ArXiv, ProductHunt, GitHub Trending) and summarizes them via a local LLM.

## Quick Start (local)

Requires Python 3 and [Ollama](https://ollama.com) running with the `qwen2.5:3b` model.

```bash
ollama pull qwen2.5:3b
pip install httpx
python3 free_trend_aggregator.py --top 10
```

## Docker

Build once (downloads ~2GB model, takes a few minutes):

```bash
docker build -t ai-trend-finder .
```

Run:

```bash
# Terminal digest
docker run --rm ai-trend-finder --top 5

# With GPU acceleration (much faster)
docker run --rm --gpus all ai-trend-finder --top 10
```

> **Note:** CPU-only inference takes 3-4 minutes. With `--gpus all` it's seconds. The GPU flag requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Slack Posting

### Setup (one-time)

1. Go to https://api.slack.com/apps → **Create New App** → **From scratch**
2. Name it (e.g. `AI Trend Digest`), select your workspace
3. Sidebar → **Incoming Webhooks** → toggle **On**
4. Click **Add New Webhook to Workspace** → pick a channel → **Allow**
5. Copy the webhook URL

Save it locally:

```bash
cp .env.example .env
# Edit .env and paste your webhook URL
```

### Usage

```bash
# Local
python3 free_trend_aggregator.py --slack

# Docker
docker run --rm -e SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..." ai-trend-finder --slack
```

Running `--slack` twice on the same day skips the second post. Use `--fresh` to force a re-post.

## All Flags

| Flag | Description |
|------|-------------|
| `--top N` | Summarize top N items via LLM (default: 10, 0 to disable) |
| `--slack` | Post digest to Slack instead of terminal |
| `--json` | Output structured JSON |
| `--min-score N` | Minimum relevance score (default: 5.0) |
| `--fresh` | Bypass cache and re-fetch/re-post |
