#!/usr/bin/env python3
"""
Free AI-at-Work Trend Aggregator
=================================
Aggregates trending AI/work topics from free, no-auth-required sources:
  - HackerNews (Algolia API)
  - TechMeme (RSS)
  - ArXiv (API)
  - Product Hunt (Atom feed)
  - GitHub Trending (HTML scrape)

Usage:
    python3 free_trend_aggregator.py            # markdown output
    python3 free_trend_aggregator.py --json      # structured JSON output
    python3 free_trend_aggregator.py --top 15    # summarize top 15 items via local LLM
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html import unescape
from urllib.parse import quote_plus

from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HN_QUERIES = [
    "AI tools",
    "LLM",
    "GPT",
    "Claude",
    "AI productivity",
    "AI coding",
    "AI marketing",
    "AI sales",
    "AI workflow",
    "AI agents",
]

AI_KEYWORDS = [
    "ai", "artificial intelligence", "machine learning", "ml", "llm",
    "gpt", "openai", "claude", "anthropic", "gemini", "copilot",
    "chatbot", "generative", "deep learning", "neural", "transformer",
    "diffusion", "language model", "rag", "retrieval augmented",
    "fine-tun", "prompt", "agent", "agentic", "automat",
    "workflow", "productiv", "coding assistant", "ai tool",
    "mcp", "model context protocol",
]

WORK_KEYWORDS = [
    "work", "enterprise", "business", "productiv", "workflow",
    "tool", "saas", "startup", "team", "collaborat", "automat",
    "deploy", "integrat", "pipeline", "coding", "develop",
    "marketing", "sales", "customer", "support", "hiring",
    "recruit", "hr", "office", "meeting", "email",
    "document", "spreadsheet", "report", "analytic", "dashboard",
]

HEADERS = {
    "User-Agent": "FreeTrendAggregator/1.0 (research; +https://github.com)",
    "Accept": "application/json, application/xml, text/html, */*",
}

REQUEST_TIMEOUT = 20.0
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:3b"
HISTORY_FILE = Path(__file__).parent / ".trend-history.json"
_ENV_FILE = Path(__file__).parent / ".env"


def _load_env() -> None:
    """Read .env file (if present) into os.environ. No dependencies needed."""
    if not _ENV_FILE.exists():
        return
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:          # real env vars take precedence
            os.environ[key] = value


_load_env()

SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")


# ---------------------------------------------------------------------------
# History / cache
# ---------------------------------------------------------------------------

def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def load_history() -> dict:
    """Load history file. Returns dict with keys: date, cached_output, seen_keys."""
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except (json.JSONDecodeError, KeyError):
            pass
    return {"date": "", "cached_output": "", "seen_keys": []}


def save_history(date: str, cached_output: str, new_keys: list[str], old_keys: list[str],
                  *, slack_posted: bool = False) -> None:
    """Save today's output and accumulate seen keys."""
    all_keys = list(set(old_keys + new_keys))
    # Preserve existing slack_posted flag if already set today
    existing = load_history()
    if existing.get("date") == date and existing.get("slack_posted"):
        slack_posted = True
    HISTORY_FILE.write_text(json.dumps({
        "date": date,
        "cached_output": cached_output,
        "seen_keys": all_keys,
        "slack_posted": slack_posted,
    }, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TrendItem:
    title: str
    url: str
    source: str
    score: float = 0.0
    points: int = 0
    comments: int = 0
    published: str = ""
    description: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def dedup_key(self) -> str:
        """Stable key for deduplication based on normalised title."""
        norm = re.sub(r"\W+", " ", self.title.lower()).strip()
        return hashlib.md5(norm.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Relevance scoring
# ---------------------------------------------------------------------------

def relevance_score(item: TrendItem) -> float:
    """Score an item by relevance to 'AI at work' themes (0-100)."""
    text = f"{item.title} {item.description}".lower()

    ai_hits = sum(1 for kw in AI_KEYWORDS if kw in text)
    work_hits = sum(1 for kw in WORK_KEYWORDS if kw in text)

    # Base relevance: AI keywords are required; work keywords boost further
    ai_score = min(ai_hits * 8, 40)       # max 40 from AI keywords
    work_score = min(work_hits * 6, 30)    # max 30 from work keywords

    # Engagement bonus (normalised log-ish scale)
    engagement = 0
    if item.points > 0:
        engagement += min(item.points / 20, 15)  # max 15
    if item.comments > 0:
        engagement += min(item.comments / 10, 10)  # max 10

    # Recency bonus (max 5)
    recency = 0
    if item.published:
        try:
            for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ",
                        "%Y-%m-%dT%H:%M:%S.%f%z", "%a, %d %b %Y %H:%M:%S %z"):
                try:
                    pub_dt = datetime.strptime(item.published, fmt)
                    break
                except ValueError:
                    continue
            else:
                pub_dt = None
            if pub_dt:
                now = datetime.now(timezone.utc)
                age_hours = (now - pub_dt.astimezone(timezone.utc)).total_seconds() / 3600
                if age_hours < 24:
                    recency = 5
                elif age_hours < 72:
                    recency = 3
                elif age_hours < 168:
                    recency = 1
        except Exception:
            pass

    return round(ai_score + work_score + engagement + recency, 1)


# ---------------------------------------------------------------------------
# Source fetchers
# ---------------------------------------------------------------------------

async def fetch_hackernews(client: httpx.AsyncClient) -> list[TrendItem]:
    """Fetch AI-at-work stories from HackerNews Algolia API."""
    items: list[TrendItem] = []
    seen_ids: set[str] = set()

    async def _search(query: str, endpoint: str = "search") -> None:
        cutoff = int((datetime.now(timezone.utc).timestamp()) - 48 * 3600)
        url = (
            f"https://hn.algolia.com/api/v1/{endpoint}"
            f"?query={quote_plus(query)}"
            f"&tags=story"
            f"&hitsPerPage=10"
            f"&numericFilters=created_at_i>{cutoff}"
        )
        if endpoint == "search":
            url += ",points>10"

        try:
            resp = await client.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"  [HN] query={query!r} endpoint={endpoint} failed: {exc}",
                  file=sys.stderr)
            return

        for hit in data.get("hits", []):
            oid = hit.get("objectID", "")
            if oid in seen_ids:
                continue
            seen_ids.add(oid)

            story_url = hit.get("url") or f"https://news.ycombinator.com/item?id={oid}"
            items.append(TrendItem(
                title=hit.get("title", ""),
                url=story_url,
                source="HackerNews",
                points=hit.get("points", 0) or 0,
                comments=hit.get("num_comments", 0) or 0,
                published=hit.get("created_at", ""),
                extra={"hn_id": oid},
            ))

    # Fire all queries concurrently, mixing both endpoints
    tasks = []
    for q in HN_QUERIES:
        tasks.append(_search(q, "search"))
        tasks.append(_search(q, "search_by_date"))

    await asyncio.gather(*tasks)
    return items


async def fetch_techmeme(client: httpx.AsyncClient) -> list[TrendItem]:
    """Parse TechMeme RSS for AI-related articles."""
    items: list[TrendItem] = []
    try:
        resp = await client.get(
            "https://www.techmeme.com/feed.xml", timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
    except Exception as exc:
        print(f"  [TechMeme] fetch failed: {exc}", file=sys.stderr)
        return items

    for item_el in root.iter("item"):
        title = (item_el.findtext("title") or "").strip()
        link = (item_el.findtext("link") or "").strip()
        pub_date = (item_el.findtext("pubDate") or "").strip()
        desc_raw = (item_el.findtext("description") or "").strip()
        # Strip HTML tags from description
        desc = unescape(re.sub(r"<[^>]+>", " ", desc_raw)).strip()
        desc = re.sub(r"\s+", " ", desc)[:300]

        if not title or not link:
            continue

        items.append(TrendItem(
            title=title,
            url=link,
            source="TechMeme",
            published=pub_date,
            description=desc,
        ))

    return items


async def fetch_arxiv(client: httpx.AsyncClient) -> list[TrendItem]:
    """Fetch recent AI papers from ArXiv API."""
    items: list[TrendItem] = []
    url = (
        "http://export.arxiv.org/api/query"
        "?search_query=cat:cs.AI+OR+cat:cs.CL"
        "&sortBy=submittedDate&sortOrder=descending"
        "&max_results=30"
    )
    try:
        resp = await client.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
    except Exception as exc:
        print(f"  [ArXiv] fetch failed: {exc}", file=sys.stderr)
        return items

    ns = {"atom": "http://www.w3.org/2005/Atom"}

    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", "", ns) or "").strip()
        title = re.sub(r"\s+", " ", title)
        link_el = entry.find("atom:link[@rel='alternate']", ns)
        link = link_el.get("href", "") if link_el is not None else ""
        published = (entry.findtext("atom:published", "", ns) or "").strip()
        summary = (entry.findtext("atom:summary", "", ns) or "").strip()
        summary = re.sub(r"\s+", " ", summary)[:300]

        if not title:
            continue

        items.append(TrendItem(
            title=title,
            url=link,
            source="ArXiv",
            published=published,
            description=summary,
        ))

    return items


async def fetch_producthunt(client: httpx.AsyncClient) -> list[TrendItem]:
    """Parse Product Hunt Atom feed for AI-related products."""
    items: list[TrendItem] = []
    try:
        resp = await client.get(
            "https://www.producthunt.com/feed", timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
    except Exception as exc:
        print(f"  [ProductHunt] fetch failed: {exc}", file=sys.stderr)
        return items

    ns = {"atom": "http://www.w3.org/2005/Atom"}

    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", "", ns) or "").strip()
        link_el = entry.find("atom:link[@rel='alternate']", ns)
        link = link_el.get("href", "") if link_el is not None else ""
        published = (entry.findtext("atom:published", "", ns) or "").strip()
        content = (entry.findtext("atom:content", "", ns) or "").strip()
        content = unescape(re.sub(r"<[^>]+>", " ", content)).strip()
        content = re.sub(r"\s+", " ", content)[:300]

        if not title:
            continue

        items.append(TrendItem(
            title=title,
            url=link,
            source="ProductHunt",
            published=published,
            description=content,
        ))

    return items


async def fetch_github_trending(client: httpx.AsyncClient) -> list[TrendItem]:
    """Scrape GitHub trending page for AI-related repos."""
    items: list[TrendItem] = []
    try:
        resp = await client.get(
            "https://github.com/trending?since=daily",
            timeout=REQUEST_TIMEOUT,
            headers={**HEADERS, "Accept": "text/html"},
        )
        resp.raise_for_status()
        html = resp.text
    except Exception as exc:
        print(f"  [GitHub] fetch failed: {exc}", file=sys.stderr)
        return items

    # Parse the trending page using regex on the known HTML structure.
    # Each repo row is inside <article class="Box-row">
    article_pattern = re.compile(
        r'<article\s+class="Box-row">(.*?)</article>', re.DOTALL
    )
    # Repo link: inside <h2>, find the <a> with href="/owner/repo" and class containing "Link"
    repo_link_pattern = re.compile(
        r'<h2[^>]*>.*?<a[^>]*href="(/[^/]+/[^"]+)"[^>]*class="[^"]*Link[^"]*"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    # Description: <p class="col-9 ...">...</p>
    desc_pattern = re.compile(
        r'<p\s+class="[^"]*col-9[^"]*"[^>]*>(.*?)</p>', re.DOTALL
    )
    # Stars today: "N stars today"
    stars_today_pattern = re.compile(r'([\d,]+)\s+stars\s+today')
    # Total stars: inside the stargazers link text (after stripping SVG)
    total_stars_pattern = re.compile(
        r'href="[^"]+/stargazers"[^>]*>(.*?)</a>', re.DOTALL
    )

    for article_match in article_pattern.finditer(html):
        block = article_match.group(1)

        repo_match = repo_link_pattern.search(block)
        if not repo_match:
            continue
        repo_path = repo_match.group(1).strip()
        repo_name_raw = re.sub(r"<[^>]+>", " ", repo_match.group(2))
        repo_name = re.sub(r"\s+", " ", repo_name_raw).strip()
        repo_name = re.sub(r"\s*/\s*", "/", repo_name)

        desc = ""
        desc_match = desc_pattern.search(block)
        if desc_match:
            desc = re.sub(r"<[^>]+>", "", desc_match.group(1)).strip()
            desc = re.sub(r"\s+", " ", desc)

        stars_today = 0
        st_match = stars_today_pattern.search(block)
        if st_match:
            stars_today = int(st_match.group(1).replace(",", ""))

        total_stars = 0
        ts_match = total_stars_pattern.search(block)
        if ts_match:
            stars_raw = re.sub(r"<[^>]+>", "", ts_match.group(1)).strip()
            stars_raw = stars_raw.replace(",", "").replace(" ", "")
            if stars_raw.isdigit():
                total_stars = int(stars_raw)

        # Skip repos that aren't newly trending — if today's stars are
        # a tiny fraction of total stars, the repo has been popular for a while.
        if total_stars > 0 and stars_today > 0 and stars_today / total_stars < 0.02:
            continue

        items.append(TrendItem(
            title=repo_name,
            url=f"https://github.com{repo_path}",
            source="GitHub Trending",
            points=total_stars,
            description=desc,
            extra={"stars_today": stars_today},
        ))

    return items


# ---------------------------------------------------------------------------
# Aggregation pipeline
# ---------------------------------------------------------------------------

def deduplicate(items: list[TrendItem]) -> list[TrendItem]:
    """Remove duplicates, keeping the copy with the highest score."""
    best: dict[str, TrendItem] = {}
    for item in items:
        key = item.dedup_key
        if key not in best or item.score > best[key].score:
            best[key] = item
    return list(best.values())


def filter_ai_relevant(items: list[TrendItem], min_score: float = 5.0) -> list[TrendItem]:
    """Keep only items that pass a minimum relevance threshold."""
    return [it for it in items if it.score >= min_score]


def format_markdown(items_by_source: dict[str, list[TrendItem]]) -> str:
    """Render the aggregated results as a markdown report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# AI-at-Work Trend Report",
        f"_Generated {now}_\n",
    ]

    total = sum(len(v) for v in items_by_source.values())
    lines.append(f"**{total} relevant items** across {len(items_by_source)} sources.\n")

    for source, items in items_by_source.items():
        if not items:
            continue
        lines.append(f"---\n## {source} ({len(items)} items)\n")
        for item in items:
            badge = f"[score {item.score}]"
            engagement = ""
            if item.points:
                engagement += f" | {item.points} pts"
            if item.comments:
                engagement += f" | {item.comments} comments"
            stars_today = item.extra.get("stars_today")
            if stars_today:
                engagement += f" | +{stars_today} stars today"

            lines.append(f"- **[{item.title}]({item.url})** {badge}{engagement}")
            if item.description:
                short = item.description[:180]
                if len(item.description) > 180:
                    short += "..."
                lines.append(f"  _{short}_")
            lines.append("")

    return "\n".join(lines)


def format_json(items_by_source: dict[str, list[TrendItem]]) -> str:
    """Render the aggregated results as a JSON document."""
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {},
    }
    for source, items in items_by_source.items():
        output["sources"][source] = [asdict(it) for it in items]
    return json.dumps(output, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# LLM summarisation via Ollama
# ---------------------------------------------------------------------------

async def summarize_with_ollama(items: list[TrendItem]) -> list[dict]:
    """Send top items to a local Ollama model and get one-line summaries.

    Returns a list of dicts: {title, url, summary}.
    """
    # Build a numbered list for the prompt
    numbered = []
    for i, item in enumerate(items, 1):
        context = item.description[:200] if item.description else ""
        numbered.append(f"{i}. TITLE: {item.title}\n   CONTEXT: {context}")

    prompt = (
        "You are a tech news editor. For each item below, write exactly ONE short sentence "
        "(max 20 words) that explains why it matters to professionals using AI at work. "
        "Do NOT repeat the title. Focus on the practical implication or insight.\n\n"
        "Reply with ONLY a numbered list — one line per item, matching the numbers below. "
        "No bullet points, no extra commentary.\n\n"
        + "\n".join(numbered)
    )

    body = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 512},
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/generate", json=body)
        resp.raise_for_status()
        raw = resp.json().get("response", "")

    # Parse numbered responses back
    summaries: dict[int, str] = {}
    for line in raw.strip().splitlines():
        line = line.strip()
        m = re.match(r"(\d+)\.\s*(.+)", line)
        if m:
            summaries[int(m.group(1))] = m.group(2).strip()

    results = []
    for i, item in enumerate(items, 1):
        results.append({
            "title": item.title,
            "url": item.url,
            "source": item.source,
            "summary": summaries.get(i, ""),
        })
    return results


def format_digest(summarized: list[dict]) -> str:
    """Render the LLM-summarized digest as plain text for terminal."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"AI at Work — Daily Digest",
        f"Generated {now}",
        "=" * 50,
        "",
    ]
    for i, entry in enumerate(summarized, 1):
        lines.append(f"{i}. {entry['title']}")
        lines.append(f"   {entry['url']}")
        if entry["summary"]:
            lines.append(f"   → {entry['summary']}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Slack integration
# ---------------------------------------------------------------------------

_SOURCE_EMOJI = {
    "HackerNews": ":fire:",
    "TechMeme": ":newspaper:",
    "ArXiv": ":mortar_board:",
    "ProductHunt": ":rocket:",
    "GitHub Trending": ":star:",
}


def _escape_mrkdwn(text: str) -> str:
    """Escape &, <, > so they don't break Slack mrkdwn."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def format_slack_blocks(summarized: list[dict]) -> dict:
    """Build a Slack Block Kit payload from LLM-summarized items."""
    today = _today()
    sources = sorted({e["source"] for e in summarized})
    source_list = ", ".join(sources) if sources else "various"

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "AI at Work \u2014 Daily Digest", "emoji": True},
        },
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f":calendar: *{today}*  |  {len(summarized)} items  |  Sources: {source_list}"},
            ],
        },
        {"type": "divider"},
    ]

    for i, entry in enumerate(summarized, 1):
        emoji = _SOURCE_EMOJI.get(entry["source"], ":link:")
        title_escaped = _escape_mrkdwn(entry["title"])
        summary_escaped = _escape_mrkdwn(entry.get("summary", ""))
        url = entry["url"]

        text = f"*{i}. <{url}|{title_escaped}>*\n{emoji} {entry['source']}"
        if summary_escaped:
            text += f"\n{summary_escaped}"

        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": text},
        })

    blocks.append({"type": "divider"})
    blocks.append({
        "type": "context",
        "elements": [
            {"type": "mrkdwn", "text": "_Generated by AI Trend Finder_"},
        ],
    })

    return {"blocks": blocks}


async def post_to_slack(payload: dict) -> None:
    """Post a Block Kit payload to Slack via incoming webhook."""
    if not SLACK_WEBHOOK_URL:
        print("Error: SLACK_WEBHOOK_URL environment variable is not set.", file=sys.stderr)
        print("Set it with: export SLACK_WEBHOOK_URL='https://hooks.slack.com/services/...'", file=sys.stderr)
        sys.exit(1)

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(SLACK_WEBHOOK_URL, json=payload)
        if resp.status_code != 200:
            print(f"Slack webhook error ({resp.status_code}): {resp.text}", file=sys.stderr)
            sys.exit(1)

    print("Posted digest to Slack.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate AI-at-work trends from free sources."
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output structured JSON instead of markdown",
    )
    parser.add_argument(
        "--min-score", type=float, default=5.0,
        help="Minimum relevance score to include (default: 5.0)",
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="Summarize the top N items via local Ollama LLM (default: 10, 0 to disable)",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore cache and fetch fresh data even if already run today",
    )
    parser.add_argument(
        "--slack", action="store_true",
        help="Post digest to Slack (requires SLACK_WEBHOOK_URL env var)",
    )
    args = parser.parse_args()

    # --- Cache check: same day = return cached output ---
    history = load_history()
    today = _today()

    if history["date"] == today and not args.fresh:
        # Slack double-post guard
        if args.slack and history.get("slack_posted"):
            print("(Already posted to Slack today — skipping. Use --fresh to force.)", file=sys.stderr)
            return
        # Terminal cache
        if not args.slack and history["cached_output"]:
            print("(Returning cached results from earlier today)", file=sys.stderr)
            print(history["cached_output"])
            return

    # --- Fetch fresh data ---
    print("Fetching trends from free sources...", file=sys.stderr)

    async with httpx.AsyncClient(
        headers=HEADERS,
        follow_redirects=True,
        http2=False,
    ) as client:
        results = await asyncio.gather(
            fetch_hackernews(client),
            fetch_techmeme(client),
            fetch_arxiv(client),
            fetch_producthunt(client),
            fetch_github_trending(client),
            return_exceptions=True,
        )

    # Flatten, handling any top-level exceptions
    all_items: list[TrendItem] = []
    source_names = ["HackerNews", "TechMeme", "ArXiv", "ProductHunt", "GitHub Trending"]
    for name, result in zip(source_names, results):
        if isinstance(result, Exception):
            print(f"  [{name}] source failed entirely: {result}", file=sys.stderr)
            continue
        all_items.extend(result)

    print(f"  Fetched {len(all_items)} raw items.", file=sys.stderr)

    # Score every item
    for item in all_items:
        item.score = relevance_score(item)

    # Deduplicate
    all_items = deduplicate(all_items)
    print(f"  {len(all_items)} items after deduplication.", file=sys.stderr)

    # Filter by relevance
    relevant = filter_ai_relevant(all_items, min_score=args.min_score)
    print(f"  {len(relevant)} items above score threshold {args.min_score}.", file=sys.stderr)

    # Filter out previously seen items (from prior days)
    seen_keys = set(history.get("seen_keys", []))
    if seen_keys:
        before = len(relevant)
        relevant = [it for it in relevant if it.dedup_key not in seen_keys]
        filtered = before - len(relevant)
        if filtered:
            print(f"  {filtered} items filtered (already shown on previous days).", file=sys.stderr)

    if not relevant:
        msg = "No new AI-at-work trends found since last run."
        print(msg)
        save_history(today, msg, [], list(seen_keys))
        return

    # Sort globally by score
    relevant.sort(key=lambda x: -x.score)

    # Collect dedup keys for items we're about to show
    shown_keys = [it.dedup_key for it in relevant]

    # LLM digest mode
    if args.top > 0:
        top_items = relevant[: args.top]
        print(f"  Summarizing top {len(top_items)} items via Ollama ({OLLAMA_MODEL})...",
              file=sys.stderr)
        try:
            summarized = await summarize_with_ollama(top_items)
        except Exception as exc:
            print(f"  Ollama error: {exc}", file=sys.stderr)
            print("  Is Ollama running? Start it with: ollama serve", file=sys.stderr)
            sys.exit(1)

        if args.slack:
            payload = format_slack_blocks(summarized)
            await post_to_slack(payload)
            # Also cache the terminal version
            output = format_digest(summarized)
            save_history(today, output, shown_keys, list(seen_keys), slack_posted=True)
            return

        if args.json_output:
            output = json.dumps(summarized, indent=2, ensure_ascii=False)
        else:
            output = format_digest(summarized)

        print(output)
        save_history(today, output, shown_keys, list(seen_keys))
        return

    # Group by source and sort within each group (highest score first)
    items_by_source: dict[str, list[TrendItem]] = {}
    for item in relevant:
        items_by_source.setdefault(item.source, []).append(item)

    # Output
    if args.json_output:
        output = format_json(items_by_source)
    else:
        output = format_markdown(items_by_source)

    print(output)
    save_history(today, output, shown_keys, list(seen_keys))


if __name__ == "__main__":
    asyncio.run(main())
