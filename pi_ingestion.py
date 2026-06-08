import asyncio
import aiohttp
from bs4 import BeautifulSoup
import redis
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Configuration
REDIS_HOST = os.environ.get("REDIS_HOST", "192.168.1.100") # Replace with Pi 3 B IP
REDIS_PORT = 6379
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "your_password_here")
REDIS_QUEUE_KEY = "logic_engine:ingest_queue"

# Connect to Redis message broker
try:
    r_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, db=0)
    r_client.ping()
    logging.info("Connected to Redis broker.")
except Exception as e:
    logging.error(f"Failed to connect to Redis: {e}")
    exit(1)

async def fetch_and_parse(session, url, domain, source_type):
    try:
        async with session.get(url, timeout=15) as response:
            if response.status == 200:
                html = await response.text()
                # Use XML parser for ArXiv endpoints to prevent HTML parsing warnings
                if "arxiv.org/api" in url:
                    soup = BeautifulSoup(html, features="xml")
                else:
                    soup = BeautifulSoup(html, "html.parser")
                
                # We need raw text. This fetches pure code or raw LaTeX tags.
                text = soup.get_text(separator="\n", strip=True)
                
                # Payload schema enforcement
                payload = {
                    "domain": domain,
                    "source_uri": url,
                    "text": text[:50000] # Limiting size per payload for memory safety
                }
                
                # Push to Redis queue
                r_client.rpush(REDIS_QUEUE_KEY, json.dumps(payload))
                logging.info(f"Successfully ingested & queued: {url}")
            else:
                logging.warning(f"Failed to fetch {url} - Status: {response.status}")
    except Exception as e:
        logging.error(f"Error processing {url}: {e}")

async def run_pipeline():
    # Full Production Scale Target Array
    targets = [
        # CS Corpuses: High-complexity systems, memory management, and compilers
        ("https://raw.githubusercontent.com/torvalds/linux/master/mm/memory.c", "cs", "github"),
        ("https://raw.githubusercontent.com/torvalds/linux/master/kernel/sched/core.c", "cs", "github"),
        ("https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/cuda/Indexing.cu", "cs", "github"),
        ("https://raw.githubusercontent.com/rust-lang/rust/master/library/core/src/slice/mod.rs", "cs", "github"),
        ("https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/lib/CodeGen/SelectionDAG/SelectionDAG.cpp", "cs", "github"),
        
        # Data Science Corpuses: Heavily vectorized Python and C matrix math
        ("https://raw.githubusercontent.com/pandas-dev/pandas/main/pandas/core/frame.py", "data", "github"),
        ("https://raw.githubusercontent.com/numpy/numpy/main/numpy/_core/src/multiarray/array_assign_array.c", "data", "github"),
        ("https://raw.githubusercontent.com/scipy/scipy/main/scipy/linalg/basic.py", "data", "github"),
        
        # Mathematics Corpuses: Raw LaTeX and theorem structures via ArXiv API
        ("https://export.arxiv.org/api/query?search_query=cat:math.RT&start=0&max_results=50", "math", "arxiv"),
        ("https://export.arxiv.org/api/query?search_query=cat:math.DG&start=0&max_results=50", "math", "arxiv"),
        ("https://export.arxiv.org/api/query?search_query=cat:cs.DS&start=0&max_results=50", "math", "arxiv")
    ]
    headers = {"User-Agent": "LogicEngine/1.0 (Research Pipeline; +http://localhost)"}
    async with aiohttp.ClientSession(headers=headers) as session:
        for url, domain, source in targets:
            await fetch_and_parse(session, url, domain, source)
            # Enforce strict 3-second delay to comply with ArXiv API rate limits
            await asyncio.sleep(3)

if __name__ == "__main__":
    logging.info("Starting Data Ingestion Pipeline on Pi 5...")
    asyncio.run(run_pipeline())
    logging.info("Pipeline batch complete. Waiting for next cron/trigger.")
