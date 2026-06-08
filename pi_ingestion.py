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
                soup = BeautifulSoup(html, "html.parser")
                
                # We need raw text, but specifically ignoring HTML structure.
                # For a real pipeline, we would parse .tex files or raw GitHub user content directly.
                # This is a simplified fetcher for demonstration of the pipeline architecture.
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
    # Example targets (In reality, we would stream via ArXiv API and GitHub API)
    targets = [
        # CS Corpuses (Raw file URLs preferred in reality)
        ("https://raw.githubusercontent.com/torvalds/linux/master/mm/memory.c", "cs", "github"),
        ("https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/cuda/Attention.cu", "cs", "github"),
        # Math Corpuses (ArXiv API/Export)
        ("https://export.arxiv.org/api/query?search_query=cat:math.RT&start=0&max_results=5", "math", "arxiv")
    ]
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url, domain, source in targets:
            tasks.append(fetch_and_parse(session, url, domain, source))
        
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    logging.info("Starting Data Ingestion Pipeline on Pi 5...")
    asyncio.run(run_pipeline())
    logging.info("Pipeline batch complete. Waiting for next cron/trigger.")
