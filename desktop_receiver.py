import redis
import json
import logging
import os
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Configuration
REDIS_HOST = os.environ.get("REDIS_HOST", "192.168.1.100") # Replace with Pi 3 B IP
REDIS_PORT = 6379
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "your_password_here")
REDIS_QUEUE_KEY = "logic_engine:ingest_queue"
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))

os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "corpus_raw.jsonl")

# Connect to Redis
try:
    r_client = redis.Redis(
        host=REDIS_HOST, 
        port=REDIS_PORT, 
        password=REDIS_PASSWORD, 
        db=0,
        health_check_interval=30,
        socket_keepalive=True
    )
    r_client.ping()
    logging.info("Connected to Redis broker.")
except Exception as e:
    logging.error(f"Failed to connect to Redis: {e}")
    exit(1)

def listen_and_write():
    logging.info(f"Listening for data from Pi 5. Writing to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        while True:
            try:
                # BLPOP blocks until an item is available in the queue
                # Use a bounded timeout instead of 0 to prevent dead sockets.
                result = r_client.blpop(REDIS_QUEUE_KEY, timeout=10)
                if result:
                    _, payload_bytes = result
                    payload_str = payload_bytes.decode('utf-8')
                    
                    # Verify it's valid JSON
                    data = json.loads(payload_str)
                    
                    # Write as JSONL
                    f.write(json.dumps(data) + "\n")
                    f.flush()
                    logging.info(f"Wrote payload from {data.get('domain')} to disk.")
            except redis.exceptions.TimeoutError:
                # Safely loop on queue empty timeout
                continue
            except redis.exceptions.ConnectionError as ce:
                logging.warning(f"Connection dropped, retrying: {ce}")
                time.sleep(2)
            except Exception as e:
                logging.error(f"Error processing payload: {e}")
                time.sleep(1)

if __name__ == "__main__":
    listen_and_write()
