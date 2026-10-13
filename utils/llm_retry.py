import time
import random

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RateLimitException(Exception):
    pass

def llm_call_with_retry(fn, max_retries=5, base_wait=1):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate limit" in msg:
                wait = base_wait * (2 ** attempt) + random.uniform(0, 0.5)
                logger.info(f"LLM rate limited, retrying in {wait:.2f}s (attempt {attempt+1})")
                time.sleep(wait)
            else:
                raise
    logger.info("LLM still rate limited, marking as pending")
    raise RateLimitException()
