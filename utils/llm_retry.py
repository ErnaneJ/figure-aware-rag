import time
import random

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
                print(f"[⏳ LLM] Rate limit, retry em {wait:.2f}s (tentativa {attempt+1})")
                time.sleep(wait)
            else:
                raise
    print("[⚠️ LLM] Rate limit persistente, marcando como pendente")
    raise RateLimitException()
