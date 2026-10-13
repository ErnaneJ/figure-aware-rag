from celery import Celery
import os
from indexing.processor import extract_text_and_images
from db.database import SessionLocal
from db.models import Page, Image, File
from indexing.indexer import index_file
import time

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

redis_url = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
celery_app = Celery("papyrus_worker", broker=redis_url)

@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
    retry_kwargs={"max_retries": 3},
)
def process_pdf_task(self, file_hash: str, path: str):
    logger.info(f"Processing PDF: {path} (hash={file_hash})")

    try:
        start_time = time.time()
        t0 = time.time()
        content = extract_text_and_images(path, file_hash)
        logger.info(f"[TIME] Extraction: {(time.time()-t0)/60:.2f} min")

        t1 = time.time()
        db = SessionLocal()

        for page in content["pages"]:
            db.add(Page(
                file_hash=file_hash,
                page_number=page["page"],
                text=page["text"]
            ))

        for img in content["images"]:
            caption = img.get("caption")
            llm_description = img.get("llm_description")

            if caption == "__PENDING__":
                caption = None

            if llm_description == "__PENDING__":
                llm_description = None
    
            db.add(Image(
                file_hash=file_hash,
                page_number=img["page"],
                img_doc_index=img["img_doc_index"],
                img_page_index=img["img_page_index"],
                ext=img["ext"],
                path=img["path"],
                caption=caption,
                ocr_text=img.get("ocr_text", None),
                llm_description=llm_description
            ))

        file = db.query(File).filter(File.hash == file_hash).first()
        file.processed = True

        db.commit()
        db.close()
        
        logger.info(f"[TIME] DB save: {(time.time()-t1)/60:.2f} min")

        t2 = time.time()
        index_file(file_hash)
        logger.info(f"[TIME] Indexing: {(time.time()-t2)/60:.2f} min")

        elapsed = time.time() - start_time
        logger.info(f"[TIME] Finished {file_hash} in {elapsed/60:.2f} min")
        return True

    except Exception as e:
        logger.info(f"Failed to process PDF: {e}")
        raise
