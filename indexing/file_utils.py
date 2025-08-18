from db.queries import get_file_by_hash, add_file
from tasks.process_queue import process_pdf_task
from pathlib import Path
import hashlib

BASE_FILES_DIR = Path("files")

def calculate_sha256(file_bytes: bytes) -> str:
    sha256 = hashlib.sha256()
    sha256.update(file_bytes)
    return sha256.hexdigest()

def save_file_if_new(file_bytes: bytes, filename: str) -> dict:
    file_hash = calculate_sha256(file_bytes)
    existing = get_file_by_hash(file_hash)

    file_dir = BASE_FILES_DIR / file_hash
    file_dir.mkdir(parents=True, exist_ok=True)

    file_path = file_dir / "original.pdf"

    if existing:
        return {
            "hash": file_hash,
            "filename": existing.filename,
            "path": existing.path,
            "new": False,
        }

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    add_file({
        "hash": str(file_hash),
        "filename": str(filename),
        "path": str(file_path),
    })

    process_pdf_task.delay(file_hash, str(file_path))

    return {
        "hash": file_hash,
        "filename": filename,
        "path": str(file_path),
        "new": True,
    }
