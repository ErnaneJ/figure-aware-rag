import fitz  # PyMuPDF
from pathlib import Path
from utils.ocr import extract_ocr_text
from llm.image_describer import describe_image_with_context
import time
import hashlib
from dotenv import load_dotenv
from llm.caption_extractor import extract_caption_with_llm
import os

load_dotenv(override=True, verbose=True)

def extract_text_and_images(pdf_path: str, file_hash: str) -> dict:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {pdf_path}")

    EXTRACTED_DIR = Path("files") / file_hash / "extracted"
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    result = {"pages": [], "images": []}
    img_doc_index = 1
    processed_hashes = {}

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        result["pages"].append({
            "page": page_num,
            "text": text.strip(),
        })

        image_dir = EXTRACTED_DIR

        for img_page_index, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            current_hash = hashlib.md5(image_bytes).hexdigest()
            
            if current_hash in processed_hashes:
                existing_info = processed_hashes[current_hash]
                image_info = {
                    "page": page_num,
                    "img_doc_index": img_doc_index,
                    "img_page_index": img_page_index,
                    "ext": image_ext,
                    "path": str(image_dir / f"img{img_doc_index}-page{page_num}_img_{img_page_index}.{image_ext}"),
                    "caption": existing_info["caption"],
                    "ocr_text": existing_info["ocr_text"],
                    "llm_description": existing_info["llm_description"],
                    "is_duplicate": True
                }
                result["images"].append(image_info)
                img_doc_index += 1
                continue
            
            image_filename = f"img{img_doc_index}-page{page_num}_img_{img_page_index}.{image_ext}"
            image_path = image_dir / image_filename

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            if os.getenv("DISABLE_OCR", "false").lower() != "true":
                print(f"Extraindo OCR para {image_path}")
                ocr_text = extract_ocr_text(image_path)
                print(f"OCR extraído: {ocr_text}")
            else:
                print(f"OCR desativado para {image_path}")
                ocr_text = ""

            if os.getenv("DISABLE_MULTIMODAL_DESCRIPTION", "false").lower() != "true":
                print(f"Extraindo descrição multimodal para {image_path}")
                llm_description = describe_image_with_context(image_path=image_path, page_text=text)
                print(f"Descrição obtida: {llm_description}")
                time.sleep(3)
            else:
                print(f"Descrição multimodal desativada para {image_path}")
                llm_description = ""
            
            if os.getenv("DISABLE_LLM_CAPTION_EXTRACTION", "false").lower() != "true":
                print(f"Extraindo legenda LLM para {image_path}")
                caption = extract_caption_with_llm(
                    page_text=text,
                    filename=path.name,
                    page_number=page_num,
                    img_doc_index=img_doc_index
                )
                print(f"Legenda obtida: {caption}")
                time.sleep(3)
            else:
                print(f"Extração de legenda LLM desativada para {image_path}")
                caption = ""

            image_info = {
                "page": page_num,
                "img_doc_index": img_doc_index,
                "img_page_index": img_page_index,
                "ext": image_ext,
                "path": str(image_path),
                "caption": caption,
                "ocr_text": ocr_text,
                "llm_description": llm_description,
                "is_duplicate": False
            }

            result["images"].append(image_info)
            processed_hashes[current_hash] = {
                "caption": caption,
                "ocr_text": ocr_text,
                "llm_description": llm_description
            }
            img_doc_index += 1

    doc.close()
    return result