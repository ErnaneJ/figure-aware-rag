import fitz  # PyMuPDF
from pathlib import Path

def extract_text_and_images(pdf_path: str, file_hash: str) -> dict:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {pdf_path}")

    # Nova pasta: files/<hash>/extracted/
    EXTRACTED_DIR = Path("files") / file_hash / "extracted"
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    result = {
        "pages": [],
        "images": [],
    }

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        result["pages"].append({
            "page": page_num,
            "text": text.strip(),
        })

        for img_index, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_filename = f"img_{img_index}_page{page_num}.{image_ext}"
            image_path = EXTRACTED_DIR / image_filename

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            image_info = {
                "page": page_num,
                "index": img_index,
                "ext": image_ext,
                "path": str(image_path),
            }

            result["images"].append(image_info)

    doc.close()
    return result
