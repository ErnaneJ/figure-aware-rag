from PIL import Image
import pytesseract
import re

def extract_ocr_text(image_path: str) -> str | None:
    """
    Realiza OCR na imagem indicada usando português e inglês.
    Retorna o texto extraído se houver conteúdo significativo, caso contrário retorna None.
    """
    try:
        img = Image.open(image_path).convert("RGB")

        raw_text = pytesseract.image_to_string(img, lang="por+eng")
        cleaned_text = raw_text.strip()

        cleaned_text = re.sub(r"[\n\r]+", " ", cleaned_text)
        cleaned_text = re.sub(r"[^\w\s,.!?€$%-]", "", cleaned_text).strip()

        return cleaned_text

    except Exception as e:
        print(f"[OCR] Erro ao processar imagem '{image_path}': {e}")
        return None