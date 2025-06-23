import os
import base64
import mimetypes
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

def describe_image_with_context(image_path: str, page_text: str = "") -> str | None:
    """
    Usa o Gemini multimodal para descrever detalhadamente uma imagem,
    com base no conteúdo da página onde ela está inserida (se fornecido).
    """
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"

        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                (
                    "Considere a imagem a seguir. Descreva em um único parágrafo ela, com riqueza de detalhes, "
                    "sem fazer suposições ou inferências com base em qualquer contexto textual. Ignore o conteúdo da página se ele não estiver "
                    "visivelmente representado na imagem.\n\n"
                    f"O texto da página é apenas para fornecer contexto do tipo de documento:\n\n{page_text}\n\n"
                    "Use terceira pessoa, seja objetivo e descreva somente do que se trata a imagem."
                ),
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            ],
        )

        return response.text.strip()

    except Exception as e:
        print(f"[Multimodal] Erro ao descrever imagem '{image_path}': {e}")
        return None