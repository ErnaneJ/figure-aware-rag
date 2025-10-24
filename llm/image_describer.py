import os
import base64
from dotenv import load_dotenv
import mimetypes
import base64
import httpx
import json

load_dotenv(override=True, verbose=True)

def get_multimodal_llm():
    provider = os.getenv("LLM_PROVIDER_IMAGE_DESCRIBER", "gemini").lower()
    model = os.getenv("LLM_MODEL_IMAGE_DESCRIBER", "gemini-2.0-flash")
    api_key = os.getenv("LLM_API_KEY_IMAGE_DESCRIBER", "")

    if provider == "gemini":
        from google import genai
        client = genai.Client(api_key=api_key)
        return lambda prompt, image_bytes, mime_type: client.models.generate_content(
            model=model,
            contents=[
                prompt,
                genai.types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            ],
        ).text.strip()

    elif provider == "ollama":
        def ollama_image_call(prompt: str, image_bytes: bytes, mime_type: str) -> str:
            img_b64 = base64.b64encode(image_bytes).decode("utf-8")
            base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

            try:
                response = httpx.post(
                    f"{base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "images": [img_b64]
                    },
                    timeout=120
                )

                chunks = []
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                        if "response" in parsed:
                            chunks.append(parsed["response"])
                    except Exception as e:
                        print(f"[⚠️ Erro ao parsear chunk] {e} > {line}")

                return "".join(chunks).strip()

            except Exception as e:
                print(f"[❌ Ollama LLM] Erro: {e}")
                return None

        return ollama_image_call

    else:
        raise ValueError(f"LLM provider '{provider}' não suportado para imagem multimodal.")
    
def describe_image_with_context(image_path: str, page_text: str = "") -> str | None:
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"

        prompt = (
            "Considere a imagem a seguir. Descreva em um único parágrafo ela, com riqueza de detalhes, "
            "sem fazer suposições ou inferências com base em qualquer contexto textual. Ignore o conteúdo da página se ele não estiver "
            "visivelmente representado na imagem.\n\n"
            f"O texto da página é apenas para fornecer contexto do tipo de documento:\n\n{page_text}\n\n"
            "Use terceira pessoa, seja objetivo e descreva somente do que se trata a imagem."
        )

        multimodal_llm = get_multimodal_llm()
        return multimodal_llm(prompt, image_bytes, mime_type)

    except Exception as e:
        print(f"[Multimodal] Error to describe image {image_path}: {e}")
        return None