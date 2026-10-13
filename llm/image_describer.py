import os
import base64
from dotenv import load_dotenv
import mimetypes
import httpx
import json
from openai import AzureOpenAI

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(override=True, verbose=True)

def get_api_key(provider):
    provider_keys = {
        'openai': 'OPENAI_API_KEY',
        'gemini': 'GEMINI_API_KEY', 
        'ollama': 'OLLAMA_API_KEY',
        'azure-petrobras': 'AZURE_PETROBRAS_API_KEY'
    }
    
    key_name = provider_keys.get(provider)
    if not key_name:
        raise ValueError(f"Unsupported provider: '{provider}'")
    
    api_key = os.getenv(key_name)
    if not api_key and provider != 'ollama':
        raise ValueError(f"{key_name} is required for {provider}")
    
    return api_key

def get_multimodal_llm():
    provider = os.getenv("LLM_PROVIDER_IMAGE_DESCRIBER", "gemini").lower()
    model = os.getenv("LLM_MODEL_IMAGE_DESCRIBER", "gemini-2.0-flash")
    api_key = get_api_key(provider)

    if provider == "gemini":
        logger.info("Multimodal LLM: Gemini")
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
        logger.info("Multimodal LLM: Ollama")
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
                        logger.info(f"Could not parse Ollama chunk: {e} > {line}")

                return "".join(chunks).strip()

            except Exception as e:
                logger.info(f"Ollama LLM error: {e}")
                return None

        return ollama_image_call

    elif provider == "openai":
        logger.info("Multimodal LLM: OpenAI")
        def openai_image_call(prompt: str, image_bytes: bytes, mime_type: str) -> str:
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            img_b64 = base64.b64encode(image_bytes).decode("utf-8")

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{img_b64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1000
            }

            try:
                response = httpx.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                response.raise_for_status()
                
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()

            except Exception as e:
                logger.info(f"OpenAI LLM error: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    logger.info(f"Error response: {e.response.text}")
                return None

        return openai_image_call

    elif provider == "azure-petrobras":
        logger.info("Multimodal LLM: Azure Petrobras")
        
        def azure_openai_image_call(prompt: str, image_bytes: bytes, mime_type: str) -> str:
            try:
                azure_base_url = os.getenv("AZURE_PETROBRAS_BASE_URL", "")
                api_version = os.getenv("AZURE_PETROBRAS_API_VERSION", "2025-01-01-preview")
                
                if not azure_base_url:
                    raise ValueError("AZURE_PETROBRAS_BASE_URL is required for azure-petrobras")

                http_client = None
                ca_cert_path = os.getenv("AZURE_CA_CERT_PATH", "./petrobras-ca-root.pem")
                if os.path.exists(ca_cert_path):
                    http_client = httpx.Client(verify=ca_cert_path)
                    logger.info(f"Using custom CA certificate: {ca_cert_path}")
                else:
                    logger.info(f"CA certificate not found at {ca_cert_path}, using default verification")

                client = AzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    base_url=azure_base_url,
                    http_client=http_client,
                )

                img_b64 = base64.b64encode(image_bytes).decode("utf-8")

                messages = [
                    {
                        "role": "system",
                        "content": "You are an assistant who describes images in an objective and detailed manner."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{img_b64}"
                                }
                            }
                        ]
                    }
                ]

                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=1,
                    max_completion_tokens=1000,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                
                return response.choices[0].message.content.strip()

            except Exception as e:
                logger.info(f"Azure Petrobras LLM error: {e}")
                return None

        return azure_openai_image_call

    else:
        raise ValueError(f"Unsupported multimodal LLM provider: '{provider}'")
    
def describe_image_with_context(image_path: str, page_text: str = "") -> str | None:
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"

        prompt = (
            "Consider the following image. Describe it in a single paragraph, with rich detail, "
            "without making assumptions or inferences based on any textual context. Ignore the page content if it is not "
            "visibly represented in the image.\n\n"
            f"The page text is only to provide context for the document type:\n\n{page_text}\n\n"
            "Use the third person, be objective, and describe only what the image is about."
        )

        multimodal_llm = get_multimodal_llm()
        return multimodal_llm(prompt, image_bytes, mime_type)

    except Exception as e:
        logger.info(f"Image description failed for {image_path}: {e}")
        return None