import os
from dotenv import load_dotenv
import httpx
from openai import AzureOpenAI
from types import SimpleNamespace

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

def llm_constructor():
    provider = os.getenv("LLM_PROVIDER_CAPTIONS", "gemini").lower()
    model_name = os.getenv("LLM_MODEL_CAPTIONS", "gemini-2.0-flash")
    api_key = get_api_key(provider)

    if provider == "gemini":
        logger.info("Caption LLM: Gemini")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.3,
        )

    elif provider == "ollama":
        logger.info("Caption LLM: Ollama")
        from langchain_ollama import OllamaLLM

        return OllamaLLM(
            model=model_name,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
            temperature=0.3,
        )

    elif provider == "openai":
        logger.info("Caption LLM: OpenAI")
        from langchain_openai import ChatOpenAI
        
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.3,
        )

    elif provider == "azure-petrobras":
        logger.info("Caption LLM: Azure Petrobras")
        
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

        class AzureCaptionWrapper:
            def __init__(self, client, model_name):
                self.client = client
                self.model_name = model_name
            
            def invoke(self, context: str):
                try:
                    messages = [
                        {
                            "role": "system", 
                            "content": "You are an assistant who must respond based on the textual content of the page."
                        },
                        {
                            "role": "user", 
                            "content": context
                        }
                    ]
                    
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=1,
                        max_completion_tokens=1000,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
                    
                    return SimpleNamespace(content=response.choices[0].message.content.strip())
                
                except Exception as e:
                    logger.info(f"Azure caption LLM error: {e}")
                    return SimpleNamespace(content="UNDEFINED")

        return AzureCaptionWrapper(client, model_name)

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def extract_caption_with_llm(
    page_text: str,
    filename: str,
    page_number: int,
    img_doc_index: int
) -> str | None:
    """Ask the LLM for the caption of image `img_doc_index` as written in the page text.
    Returns None when no caption is found."""

    question = (
        f"Does the image ({img_doc_index}) on page {page_number} of file {filename} have a visible caption in the text?"
        f"If so, return only the exact caption as it appears in the text, without interpreting or rewriting it."
        f"The caption can be anywhere in the page text, but is usually near the image."
        f"The content can begin with words such as 'Fig.', 'Figure', 'Image', 'Photo', 'Graph', 'Table', or similar."
        f"and can contain numbers, letters, or symbols. The caption can be long or short."
        f"and can contain punctuation, such as a colon, dash, or parentheses."
        f"If there is no clearly defined caption for this image, simply say: 'UNDEFINED'."
    )

    context = (
        f"You are an assistant who must answer based on the textual content of the page.\n\n"
        f"FILE: {filename} | PAGE: {page_number} | IMAGE: {img_doc_index}\n\n"
        f"PAGE CONTENT:\n---\n{page_text}\n---\n\n"
        f"QUESTION:\n{question}\n\n"
        f"Answer exclusively with the exact caption or with the term UNDEFINED."
    )

    llm = llm_constructor()

    response = llm.invoke(context)

    if response.__class__ is str:
        answer = response.strip()
    else:
        answer = response.content.strip()

    if "UNDEFINED" in answer.upper():
        return None

    return answer