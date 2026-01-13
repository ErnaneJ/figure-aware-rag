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
    """Obtém a chave API baseada no provedor"""
    provider_keys = {
        'openai': 'OPENAI_API_KEY',
        'gemini': 'GEMINI_API_KEY', 
        'ollama': 'OLLAMA_API_KEY',
        'azure-petrobras': 'AZURE_PETROBRAS_API_KEY'
    }
    
    key_name = provider_keys.get(provider)
    if not key_name:
        raise ValueError(f"Provedor '{provider}' não suportado")
    
    api_key = os.getenv(key_name)
    if not api_key and provider != 'ollama':
        raise ValueError(f"{key_name} necessário para uso com {provider}")
    
    return api_key

def llm_constructor():
    provider = os.getenv("LLM_PROVIDER_CAPTIONS", "gemini").lower()
    model_name = os.getenv("LLM_MODEL_CAPTIONS", "gemini-2.0-flash")
    api_key = get_api_key(provider)

    if provider == "gemini":
        logger.info("[ℹ️ Caption LLM] Usando Gemini LLM para extração de legendas.")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.3,
        )

    elif provider == "ollama":
        logger.info("[ℹ️ Caption LLM] Usando Ollama LLM para extração de legendas.")
        from langchain_ollama import OllamaLLM

        return OllamaLLM(
            model=model_name,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
            temperature=0.3,
        )

    elif provider == "openai":
        logger.info("[ℹ️ Caption LLM] Usando OpenAI LLM para extração de legendas.")
        from langchain_openai import ChatOpenAI
        
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.3,
        )

    elif provider == "azure-petrobras":
        logger.info("[ℹ️ Caption LLM] Usando Azure Petrobras LLM para extração de legendas.")
        
        azure_base_url = os.getenv("AZURE_PETROBRAS_BASE_URL", "")
        api_version = os.getenv("AZURE_PETROBRAS_API_VERSION", "2025-01-01-preview")
        
        if not azure_base_url:
            raise ValueError("AZURE_PETROBRAS_BASE_URL necessário para uso com Azure Petrobras.")

        http_client = None
        ca_cert_path = os.getenv("AZURE_CA_CERT_PATH", "./petrobras-ca-root.pem")
        if os.path.exists(ca_cert_path):
            http_client = httpx.Client(verify=ca_cert_path)
            logger.info(f"[ℹ️] Usando certificado customizado: {ca_cert_path}")
        else:
            logger.info(f"[⚠️] Certificado não encontrado em {ca_cert_path}, usando verificação padrão")

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
                            "content": "Você é um assistente que deve responder com base no conteúdo textual da página."
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
                    logger.info(f"[❌ Azure Caption LLM] Erro: {e}")
                    return SimpleNamespace(content="UNDEFINED")

        return AzureCaptionWrapper(client, model_name)

    else:
        raise ValueError(f"LLM provider não suportado: {provider}")

def extract_caption_with_llm(
    page_text: str,
    filename: str,
    page_number: int,
    img_doc_index: int
) -> str | None:
    """
    Usa LLM para identificar e retornar a legenda da imagem `img_doc_index` com base no texto da página.
    Retorna apenas a legenda exata, ou None se não houver.
    """

    question = (
        f"A imagem ({img_doc_index}) presente na página {page_number} do arquivo {filename} possui alguma legenda visível no texto?"
        f" Se sim, retorne apenas a legenda exata como ela aparece no texto, sem interpretar ou reescrever."
        f" A legenda pode estar em qualquer parte do texto da página, mas geralmente está próxima da imagem."
        f" O conteúdo pode iniciar com palavras como 'Fig.', 'Figura', 'Imagem', 'Foto', 'Gráfico', 'Tabela' ou similares,"
        f" e pode conter números, letras ou símbolos. A legenda pode ser longa ou curta,"
        f" e pode conter pontuação, como dois pontos, travessão ou parênteses."
        f" Caso não exista nenhuma legenda claramente relacionada a essa imagem, diga apenas: 'UNDEFINED'."
    )

    context = (
        f"Você é um assistente que deve responder com base no conteúdo textual da página.\n\n"
        f"ARQUIVO: {filename} | PÁGINA: {page_number} | IMAGEM: {img_doc_index}\n\n"
        f"CONTEÚDO DA PÁGINA:\n---\n{page_text}\n---\n\n"
        f"PERGUNTA:\n{question}\n\n"
        f"Responda exclusivamente com a legenda exata ou com o termo UNDEFINED."
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