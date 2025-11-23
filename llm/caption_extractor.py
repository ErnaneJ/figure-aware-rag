import os
from dotenv import load_dotenv

load_dotenv(override=True, verbose=True)

def llm_constructor() -> object:
    provider = os.getenv("LLM_PROVIDER_CAPTIONS", "gemini").lower()
    model_name = os.getenv("LLM_MODEL_CAPTIONS", "gemini-2.0-flash")
    api_key = os.getenv("LLM_API_KEY_CAPTIONS", "")

    if provider == "gemini":
        print("[ℹ️ Caption LLM] Usando Gemini LLM para extração de legendas.")
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not api_key:
            raise ValueError("LLM_API_KEY_CAPTIONS necessário para uso com Gemini.")

        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.3,
        )

    elif provider == "ollama":
        print("[ℹ️ Caption LLM] Usando Ollama LLM para extração de legendas.")
        from langchain_ollama import OllamaLLM

        return OllamaLLM(
            model=model_name,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
            temperature=0.3,
        )

    elif provider == "openai":
        print("[ℹ️ Caption LLM] Usando OpenAI LLM para extração de legendas.")
        from langchain_openai import ChatOpenAI
        
        if not api_key:
            raise ValueError("LLM_API_KEY_CAPTIONS necessário para uso com OpenAI.")

        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.3,
        )

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