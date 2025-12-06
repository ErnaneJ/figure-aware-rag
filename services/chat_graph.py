from langgraph.graph import StateGraph
import re
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from openai import AzureOpenAI

from types import SimpleNamespace

from pathlib import Path
from indexing.retriever import retrieve_relevant_chunks
from db.queries import get_chat_messages
import os
import httpx

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
    if not api_key and provider != 'ollama':  # Ollama pode não precisar de chave
        raise ValueError(f"{key_name} necessário para uso com {provider}")
    
    return api_key

def get_llm_chat():
    provider = os.getenv("LLM_PROVIDER_CHAT", "gemini").lower()
    model = os.getenv("LLM_MODEL_CHAT", "gemini-2.0-flash")
    api_key = get_api_key(provider)

    if provider == "gemini":
        print("[ℹ️ Chat LLM] Usando Gemini LLM para chat.")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.3
        )

    elif provider == "ollama":
        print("[ℹ️ Chat LLM] Usando Ollama LLM para chat.")
        from langchain_ollama import OllamaLLM

        return OllamaLLM(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
            temperature=0.3
        )

    elif provider == "openai":
        print("[ℹ️ Chat LLM] Usando OpenAI LLM para chat.")
        from langchain_openai import ChatOpenAI
        
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.3
        )

    elif provider == "azure-petrobras":
        print("[ℹ️ Chat LLM] Usando Azure Petrobras LLM para chat.")
        
        azure_base_url = os.getenv("AZURE_PETROBRAS_BASE_URL", "")
        api_version = os.getenv("AZURE_PETROBRAS_API_VERSION", "2025-01-01-preview")
        
        if not azure_base_url:
            raise ValueError("AZURE_PETROBRAS_BASE_URL necessário para uso com Azure Petrobras.")

        http_client = None
        ca_cert_path = os.getenv("AZURE_CA_CERT_PATH", "./petrobras-ca-root.pem")
        if os.path.exists(ca_cert_path):
            http_client = httpx.Client(verify=ca_cert_path)
            print(f"[ℹ️] Usando certificado customizado: {ca_cert_path}")
        else:
            print(f"[⚠️] Certificado não encontrado em {ca_cert_path}, usando verificação padrão")

        return AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            base_url=azure_base_url,
            http_client=http_client,
        )

    else:
        raise ValueError(f"LLM_PROVIDER_CHAT '{provider}' não suportado.")

def fetch_context(state):
    print("Fetching context for chat...")
    chat_id = state["chat_id"]
    question = state["question"]

    history = get_chat_messages(chat_id)
    memory = []

    for m in history[-5:]:
        if m.role == "user":
            memory.append(HumanMessage(content=m.content))
        else:
            memory.append(AIMessage(content=m.content))

    file_hashes = state["file_hashes"]

    all_docs = []
    for file_hash in file_hashes:
        docs = retrieve_relevant_chunks(file_hash, question, k=25)
        all_docs.extend(docs)

    print(f"Found {len(all_docs)} relevant documents for chat ID {chat_id}.")
    return {
        "chat_id": chat_id,
        "question": question,
        "docs": all_docs,
        "memory": memory
    }

def build_prompt_and_run_llm(state):
    try:
        from llm.generator import build_prompt

        print("Building prompt and invoking LLM...")

        llm = get_llm_chat()

        text_docs = [d for d in state["docs"] if d.metadata.get("source") == "text"]
        image_docs = [d for d in state["docs"] if d.metadata.get("source") == "image"]

        prompt = build_prompt(state["question"], text_docs, image_docs)

        response = None
        if isinstance(llm, AzureOpenAI):
            messages = state["memory"] + [{"role": "user", "content": [{ "type": "text", "text": prompt}]}]
            response = llm.chat.completions.create(
                model=os.getenv("LLM_MODEL_CHAT", "o4-mini-petrobras"),
                messages=messages,
                temperature=1,
                max_completion_tokens=500,
                frequency_penalty=0,
                presence_penalty=0,
            )
            
            response = SimpleNamespace(**{'content': response.choices[0].message.content})
        else:
            messages = state["memory"] + [HumanMessage(content=prompt)]
            response = llm.invoke(messages, request_timeout=120)

        answer = ""
        think = ""
        if response.__class__ is str:
            think = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
            think = think.group(1).strip() if think else ""
            answer = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        else:
            answer = response.content.strip()

        print(f"LLM response: {answer}")
        return {
            "chat_id": state["chat_id"],
            "answer": answer,
            "think": think,
            "text_docs": text_docs,
            "image_docs": image_docs,
            "llm_model": os.getenv("LLM_MODEL_CHAT", "unknown").lower(),
            "llm_provider": os.getenv("LLM_PROVIDER_CHAT", "unknown").lower(),
        }
    except Exception as e:
        return {
            "chat_id": state["chat_id"],
            "answer": f"Desculpe, ocorreu um erro ao processar sua solicitação. Erro: {str(e)}",
            "think": "...",
            "text_docs": [],
            "image_docs": [],
            "llm_model": os.getenv("LLM_MODEL_CHAT", "unknown").lower(),
            "llm_provider": os.getenv("LLM_PROVIDER_CHAT", "unknown").lower(),
        }

def build_chat_graph(save_workflow: bool = False):
    builder = StateGraph(state_schema=dict)

    builder.add_node("search", RunnableLambda(fetch_context))
    builder.add_node("llm", RunnableLambda(build_prompt_and_run_llm))

    builder.set_entry_point("search")
    builder.add_edge("search", "llm")
    builder.set_finish_point("llm")

    graph = builder.compile()
    
    if save_workflow:
        output_path = Path("llm/graph_outputs")
        output_path.mkdir(exist_ok=True)

        graph_path = output_path / "chat_workflow.png"
        image_data = graph.get_graph().draw_mermaid_png()
        with open(graph_path, "wb") as f:
            f.write(image_data)
        print(f"🖼️ Grafo salvo em: {graph_path}")

    return graph