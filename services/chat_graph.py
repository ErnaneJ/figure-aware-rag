from langgraph.graph import StateGraph
import re
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from pathlib import Path
from indexing.retriever import retrieve_relevant_chunks
from db.queries import get_chat_messages
import os

def get_llm_chat():
    provider = os.getenv("LLM_PROVIDER_CHAT", "gemini").lower()
    model = os.getenv("LLM_MODEL_CHAT", "gemini-2.0-flash")
    api_key = os.getenv("LLM_API_KEY_CHAT", "")

    if provider == "gemini":
        print("[ℹ️ Chat LLM] Usando Gemini LLM para chat.")
        if not api_key:
            raise ValueError("LLM_API_KEY_CHAT necessário para uso com Gemini.")
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
        docs = retrieve_relevant_chunks(file_hash, question, k=10)
        all_docs.extend(docs)

    print(f"Found {len(all_docs)} relevant documents for chat ID {chat_id}.")
    return {
        "chat_id": chat_id,
        "question": question,
        "docs": all_docs,
        "memory": memory
    }

def build_prompt_and_run_llm(state):
    from llm.generator import build_prompt

    print("Building prompt and invoking LLM...")

    llm = get_llm_chat()

    text_docs = [d for d in state["docs"] if d.metadata.get("source") == "text"]
    image_docs = [d for d in state["docs"] if d.metadata.get("source") == "image"]

    prompt = build_prompt(state["question"], text_docs, image_docs)

    messages = state["memory"] + [HumanMessage(content=prompt)]
    response = llm.invoke(messages)

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
        "image_docs": image_docs
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

