from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from indexing.retriever import retrieve_relevant_chunks
from db.queries import get_chat_messages
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)

def fetch_context(state):
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

    return {
        "chat_id": chat_id,
        "question": question,
        "docs": all_docs,
        "memory": memory
    }

def build_prompt_and_run_llm(state):
    from llm.generator import build_prompt

    text_docs = [d for d in state["docs"] if d.metadata.get("source") == "text"]
    image_docs = [d for d in state["docs"] if d.metadata.get("source") == "image"]

    prompt = build_prompt(state["question"], text_docs, image_docs)

    messages = state["memory"] + [HumanMessage(content=prompt)]
    response = llm.invoke(messages)

    return {
        "chat_id": state["chat_id"],
        "answer": response.content,
        "text_docs": text_docs,
        "image_docs": image_docs
    }

def build_chat_graph():
    graph = StateGraph(state_schema=dict)

    graph.add_node("search", RunnableLambda(fetch_context))
    graph.add_node("llm", RunnableLambda(build_prompt_and_run_llm))
    
    graph.set_entry_point("search")
    graph.add_edge("search", "llm")
    graph.set_finish_point("llm")

    return graph.compile()
