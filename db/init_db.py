from db.database import Base, engine
from db.models import File, Page, Image, Chat, ChatMessage
# from indexing.retriever import retrieve_relevant_chunks
# from llm.generator import generate_answer

def init():
    print("🔧 Criando tabelas no banco...")
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init()

    # question = "What is the authors in the document?"
    # print(f"❓ PERGUNTA: {question}")

    # docs = retrieve_relevant_chunks("fe523df48ad2837d60c3b3b2e227594a911f06c2b990a215e6ff8cc44de4fa4f", question)
    # print("📄 DOCUMENTOS RECUPERADOS:")
    # for doc in docs:
    #     print(f"[Página {doc.metadata.get('page_number')}] {doc.page_content[:100]}...")
    # print("\n🔍 BUSCANDO RESPOSTAS...")
    # resposta = generate_answer(question, docs)
    # print("\n💬 RESPOSTA DO GEMINI:\n", resposta)