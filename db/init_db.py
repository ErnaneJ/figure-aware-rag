from sqlalchemy import inspect
from db.database import Base, engine
from db import models  # noqa: F401

def reset_db():
    print("ENGINE:", engine.url)
    print("⚠️  Apagando tabelas existentes...")

    print("ℹ️ Inspect:", inspect(engine).get_table_names())

    Base.metadata.drop_all(bind=engine, checkfirst=False)

    print("🔧 Criando tabelas do zero...")
    Base.metadata.create_all(bind=engine, checkfirst=False)

    print("Depois do create:", inspect(engine).get_table_names())
    print("✅ Banco de dados reinicializado com sucesso.")

if __name__ == "__main__":
    reset_db()
