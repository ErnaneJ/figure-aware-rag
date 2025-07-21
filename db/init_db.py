from db.database import Base, engine

def reset_db():
    print("⚠️  Apagando tabelas existentes...")
    Base.metadata.drop_all(bind=engine)

    print("🔧 Criando tabelas do zero...")
    Base.metadata.create_all(bind=engine)

    print("✅ Banco de dados reinicializado com sucesso.")

if __name__ == "__main__":
    reset_db()
