from sqlalchemy import inspect
from db.database import Base, engine
from db import models  # noqa: F401

def reset_db():
    print("ENGINE:", engine.url)
    print("Dropping existing tables...")

    print("Tables before drop:", inspect(engine).get_table_names())

    try:
        Base.metadata.drop_all(bind=engine, checkfirst=False)
    except Exception as e:
        print("Failed to drop tables:", e)
        
    print("Creating tables...")
    Base.metadata.create_all(bind=engine, checkfirst=False)

    print("Tables after create:", inspect(engine).get_table_names())
    print("Database reset complete.")

if __name__ == "__main__":
    reset_db()
