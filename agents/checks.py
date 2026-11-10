import os 
from dotenv import load_dotenv 
from google import genai
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") 
client = genai.Client(api_key=api_key)
for model in client.models.list():
    print(model.name)


# from sqlalchemy import create_engine, text
# from sqlalchemy.exc import SQLAlchemyError

# from config import settings


# engine = create_engine(
#     settings.DATABASE_URL,
#     echo=False,
#     pool_pre_ping=True,
# )


# def check_database_connection() -> bool:
#     """Check whether PostgreSQL is reachable."""
#     try:
#         with engine.connect() as connection:
#             connection.execute(text("SELECT 1"))
#         return True
#     except SQLAlchemyError as exc:
#         raise RuntimeError(
#             "Unable to connect to the configured database."
#         ) from exc


# if __name__ == "__main__":
#     try:
#         check_database_connection()
#         print("Database connection successful.")
#     except RuntimeError as exc:
#         print(f"Database connection failed: {exc}")
