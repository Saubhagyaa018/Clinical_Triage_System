import os

from dotenv import load_dotenv


load_dotenv()


class Settings:
    """Central application configuration."""
    DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:YOUR_PASSWORD@127.0.0.1:5432/clinical_triage"
    )

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    CSV_FILE_PATH: str = "data/raw_patients.csv"

    RETRIEVAL_ALPHA: float = 0.7
    RETRIEVAL_BETA: float = 0.3
    RETRIEVAL_DECAY_LAMBDA: float = 0.05
    

    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    GEMINI_MODEL: str = "gemini-3.5-flash-lite"


settings = Settings()

# models/gemini-2.5-flash
# models/gemini-2.5-pro
# models/gemini-2.5-flash-preview-tts
# models/gemini-2.5-pro-preview-tts
# models/gemma-4-26b-a4b-it
# models/gemma-4-31b-it
# models/gemini-flash-latest
# models/gemini-flash-lite-latest
# models/gemini-pro-latest
# models/gemini-2.5-flash-lite
# models/gemini-2.5-flash-image
# models/gemini-3-flash-preview
# models/gemini-3.1-pro-preview
# models/gemini-3.1-pro-preview-customtools
# models/gemini-3.1-flash-lite-preview
# models/gemini-3.1-flash-lite
# models/gemini-3-pro-image-preview
# models/gemini-3-pro-image
# models/nano-banana-pro-preview
# models/gemini-3.1-flash-image-preview
# models/gemini-3.1-flash-image
# models/gemini-3.1-flash-lite-image
# models/gemini-nano-banana-2.1
# models/gemini-3.5-flash
# models/gemini-3.5-flash-lite
# models/gemini-omni-flash-preview
# models/gemini-omni-1.1-flash
# models/gemini-3.5-transcribe
# models/gemini-3.6-flash
# models/gemini-3.7-flash
# models/gemini-3.8-flash
# models/gemini-4-argon
# models/lyria-3-clip-preview
# models/lyria-3-pro-preview
# models/lyria-3.5
# models/gemini-3.1-flash-tts-preview
# models/gemini-3.8-flash-tts