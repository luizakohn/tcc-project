import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
POSTGRES_DSN = os.getenv("POSTGRES_DSN")

DIMENSIONS = [768, 1024, 1536]

BASE_NAMES = {
    768: "base_768d",
    1024: "base_1024d",
    1536: "base_1536d",
}

EMBEDDING_MODEL = "text-embedding-3-small"

K = 25
