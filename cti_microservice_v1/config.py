import os

class Config:
    CHROMA_PERSIST_DIR    = os.getenv("CHROMA_PERSIST_DIR", "./vector_store")
    COLLECTION_NAME       = os.getenv("COLLECTION_NAME", "soc_threat_intel")
    EMBEDDING_MODEL_NAME  = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    GROQ_MODEL            = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    GROQ_API_KEY          = os.getenv("GROQ_API_KEY", "")
    RETRIEVER_K           = int(os.getenv("RETRIEVER_K", "8"))
