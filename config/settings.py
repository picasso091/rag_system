import os

class Config:
    PDF_PATH = "data/2022_Q3_AAPL.pdf"
    CHROMA_DB_PATH = "chroma_db"
    COLLECTION_NAME = "apple_q3_2022"
    
    CHUNK_SIZE = 1024
    CHUNK_OVERLAP = 200
    
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "mistral"
    LLM_TEMPERATURE = 0.1
    LLM_BASE_URL = "http://localhost:11434"
    
    TOP_K_RETRIEVAL = 5
