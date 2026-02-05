import os

class Config:
    PDF_PATH = "data/2022_Q3_AAPL.pdf"
    CHROMA_DB_PATH = "chroma_db"
    COLLECTION_NAME = "apple_q3_2022"
    CHUNK_SIZE = 1024
    CHUNK_OVERLAP = 200
    
    USE_SEMANTIC_CHUNKING = True
    SEMANTIC_SIMILARITY_THRESHOLD = 0.5
    SEMANTIC_BUFFER_SIZE = 1
    
    EXTRACT_TABLES = True
    EXTRACT_IMAGES = False
    TABLE_STRATEGY = "lattice"
    
    USE_VISION_MODEL = True
    VISION_MODEL = "llava"
    
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    SPARSE_EMBEDDING_MODEL = "bm25"
    
    LLM_MODEL = "mistral"
    LLM_TEMPERATURE = 0.1
    LLM_BASE_URL = "http://localhost:11434"
    
    TOP_K_RETRIEVAL = 5

    USE_HYBRID_SEARCH = True
    DENSE_WEIGHT = 0.7
    SPARSE_WEIGHT = 0.3
    
    USE_RERANKING = True
    RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_TOP_K = 3