from src.ingestion import DataIngestion
from src.chunking import Chunking
from src.retrieval import Retrieval
from src.generation import LLMGeneration
from config.settings import Config

class RAGPipeline:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.ingestion = DataIngestion(config.PDF_PATH)
        self.chunking = Chunking(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        self.retrieval = Retrieval(config.EMBEDDING_MODEL, config.CHROMA_DB_PATH)
        self.generation = LLMGeneration(config.LLM_MODEL, config.LLM_BASE_URL, config.LLM_TEMPERATURE)
    
    def build(self):
        print("\n" + "="*60)
        print("Building RAG Pipeline")
        print("="*60 + "\n")
        
        docs = self.ingestion.load()
        chunks = self.chunking.chunk(docs)
        self.retrieval.build(chunks, self.config.COLLECTION_NAME)
        
        print("\nRAG Pipeline built successfully!\n")
    
    def load(self):
        print("\nLoading existing RAG Pipeline...\n")
        self.retrieval.load(self.config.COLLECTION_NAME)
        print("Pipeline loaded!\n")
    
    def query(self, question: str) -> dict:
        print(f"\nQuestion: {question}\n")
        
        docs = self.retrieval.search(question, k=self.config.TOP_K_RETRIEVAL)
        answer = self.generation.generate(question, docs)
        
        return {
            "question": question,
            "answer": answer,
            "sources": [{"content": doc.page_content[:200], "page": doc.metadata.get("page", "N/A")} for doc in docs]
        }
