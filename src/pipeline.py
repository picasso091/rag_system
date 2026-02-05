from src.ingestion import DataIngestion
from src.chunking import Chunking
from src.retrieval import Retrieval
from src.generation import LLMGeneration
from config.settings import Config

class RAGPipeline:
    def __init__(self, config: Config = Config()):
        self.config = config
        
        self.ingestion = DataIngestion(
            config.PDF_PATH,
            extract_tables=config.EXTRACT_TABLES,
            extract_images=config.EXTRACT_IMAGES
        )
        
        self.chunking = Chunking(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            use_semantic=config.USE_SEMANTIC_CHUNKING,
            semantic_threshold=config.SEMANTIC_SIMILARITY_THRESHOLD,
            embedding_model=config.EMBEDDING_MODEL
        )
        
        self.retrieval = Retrieval(
            embedding_model=config.EMBEDDING_MODEL,
            db_path=config.CHROMA_DB_PATH,
            use_hybrid=config.USE_HYBRID_SEARCH,
            dense_weight=config.DENSE_WEIGHT,
            sparse_weight=config.SPARSE_WEIGHT,
            use_reranking=config.USE_RERANKING,
            rerank_model=config.RERANK_MODEL
        )
        
        self.generation = LLMGeneration(
            model=config.LLM_MODEL,
            base_url=config.LLM_BASE_URL,
            temperature=config.LLM_TEMPERATURE,
            vision_model=config.VISION_MODEL,
            use_vision=config.USE_VISION_MODEL
        )
        
        self.documents = []
    
    def build(self):
        print("\n" + "="*60)
        print("Building RAG Pipeline")
        print("="*60 + "\n")
        
        print("Step 1: Ingesting documents...")
        docs = self.ingestion.load()
        
        print("\nStep 2: Chunking documents...")
        chunks = self.chunking.chunk(docs)
        
        print("\nStep 3: Building retrieval system...")
        self.retrieval.build(chunks, self.config.COLLECTION_NAME)
        
        self.documents = chunks
        
        print("\n" + "="*60)
        print("RAG Pipeline built successfully!")
        print("="*60 + "\n")
    
    def load(self):
        print("\n" + "="*60)
        print("Loading RAG Pipeline")
        print("="*60 + "\n")
        
        # Load vector store
        self.retrieval.load(self.config.COLLECTION_NAME)
        
        if self.config.USE_HYBRID_SEARCH:
            print("\nRebuilding sparse index for hybrid search...")
            print("Note: For full hybrid search support, rebuild the pipeline")
        
        print("\nPipeline loaded!")
        # self._print_config_summary()
    
    def query(self, question: str) -> dict:
        print(f"\n{'='*60}")
        print(f"Processing Query: {question}")
        print(f"{'='*60}\n")
        
        print("Retrieving relevant documents...")
        docs = self.retrieval.search(question, k=self.config.TOP_K_RETRIEVAL)
        
        print(f"Retrieved {len(docs)} documents.")
        # for i, doc in enumerate(docs, 1):
        #     doc_type = doc.metadata.get("type", "text")
        #     page = doc.metadata.get("page", "N/A")
        #     print(f"  {i}. Type: {doc_type}, Page: {page + 1 if isinstance(page, int) else page}")
        
        print("\nGenerating answer...")
        answer = self.generation.generate(question, docs)
        
        result = {
            "question": question,
            "answer": answer,
            "sources": self._format_sources(docs),
            "config": {
                "hybrid_search": self.config.USE_HYBRID_SEARCH,
                "semantic_chunking": self.config.USE_SEMANTIC_CHUNKING,
                "reranking": self.config.USE_RERANKING,
                "vision_model": self.config.USE_VISION_MODEL
            }
        }
        
        return result
    
    def _format_sources(self, docs) -> list:
        sources = []
        for doc in docs:
            source = {
                "page": doc.metadata.get("page", "N/A"),
                "type": doc.metadata.get("type", "text"),
                "content": doc.page_content[:200]
            }
            
            if doc.metadata.get("type") == "table":
                source["content"] = doc.page_content[:300]  # Show more for tables
                source["accuracy"] = doc.metadata.get("accuracy", "N/A")
            
            sources.append(source)
        
        return sources
    

    def extract_and_save_images(self, output_dir: str = "extracted_images"):
        print(f"\nExtracting images to {output_dir}...")
        self.ingestion.save_images(output_dir)
        print("Images extracted successfully!")
  