from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List
import os

class Retrieval:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", db_path: str = "chroma_db"):
        self.db_path = db_path
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = None
    
    def build(self, documents: List[Document], collection_name: str = "documents"):
        print(f"Building vector store with Chroma DB...")
        
        if os.path.exists(self.db_path):
            print(f"Clearing existing database at {self.db_path}")
            import shutil
            shutil.rmtree(self.db_path)
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=self.db_path
        )
        # self.vectorstore.persist()
        print(f"Vector store created with {len(documents)} documents")
    
    def load(self, collection_name: str = "documents"):
        print(f"Loading vector store from {self.db_path}...")
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            collection_name=collection_name,
            persist_directory=self.db_path
        )
        print(f"Vector store loaded")
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        if not self.vectorstore:
            raise ValueError("Vector store not built. Call build() first.")
        
        results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def search_with_scores(self, query: str, k: int = 5):
        if not self.vectorstore:
            raise ValueError("Vector store not built. Call build() first.")
        
        results = self.vectorstore.similarity_search_with_scores(query, k=k)
        return results
