from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Tuple
import os
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import CrossEncoder

class Retrieval:
    def __init__(
        self, 
        embedding_model: str = "all-MiniLM-L6-v2", 
        db_path: str = "chroma_db",
        use_hybrid: bool = True,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        use_reranking: bool = True,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.db_path = db_path
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = None
        self.use_hybrid = use_hybrid
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.use_reranking = use_reranking
        
        # BM25 for sparse retrieval
        self.bm25 = None
        self.documents = []
        self.tokenized_corpus = []
        
        # Cross-encoder for reranking
        if use_reranking:
            print(f"Loading reranking model: {rerank_model}")
            self.reranker = CrossEncoder(rerank_model)
        else:
            self.reranker = None
    
    def build(self, documents: List[Document], collection_name: str = "documents"):
        print(f"Building vector store with Chroma DB...")
        
        if os.path.exists(self.db_path):
            print(f"Clearing existing database at {self.db_path}")
            import shutil
            shutil.rmtree(self.db_path)
        
        # Build dense vector store (Chroma)
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=self.db_path
        )
        
        # Build sparse index (BM25) for hybrid search
        if self.use_hybrid:
            print("Building BM25 index for hybrid search...")
            self.documents = documents
            self.tokenized_corpus = [doc.page_content.lower().split() for doc in documents]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            print(f"BM25 index created with {len(documents)} documents")
        
        print(f"Vector store created with {len(documents)} documents")
    
    def load(self, collection_name: str = "documents"):
        print(f"Loading vector store from {self.db_path}...")
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            collection_name=collection_name,
            persist_directory=self.db_path
        )
        
        print(f"Vector store loaded")
        print("Warning: BM25 index needs to be rebuilt for hybrid search")
    
    def rebuild_bm25(self, documents: List[Document]):
        if self.use_hybrid:
            print("Rebuilding BM25 index...")
            self.documents = documents
            self.tokenized_corpus = [doc.page_content.lower().split() for doc in documents]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            print(f"BM25 index rebuilt with {len(documents)} documents")
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        if not self.vectorstore:
            raise ValueError("Vector store not built. Call build() first.")
        
        if self.use_hybrid and self.bm25:
            results = self._hybrid_search(query, k)
        else:
            results = self.vectorstore.similarity_search(query, k=k)
        
        if self.use_reranking and self.reranker and len(results) > 0:
            results = self._rerank_results(query, results, k)
        
        return results
    
    def search_with_scores(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        if not self.vectorstore:
            raise ValueError("Vector store not built. Call build() first.")
        
        if self.use_hybrid and self.bm25:
            results = self._hybrid_search_with_scores(query, k)
        else:
            results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k)
        
        if self.use_reranking and self.reranker and len(results) > 0:
            docs = [doc for doc, score in results]
            reranked_docs = self._rerank_results(query, docs, k)
            results = [(doc, 0.0) for doc in reranked_docs]  # Placeholder scores
        
        return results
    
    def _hybrid_search(self, query: str, k: int = 5) -> List[Document]:
        # Dense retrieval (Chroma/embeddings)
        dense_results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k*2)
        
        # Sparse retrieval (BM25)
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Get top k from BM25
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:k*2]
        sparse_results = [(self.documents[i], bm25_scores[i]) for i in top_bm25_indices]
        
        # Normalize scores
        dense_scores_norm = self._normalize_scores([score for _, score in dense_results])
        sparse_scores_norm = self._normalize_scores([score for _, score in sparse_results])
        
        # Create score dictionary for all documents
        doc_scores = {}
        
        # Add dense scores
        for i, (doc, _) in enumerate(dense_results):
            doc_id = self._get_doc_id(doc)
            doc_scores[doc_id] = {
                'doc': doc,
                'dense': dense_scores_norm[i] * self.dense_weight,
                'sparse': 0.0
            }
        
        # Add sparse scores
        for i, (doc, _) in enumerate(sparse_results):
            doc_id = self._get_doc_id(doc)
            if doc_id in doc_scores:
                doc_scores[doc_id]['sparse'] = sparse_scores_norm[i] * self.sparse_weight
            else:
                doc_scores[doc_id] = {
                    'doc': doc,
                    'dense': 0.0,
                    'sparse': sparse_scores_norm[i] * self.sparse_weight
                }
        
        # Calculate combined scores
        combined_results = []
        for doc_id, scores in doc_scores.items():
            total_score = scores['dense'] + scores['sparse']
            combined_results.append((scores['doc'], total_score))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k documents
        return [doc for doc, score in combined_results[:k]]
    
    def _hybrid_search_with_scores(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        # Dense retrieval
        dense_results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k*2)
        
        # Sparse retrieval
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:k*2]
        sparse_results = [(self.documents[i], bm25_scores[i]) for i in top_bm25_indices]
        
        # Normalize and combine
        dense_scores_norm = self._normalize_scores([score for _, score in dense_results])
        sparse_scores_norm = self._normalize_scores([score for _, score in sparse_results])
        
        doc_scores = {}
        
        for i, (doc, _) in enumerate(dense_results):
            doc_id = self._get_doc_id(doc)
            doc_scores[doc_id] = {
                'doc': doc,
                'score': dense_scores_norm[i] * self.dense_weight
            }
        
        for i, (doc, _) in enumerate(sparse_results):
            doc_id = self._get_doc_id(doc)
            sparse_contrib = sparse_scores_norm[i] * self.sparse_weight
            if doc_id in doc_scores:
                doc_scores[doc_id]['score'] += sparse_contrib
            else:
                doc_scores[doc_id] = {'doc': doc, 'score': sparse_contrib}
        
        # Sort and return
        results = [(info['doc'], info['score']) for info in doc_scores.values()]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:k]
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range"""
        if not scores:
            return []
        
        scores = np.array(scores)
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score - min_score == 0:
            return [1.0] * len(scores)
        
        return ((scores - min_score) / (max_score - min_score)).tolist()
    
    def _get_doc_id(self, doc: Document) -> str:
        return f"{hash(doc.page_content)}_{doc.metadata.get('page', 0)}_{doc.metadata.get('type', 'text')}"
    
    def _rerank_results(self, query: str, documents: List[Document], k: int) -> List[Document]:
        if not documents:
            return documents
                
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.reranker.predict(pairs)
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        reranked_docs = [doc for doc, score in doc_score_pairs[:k]]
        
        print(f"Reranking complete.")
        
        return reranked_docs
    
    def get_all_documents(self) -> List[Document]:
        if not self.vectorstore:
            raise ValueError("Vector store not built. Call build() first.")
        
        return self.documents if self.documents else []