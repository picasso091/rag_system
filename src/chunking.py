from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
import re

class Chunking:
    def __init__(
        self, 
        chunk_size: int = 1024, 
        chunk_overlap: int = 200,
        use_semantic: bool = True,
        semantic_threshold: float = 0.5,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_semantic = use_semantic
        
        # Standard recursive splitter
        self.standard_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Semantic chunker
        if use_semantic:
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            self.semantic_splitter = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=semantic_threshold * 100
            )
    
    def chunk(self, documents: List[Document]) -> List[Document]:
        print(f"Chunking {len(documents)} documents...")
        
        all_chunks = []
        
        for doc in documents:
            doc_type = doc.metadata.get("type", "text")
            
            if doc_type == "table":
                chunks = self._chunk_table(doc)
            elif doc_type == "image":
                chunks = [doc]
            else:
                chunks = self._chunk_text(doc)
            
            all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} chunks")
        return all_chunks
    
    def _chunk_text(self, doc: Document) -> List[Document]:
        content = doc.page_content
        
        if self._contains_structured_content(content):
            chunks = self._chunk_with_structure_preservation(doc)
        elif self.use_semantic:
            try:
                chunks = self.semantic_splitter.split_documents([doc])
            except Exception as e:
                print(f"Semantic chunking failed, falling back to standard: {e}")
                chunks = self.standard_splitter.split_documents([doc])
        else:
            chunks = self.standard_splitter.split_documents([doc])
        
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                **doc.metadata,
                "chunk_id": i,
                "total_chunks": len(chunks)
            })
        
        return chunks
    
    def _chunk_table(self, doc: Document) -> List[Document]:
        doc.metadata["chunk_type"] = "table"
        return [doc]
    
    def _contains_structured_content(self, text: str) -> bool:
        table_pattern = r'\|.*\|.*\|'
        image_pattern = r'\[IMAGE_PLACEHOLDER'
        
        return bool(re.search(table_pattern, text) or re.search(image_pattern, text))
    
    def _chunk_with_structure_preservation(self, doc: Document) -> List[Document]:
        content = doc.page_content
        chunks = []
        
        sections = re.split(r'\n\n+', content)
        
        current_chunk = []
        current_size = 0
        
        for section in sections:
            section_size = len(section)
            
            if self._is_table_section(section) or self._is_image_placeholder(section):
                if current_chunk:
                    chunks.append(self._create_chunk(doc, '\n\n'.join(current_chunk)))
                    current_chunk = []
                    current_size = 0
                
                chunks.append(self._create_chunk(doc, section))
            else:
                if current_size + section_size > self.chunk_size and current_chunk:
                    chunks.append(self._create_chunk(doc, '\n\n'.join(current_chunk)))
                    current_chunk = [section]
                    current_size = section_size
                else:
                    current_chunk.append(section)
                    current_size += section_size
        
        if current_chunk:
            chunks.append(self._create_chunk(doc, '\n\n'.join(current_chunk)))
        
        return chunks if chunks else [doc]
    
    def _is_table_section(self, text: str) -> bool:
        return bool(re.search(r'\|.*\|.*\|', text))
    
    def _is_image_placeholder(self, text: str) -> bool:
        return '[IMAGE_PLACEHOLDER' in text
    
    def _create_chunk(self, original_doc: Document, content: str) -> Document:
        return Document(
            page_content=content,
            metadata=original_doc.metadata.copy()
        )
