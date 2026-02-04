from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

class Chunking:
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def chunk(self, documents: List[Document]) -> List[Document]:
        print(f"Chunking {len(documents)} documents...")
        chunks = self.splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        return chunks
