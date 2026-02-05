from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List, Dict, Any
import fitz  # PyMuPDF
import camelot
import os
import base64
from io import BytesIO
from PIL import Image

class DataIngestion:
    def __init__(self, pdf_path: str, extract_tables: bool = True, extract_images: bool = True):
        self.pdf_path = pdf_path
        self.loader = PyPDFLoader(pdf_path)
        self.extract_tables = extract_tables
        self.extract_images = extract_images
    
    def load(self) -> List[Document]:
        print(f"Loading PDF: {self.pdf_path}")
        
        docs = self.loader.load()
        print(f"Loaded {len(docs)} pages")
        
        if self.extract_tables:
            table_docs = self._extract_tables()
            print(f"Extracted {len(table_docs)} tables")
            docs.extend(table_docs)
        
        # Extract images if enabled
        if self.extract_images:
            image_docs = self._extract_images()
            print(f"Extracted {len(image_docs)} images")
            docs.extend(image_docs)
        
        return docs
    
    def _extract_tables(self) -> List[Document]:
        table_docs = []
        
        try:
            # for tables with lines
            tables = camelot.read_pdf(self.pdf_path, pages='all', flavor='lattice')
            
            if len(tables) == 0:
                tables = camelot.read_pdf(self.pdf_path, pages='all', flavor='stream')
            
            for i, table in enumerate(tables):
                # Convert table to markdown format for better readability
                table_text = self._table_to_markdown(table.df)
                
                doc = Document(
                    page_content=table_text,
                    metadata={
                        "source": self.pdf_path,
                        "page": table.page - 1,  # 0-indexed
                        "type": "table",
                        "table_id": i,
                        "accuracy": table.accuracy
                    }
                )
                table_docs.append(doc)
        
        except Exception as e:
            print(f"Warning: Table extraction failed: {e}")
        
        return table_docs
    
    def _table_to_markdown(self, df) -> str:
        markdown = "TABLE:\n"
        markdown += df.to_markdown(index=False)
        return markdown
    
    def _extract_images(self) -> List[Document]:
        image_docs = []
        
        try:
            pdf_document = fitz.open(self.pdf_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                        
                        doc = Document(
                            page_content=f"[IMAGE_PLACEHOLDER_PAGE_{page_num}_IMG_{img_index}]",
                            metadata={
                                "source": self.pdf_path,
                                "page": page_num,
                                "type": "image",
                                "image_id": f"page_{page_num}_img_{img_index}",
                                "image_data": image_base64,
                                "image_ext": base_image["ext"]
                            }
                        )
                        image_docs.append(doc)
                    
                    except Exception as e:
                        print(f"Warning: Could not extract image {img_index} from page {page_num}: {e}")
            
            pdf_document.close()
        
        except Exception as e:
            print(f"Warning: Image extraction failed: {e}")
        
        return image_docs
    
    def save_images(self, output_dir: str = "extracted_images"):
        os.makedirs(output_dir, exist_ok=True)
        
        pdf_document = fitz.open(self.pdf_path)
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    image_path = os.path.join(output_dir, f"page_{page_num}_img_{img_index}.{image_ext}")
                    
                    with open(image_path, "wb") as image_file:
                        image_file.write(image_bytes)
                    
                    print(f"Saved image: {image_path}")
                
                except Exception as e:
                    print(f"Warning: Could not save image {img_index} from page {page_num}: {e}")
        
        pdf_document.close()