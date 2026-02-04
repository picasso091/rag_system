from src.pipeline import RAGPipeline
from src.cli import InteractiveCLI
from config.settings import Config
import os
import sys

def main():
    config = Config()
    
    if not os.path.exists(config.PDF_PATH):
        print(f"PDF not found at {config.PDF_PATH}")
        sys.exit(1)
    
    pipeline = RAGPipeline(config)
    
    if os.path.exists(config.CHROMA_DB_PATH) and os.listdir(config.CHROMA_DB_PATH):
        pipeline.load()
    else:
        pipeline.build()
    
    cli = InteractiveCLI(pipeline)
    cli.run()

if __name__ == "__main__":
    main()
