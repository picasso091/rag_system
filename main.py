from src.pipeline import RAGPipeline
from src.cli import InteractiveCLI
from config.settings import Config
import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="RAG System for AAPL Q3 2022"
    )
    parser.add_argument(
        '--build',
        action='store_true',
        help='Build the RAG pipeline'
    )
    parser.add_argument(
    '--extract-images',
    action='store_true',
    help='Extract and save images from PDF'
)


    args = parser.parse_args()
    
    config = Config()
    
    print("\nInitializing RAG System...")
    pipeline = RAGPipeline(config)
    
    if args.extract_images:
        pipeline.extract_and_save_images()
        return
    
    if args.build or not os.path.exists(config.CHROMA_DB_PATH):
        pipeline.build()
    else:
        print("\nLoading existing pipeline...")
        try:
            pipeline.load()
        except Exception as e:
            print(f"\nError loading pipeline: {e}")
            print("Building new pipeline instead...\n")
            pipeline.build()
    
    print("\nStarting interactive mode...\n")
    cli = InteractiveCLI(pipeline)
    cli.run()


if __name__ == "__main__":
    main()
