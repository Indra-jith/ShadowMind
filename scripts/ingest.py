"""
shadowmind/scripts/ingest.py

Responsibility: The glue that connects scraping, chunking, embedding, and saving.
WHY THIS FILE EXISTS: This is the actual engine we run to feed ShadowsMind evidence
before an investigation starts.

Run this script directly from the terminal:
> python scripts/ingest.py
"""

import sys
import os

# Add the project root to the Python path so absolute imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.rag.scraper import scrape_url
from backend.rag.chunker import chunk_text
from backend.rag.database import store_evidence

def ingest_url(url: str, source_name: str, domain_tag: str):
    """
    1. Downloads URL
    2. Cleans HTML
    3. Splits text into 500-character chunks
    4. Loops through chunks, turns them into vectors (via Cohere), and saves to Qdrant
    """
    
    print(f"\n--- INGESTING: {source_name} ---")
    
    # Step 1 & 2: PULL AND CLEAN
    raw_text = scrape_url(url)
    if not raw_text:
        print("Failed to scrape URL. Aborting.")
        return

    # Step 3: CHUNK
    chunks = chunk_text(raw_text, chunk_size=500, overlap=50)
    
    # Step 4: EMBED AND SAVE
    print(f"Began sending {len(chunks)} chunks to Qdrant...")
    
    for i, chunk_text_data in enumerate(chunks):
        try:
            # We call store_evidence, which handles both embedding generation and Qdrant DB saving.
            store_evidence(
                text=chunk_text_data,
                source_url=url,
                source_name=source_name,
                domain_tag=domain_tag
            )
            # Print a dot for every successful chunk so we can watch progress
            print(".", end="", flush=True)
            
        except Exception as e:
            print(f"\n❌ Failed to store chunk {i+1}: {e}")
            
    print(f"\n✅ Ingestion complete for {source_name}. Total chunks stored: {len(chunks)}")


# ============================================================
# RUN (Only if executing this file directly)
# ============================================================
if __name__ == "__main__":
    # Feel free to change this URL to any mystery/case file you want!
    target = "https://en.wikipedia.org/wiki/Dyatlov_Pass_incident"
    
    ingest_url(
        url=target,
        source_name="Wikipedia: Dyatlov Pass Incident",
        domain_tag="wikipedia"
    )
