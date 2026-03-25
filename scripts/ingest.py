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
import time

# Add the project root to the Python path so absolute imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.rag.scraper import scrape_url
from backend.rag.chunker import chunk_text
from backend.rag.database import store_evidence, store_evidence_batch

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
        print(f"⚠️ Failed to scrape URL {url}. Skipping.")
        return

    # Step 3: CHUNK
    chunks = chunk_text(raw_text, chunk_size=500, overlap=50)
    
    # Step 4: EMBED AND SAVE (BATCH PROCESSING)
    # We group chunks into batches of 90 because Cohere limits requests
    # to 1000/month on free keys, but one request can hold 96 chunks!
    batch_size = 90
    batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
    
    print(f"Began sending {len(chunks)} chunks to Qdrant in {len(batches)} batches...")
    
    for batch_idx, batch in enumerate(batches):
        retries = 0
        while True:
            try:
                # Send 90 chunks in a SINGLE API call!
                store_evidence_batch(
                    texts=batch,
                    source_url=url,
                    source_name=source_name,
                    domain_tag=domain_tag
                )
                print(f"[Batch {batch_idx+1}/{len(batches)} ✅] Saved {len(batch)} chunks.")
                break # Exit the while loop if successful
                
            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "too many requests" in error_msg:
                    retries += 1
                    if retries > 3:
                        print(f"\n❌ Hit Rate Limit 3 times in a row. Skipping Batch {batch_idx+1} to avoid infinite loop.")
                        break
                        
                    print(f"\n⚠️ Hit API Rate Limit on Batch {batch_idx+1}.")
                    print(f"Server message: {e}")
                    print(f"Sleeping for 60 seconds (Attempt {retries}/3)...")
                    time.sleep(60)
                    print("Resuming...")
                    continue
                else:
                    # If it's a different error, fail this batch and move on
                    print(f"\n❌ Failed to store Batch {batch_idx+1}: {e}")
                    break
            
    print(f"\n✅ Ingestion complete for {source_name}. Total chunks stored: {len(chunks)}")


# ============================================================
# RUN (Only if executing this file directly)
# ============================================================
if __name__ == "__main__":
    
    urls_file = os.path.join(project_root, "urls.txt")
    
    if not os.path.exists(urls_file):
        print(f"❌ Could not find {urls_file}.")
        sys.exit(1)
        
    print(f"📂 Opening {urls_file}...")
    
    with open(urls_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
        
    for line in lines:
        url = line.strip()
        # Skip empty lines or comments
        if not url or url.startswith("#"):
            continue
            
        # Extract a basic source name from the URL (e.g., en.wikipedia.org)
        domain = url.split("//")[-1].split("/")[0]
        
        ingest_url(
            url=url,
            source_name=f"Web Scraping: {domain}",
            domain_tag=domain
        )
    
    print("\n🎉 ALL URLs INGESTED SUCCESSFULLY!")
