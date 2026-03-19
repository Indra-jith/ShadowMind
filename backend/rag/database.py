"""
shadowmind/backend/rag/qdrant_client.py

Responsibility: Single point of contact with Qdrant Cloud.
Creates the shadowmind-evidence collection, saves text + vectors into the database,
and performs similarity searches.

Used by: Node 4 (retrieve)
Depends on: qdrant-client, backend/rag/embeddings.py, Pydantic models
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import uuid
from dotenv import load_dotenv

# QdrantClient — handles the HTTP connection to our vector database
# models (from qdrant_client) — structured data types required by Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models

import sys
# Add the project root to the Python path so absolute imports work
# when running this file directly for testing.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Our custom embedding generator we just built
from backend.rag.embeddings import generate_embedding

# We'll use this later to parse the results back into our Pydantic model
from backend.models.investigation import EvidenceChunk

# Load environment variables from the project root
env_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=env_path)

# ============================================================
# CONSTANTS
# ============================================================

# The name of our "table" in Qdrant
COLLECTION_NAME = "shadowmind_evidence"

# Cohere's embed-english-v3.0 outputs exactly 1024 numbers
VECTOR_SIZE = 1024


# ============================================================
# SETUP
# ============================================================

# Create the global Qdrant client connection (Connection Reuse!)
# Notice we pass both URL and API KEY — this connects us to Qdrant CLOUD,
# not a local instance. This means our data survives server restarts.
try:
    q_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        # Timeout is important — cloud calls can be slow
        timeout=10.0
    )
except Exception as e:
    print(f"⚠️ Warning: Qdrant client failed to initialize: {e}")
    q_client = None


def init_collection():
    """
    Creates the collection (table) in Qdrant if it doesn't already exist.
    
    WHY: You can't insert data into a collection that hasn't been created.
    When we start the server for the very first time, this function sets up
    the database schema.
    """
    if not q_client:
        return

    # Ask Qdrant: "Do you already have a collection with this name?"
    if q_client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        return

    print(f"Creating new collection: {COLLECTION_NAME}")
    
    # Create the collection
    q_client.create_collection(
        collection_name=COLLECTION_NAME,
        # Vectors config tells Qdrant what shape of data to expect
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,           # 768 dimensions for Gemini
            distance=models.Distance.COSINE  # How to calculate similarity
        )
    )
    print("Collection created successfully.")


# ============================================================
# CORE FUNCTIONS
# ============================================================

def store_evidence(text: str, source_url: str, source_name: str, domain_tag: str):
    """
    Takes plain text, converts it to a vector, and saves BOTH the vector
    and the text in Qdrant.
    
    WHY: This is how we build our knowledge base. Later (Week 3/4), scraper
    scripts will call this function thousands of times to fill the DB.
    """
    
    if not q_client:
        raise ValueError("Qdrant client not configured.")

    # Step 1: Convert the English text into a vector (768 numbers)
    # This is the "magic" that captures the meaning of the text.
    vector = generate_embedding(text)

    # Step 2: Generate a unique ID for this row in the database
    # uuid4 generates something like: "f47ac10b-58cc-4372-a567-0e02b2c3d479"
    point_id = str(uuid.uuid4())

    # Step 3: Send it to Qdrant
    q_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=point_id,      # The unique ID
                vector=vector,    # The 768 numbers (used for searching)
                payload={         # The actual readable data (returned in search results)
                    "text": text,
                    "source_url": source_url,
                    "source_name": source_name,
                    "domain_tag": domain_tag
                }
            )
        ]
    )
    
    return point_id


def search_evidence(query: str, limit: int = 3):
    """
    Takes a search query (e.g., "weather at dyatlov pass"), converts the query
    into a vector, and asks Qdrant to find the closest matching vectors in the DB.
    
    WHY: This is how ShadowMind finds evidence to prove/disprove hypotheses.
    """
    if not q_client:
        raise ValueError("Qdrant client not configured.")

    # Convert the search query into a vector so we can compare it
    query_vector = generate_embedding(query)

    # Ask Qdrant to find the closest matches using Cosine Similarity
    results = q_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=limit,                 # Return top N results
        with_payload=True            # Return the text and metadata, not just IDs
    )
    
    # We will format these results into Pydantic EvidenceChunk models later
    return results


# ============================================================
# TEST BLOCK
# ============================================================
if __name__ == "__main__":
    print("Testing Qdrant Connection...")
    init_collection()
    
    try:
        # Check connection by fetching collection info
        info = q_client.get_collection(COLLECTION_NAME)
        print("✅ Success! Qdrant is connected.")
        print(f"Collection '{COLLECTION_NAME}' has {info.points_count} items stored.")
    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {e}")
