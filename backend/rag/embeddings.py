"""
shadowmind/backend/rag/embeddings.py

Responsibility: Converts text into high-dimensional vector embeddings using Cohere's API.
WHY THIS FILE EXISTS: Qdrant doesn't understand English. It only understands numbers.
Before we can save evidence to the database, we have to turn it into a list
of 1024 numbers (a vector) that represents its meaning.

Used by: backend/rag/database.py (when upserting or searching)
Depends on: cohere
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import cohere
from dotenv import load_dotenv

# Load environment variables from the project root .env file
env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=env_path)

# ============================================================
# SETUP
# ============================================================

# Create the Cohere client ONCE when the module loads.
try:
    cohere_key = os.getenv("COHERE_API_KEY")
    if not cohere_key:
        raise ValueError("COHERE_API_KEY not found in .env file")
    
    # Initialize the Cohere client
    co_client = cohere.Client(api_key=cohere_key)
except Exception as e:
    print(f"⚠️ Warning: Cohere client failed to initialize: {e}")
    co_client = None

# ============================================================
# FUNCTIONS
# ============================================================

def generate_embedding(text: str) -> list[float]:
    """
    Takes a string of text and returns a list of 1024 float numbers.

    WHY: This translates human language into mathematical representation
    so the vector database can calculate similarity between texts.
    Cohere's embed-english-v3.0 is one of the best models on the market
    and outputs exactly 1024 dimensions.
    """
    
    if not text.strip():
        # Edge case: if we pass empty text, return a vector of zeros
        return [0.0] * 1024

    if not co_client:
        raise ValueError("Cannot generate embedding: COHERE_API_KEY is missing or invalid.")

    # ⚠️ COST WARNING: This call uses Cohere API
    # Free tier limit: 1000 Requests Per Minute (very generous!)
    
    response = co_client.embed(
        texts=[text],
        model="embed-english-v3.0",
        input_type="search_document"
    )
    
    # Extract the actual list of numbers from the response object
    vector = response.embeddings[0]
    
    return vector

# ============================================================
# TEST BLOCK
# ============================================================
if __name__ == "__main__":
    print("Testing Cohere Embeddings...")
    sample_text = "ShadowMind is an autonomous investigator."
    
    try:
        vec = generate_embedding(sample_text)
        print(f"✅ Success! Text turned into {len(vec)} numbers.")
        print(f"First 5 numbers: {vec[:5]}")
    except Exception as e:
        print(f"❌ Error: {e}")
