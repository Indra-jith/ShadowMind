"""
shadowmind/backend/rag/chunker.py

Responsibility: Splits a massive string of text into smaller, bite-sized "chunks".
WHY THIS FILE EXISTS: If you feed an entire 10,000-word Wikipedia article into
an embedding model, the model gets overwhelmed and the meaning becomes muddy (a vector
can't capture 10,000 words perfectly). We chop the text into ~200-word chunks first.
"""

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Splits text into chunks of `chunk_size` characters, with `overlap` characters
    overlapping between chunks.
    
    WHY OVERLAP? If a sentence starts at the end of Chunk 1 and finishes in Chunk 2,
    the model loses the context. An overlap ensures no sentence is completely orphaned.
    
    Note: For a production app, we would use a more advanced chunker (like LangChain's
    RecursiveCharacterTextSplitter) that smartly splits on paragraphs and periods.
    This is a basic string chunker for Day 1 simplicity.
    """
    
    print(f"Chunking {len(text)} characters of text...")
    
    if not text:
        return []
        
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        # Find the end of this chunk
        end = start + chunk_size
        
        # If this isn't the last chunk, try to avoid cutting a word in half
        if end < text_length:
            # Look backwards from 'end' to find the closest space character
            # so we cut the chunk cleanly between words.
            while end > start and text[end] != ' ' and text[end] != '\n':
                end -= 1
                
            # If we couldn't find a space (weird edge case), just cut it hard.
            if end == start:
                end = start + chunk_size
                
        # Slice the string and add it to our list
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        # Move our starting point forward, but slide it backward by `overlap`
        # so the next chunk shares some words with the end of this chunk.
        start = end - overlap
        
    print(f"✅ Text split into {len(chunks)} overlapping chunks.")
    return chunks

# ============================================================
# TEST BLOCK
# ============================================================
if __name__ == "__main__":
    test_text = "The Dyatlov Pass incident was an event in which nine Soviet trekkers died in the northern Ural Mountains between 1 and 2 February 1959. " * 10
    chunks = chunk_text(test_text, chunk_size=100, overlap=20)
    
    for i, c in enumerate(chunks[:3]):  # Just print first 3
        print(f"\n--- Chunk {i+1} ---")
        print(c)
