"""
shadowmind/backend/rag/scraper.py

Responsibility: Extracts clean, readable text from a given web URL.
WHY THIS FILE EXISTS: Evidence lives on the internet (Wikipedia, news articles, etc).
When we download a webpage, 90% of it is garbage (HTML tags, Javascript, navbars, ads).
This script downloads a URL and strips away everything except the actual readable text.

Depends on: requests, beautifulsoup4
"""

import requests
from bs4 import BeautifulSoup

def scrape_url(url: str) -> str:
    """
    Downloads a webpage and extracts only the main readable text.
    
    WHY: If we feed raw HTML into our vector database, it fills the database
    with noise like "<div>" and "{margin: 0;}", which ruins our search accuracy.
    """
    
    print(f"Scraping {url}...")
    
    # We use a User-Agent so websites don't instantly block us thinking we are a malicious bot
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ShadowMind/1.0"
    }
    
    try:
        # Step 1: Download the raw HTML
        response = requests.get(url, headers=headers, timeout=10)
        
        # If the website returns a 404 (Not Found) or 403 (Forbidden), this raises an error
        response.raise_for_status() 
        html_content = response.text
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to download {url}: {e}")
        return ""

    # Step 2: Feed the raw HTML into BeautifulSoup to make it searchable
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Step 3: Remove garbage elements (scripts, styling, headers, navbars)
    for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
        element.decompose() # .decompose() physically deletes the element from the soup
        
    # Step 4: Extract the text from the remaining HTML
    # .get_text(separator="\n", strip=True) grabs the text and separates paragraphs with a newline
    clean_text = soup.get_text(separator="\n", strip=True)
    
    # Step 5: Final cleanup (remove massive chunks of blank space)
    lines = (line.strip() for line in clean_text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    clean_text = '\n'.join(chunk for chunk in chunks if chunk)
    
    print(f"✅ Successfully extracted {len(clean_text)} characters of clean text.")
    return clean_text

# ============================================================
# TEST BLOCK
# ============================================================
if __name__ == "__main__":
    # Test on a Wikipedia article
    test_url = "https://en.wikipedia.org/wiki/Dyatlov_Pass_incident"
    text = scrape_url(test_url)
    
    # Print the first 500 characters to verify it looks like English, not code
    print("\n--- SNEAK PEEK ---")
    print(text[:500] + "...\n")
