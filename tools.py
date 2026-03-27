import datetime
import requests
from bs4 import BeautifulSoup

# ─── Tools ────────────────────────────────────────────────────────────────────

def scrape_webpage(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # remove noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        
        # trim to avoid token overflow
        text = text[:5000]

        print(f"Scraped {len(text)} characters from {url}")
        return {"content": text, "url": url}

    except Exception as e:
        return {"content": "", "error": str(e)}
    
def calculator(expression):
    try:
        result = eval(expression)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
    
def search_web(query):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = f"https://duckduckgo.com/html/?q={query}"
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for result in soup.find_all("a", class_="result__a", limit=3):
            results.append(result.get_text())
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}
    
def get_current_time():
    return {"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
