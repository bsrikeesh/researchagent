import os
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient

def web_search(query: str, max_results: int = 5) -> list:
    client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_raw_content=False
    )
    results = []
    for r in response.get("results", []):
        results.append({
            "title":   r.get("title", ""),
            "url":     r.get("url", ""),
            "snippet": r.get("content", "")
        })
    return results

def read_url(url: str, max_chars: int = 3000) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=8)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script","style","nav","footer","header","aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 40]
        clean = "\n".join(lines)
        return clean[:max_chars]
    except Exception as e:
        return f"Could not read URL: {str(e)}"
