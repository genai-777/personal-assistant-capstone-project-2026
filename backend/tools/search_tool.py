import os
import requests
from pathlib import Path
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")


@tool
def web_search(query: str) -> str:
    """
    Search the web using Tavily API.
    Returns a list of results with title, URL, and content.
    Always cite sources (title + URL) in your response.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY not set in .env"

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query":   query,
                "max_results": 5,
                "search_depth": "basic",
            },
            timeout=10,
        )
        response.raise_for_status()
        data    = response.json()
        results = data.get("results", [])

        if not results:
            return "No results found."

        out = []
        for i, r in enumerate(results, 1):
            out.append(
                f"{i}. {r.get('title','No title')}\n"
                f"   URL: {r.get('url','')}\n"
                f"   {r.get('content','')[:300]}"
            )
        return "\n\n".join(out)

    except Exception as e:
        return f"Search error: {e}"


def get_search_tool():
    return web_search
