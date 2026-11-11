def web_search(query, max_results=5):
    """Runs DuckDuckGo text search with safe fallback handling for cloud environments."""
    results = []

    try:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            from ddgs import DDGS

        with DDGS() as ddgs:
            search_results = ddgs.text(
                query,
                max_results=max_results
            )

            if search_results:
                for result in search_results:
                    results.append({
                        "title": result.get("title", ""),
                        "url": result.get("href", result.get("link", "")),
                        "snippet": result.get("body", result.get("snippet", ""))
                    })
    except Exception as error:
        print(f"Web search warning: {error}")

    return results
