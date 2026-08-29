"""Same query through similarity, MMR, and a score threshold."""

import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()  # OPENAI_API_KEY from .env

QUERY = "a short calm game for two people after work"
# The binder: title + how_it_plays is what we embed. Metadata rides along for later scripts.
games = json.loads(Path("data/games.json").read_text(encoding="utf-8"))

docs = [
    Document(
        page_content=f"{game['title']}. {game['how_it_plays']}",
        metadata={
            "id": game["id"],
            "title": game["title"],
            "players_min": game["players_min"],
            "players_max": game["players_max"],
            "minutes": game["minutes"],
            "weight": game["weight"],
            "cooperative": game["cooperative"],
            "on_shelf": game["on_shelf"],
        },
    )
    for game in games
]


class GameStore(InMemoryVectorStore):
    """In-memory store already returns cosine similarity; the retriever needs a 0 to 1 fn."""

    def _select_relevance_score_fn(self):
        return lambda score: score


# Same embedding model as 02/03. from_documents embeds every sheet, then we query three ways.
store = GameStore.from_documents(
    docs, OpenAIEmbeddings(model="text-embedding-3-small")
)


def show(label, retriever):
    print(f"\n{label}")
    hits = retriever.invoke(QUERY)
    if not hits:
        print("  (no hits)")
        return
    for doc in hits:
        print(f"  {doc.metadata['title']}")


print(f"Query: {QUERY}")
# Nearest neighbours only. k=3, no extra ranking.
show(
    "Similarity",
    store.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
)
# Fetch 8, then keep 3 that are close and not copies of each other. lambda_mult 0.3 leans diverse.
show(
    "MMR",
    store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 8, "lambda_mult": 0.3},
    ),
)
# Drop anything under 0.4 cosine. k is a cap, not a promise of 8 titles.
show(
    "Score threshold",
    store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 8, "score_threshold": 0.4},
    ),
)
