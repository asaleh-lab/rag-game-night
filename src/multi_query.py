"""LLM rewrites the question, then we union the hits."""

import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()  # OPENAI_API_KEY from .env

# Same binder and question as vector_retriever.py. Only the retriever changes.
QUERY = "a short calm game for two people after work"
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
    """Same store as vector_retriever.py. Cosine is already 0 to 1."""

    def _select_relevance_score_fn(self):
        return lambda score: score


store = GameStore.from_documents(
    docs, OpenAIEmbeddings(model="text-embedding-3-small")
)

# Plain nearest-3, so we can see what one phrasing misses.
base = store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# gpt-4o-mini writes three other phrasings. include_original keeps our question in the union.
multi = MultiQueryRetriever.from_llm(
    retriever=base,
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    include_original=True,
)

# Print the rewrites LangChain logs, then the unique titles.
mq_log = logging.getLogger("langchain.retrievers.multi_query")
mq_log.setLevel(logging.INFO)
mq_log.addHandler(logging.StreamHandler())
mq_log.propagate = False


def show(label, retriever):
    print(f"\n{label}")
    for doc in retriever.invoke(QUERY):
        print(f"  {doc.metadata['title']}")


print(f"Query: {QUERY}")
show("Similarity only", base)
show("Multi-query (rewrites, then union)", multi)
