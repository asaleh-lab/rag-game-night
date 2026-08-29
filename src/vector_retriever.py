"""Same query through similarity, MMR, and a score threshold."""

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from binder import QUERY, GameStore, binder_docs

load_dotenv()  # OPENAI_API_KEY from .env

# Same embedding model as 02/03. from_documents embeds every sheet, then we query three ways.
store = GameStore.from_documents(
    binder_docs(), OpenAIEmbeddings(model="text-embedding-3-small")
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
