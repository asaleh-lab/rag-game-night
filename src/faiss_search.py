"""Index the binder in FAISS, then search with no LLM."""

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from binder import QUERY, binder_docs

load_dotenv()  # OPENAI_API_KEY from .env

docs = binder_docs()
embed = OpenAIEmbeddings(model="text-embedding-3-small")
# Same model as the LangChain scripts. FAISS only stores the vectors.
vectors = np.array(
    embed.embed_documents([doc.page_content for doc in docs]), dtype="float32"
)
query = np.array([embed.embed_query(QUERY)], dtype="float32")
dim = vectors.shape[1]


def show(label, index):
    distances, ids = index.search(query, 3)
    print(f"\n{label}")
    for dist, i in zip(distances[0], ids[0]):
        print(f"  {dist:.4f}  {docs[int(i)].metadata['title']}")


print(f"Query: {QUERY}")
print(f"{len(docs)} sheets, dim {dim}")

# Brute-force: every query against every sheet. Exact L2.
flat = faiss.IndexFlatL2(dim)
flat.add(vectors)
show("IndexFlat (brute-force)", flat)

# HNSW: a neighbour graph. Same vectors, approximate search.
hnsw = faiss.IndexHNSWFlat(dim, 32)
hnsw.add(vectors)
show("IndexHNSWFlat", hnsw)
