"""Same binder vectors in Chroma and in FAISS."""

import chromadb
import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from binder import QUERY, binder_docs

load_dotenv()  # OPENAI_API_KEY from .env

docs = binder_docs()
embed = OpenAIEmbeddings(model="text-embedding-3-small")
texts = [doc.page_content for doc in docs]
# One embedding pass. Both stores get these vectors, not a second API call.
vectors = np.array(embed.embed_documents(texts), dtype="float32")
query = np.array([embed.embed_query(QUERY)], dtype="float32")
dim = vectors.shape[1]

print(f"Query: {QUERY}")
print(f"{len(docs)} sheets, dim {dim}")

COLLECTION = "setup_binder"
client = chromadb.PersistentClient(path="chroma_data")
if COLLECTION in [c.name for c in client.list_collections()]:
    client.delete_collection(COLLECTION)  # start clean on reruns

# No embedding function: we pass the vectors we already have. L2 matches IndexFlat.
collection = client.create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "l2"},
)
collection.add(
    ids=[doc.metadata["id"] for doc in docs],
    documents=texts,
    embeddings=vectors.tolist(),
    metadatas=[{"title": doc.metadata["title"]} for doc in docs],
)

print("\nChroma")
hits = collection.query(query_embeddings=query.tolist(), n_results=3)
for meta, dist in zip(hits["metadatas"][0], hits["distances"][0]):
    print(f"  {dist:.4f}  {meta['title']}")

# Same matrix, brute-force L2. Titles should match Chroma on this small shelf.
flat = faiss.IndexFlatL2(dim)
flat.add(vectors)
print("\nFAISS IndexFlat")
distances, ids = flat.search(query, 3)
for dist, i in zip(distances[0], ids[0]):
    print(f"  {dist:.4f}  {docs[int(i)].metadata['title']}")
