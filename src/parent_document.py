"""Embed small chunks, then hand back the parent binder page."""

from dotenv import load_dotenv
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from binder import QUERY, binder_docs, load_games

load_dotenv()  # OPENAI_API_KEY from .env

docs = binder_docs()

# Each sheet is two paragraphs, ~280 to 400 chars. 250 cuts on the blank line.
child_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)

# Children go in the vector store. Parents sit in a separate key-value store.
child_store = InMemoryVectorStore(OpenAIEmbeddings(model="text-embedding-3-small"))
parent_store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=child_store,
    docstore=parent_store,
    child_splitter=child_splitter,
    search_kwargs={"k": 3},
)
# Keep the game id as the parent key so two child hits from one sheet collapse.
retriever.add_documents(docs, ids=[game["id"] for game in load_games()])

print(f"Query: {QUERY}")
print(f"{len(docs)} parent sheets")

# The vector store only sees the small chunks. We print those first.
print("\nChild chunks (what we embed)")
for doc in child_store.similarity_search(QUERY, k=3):
    preview = doc.page_content.replace("\n", " ")[:80]
    print(f"  {doc.metadata['title']}  {len(doc.page_content)} chars")
    print(f"    {preview}")

# Same search. The retriever looks up each child parent and returns the full page.
print("\nParent sheets (what we return)")
for doc in retriever.invoke(QUERY):
    print(f"  {doc.metadata['title']}  {len(doc.page_content)} chars")
