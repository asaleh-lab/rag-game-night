"""LLM rewrites the question, then we union the hits."""

import logging

from dotenv import load_dotenv
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from binder import QUERY, GameStore, binder_docs

load_dotenv()  # OPENAI_API_KEY from .env

store = GameStore.from_documents(
    binder_docs(), OpenAIEmbeddings(model="text-embedding-3-small")
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
