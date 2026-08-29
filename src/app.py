"""The Setup binder behind Gradio: FAISS plus multi-query."""

import os

import gradio as gr
from dotenv import load_dotenv
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from binder import QUERY, binder_docs

load_dotenv()  # OPENAI_API_KEY and OPENAI_MODEL from .env

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SYSTEM = (
    "Answer from these games only. If none of them fit, say so. "
    "Name the game you would put on the table."
)

# Same binder, now in FAISS. Multi-query rewrites the question, then we union the hits.
store = FAISS.from_documents(
    binder_docs(), OpenAIEmbeddings(model="text-embedding-3-small")
)
retriever = MultiQueryRetriever.from_llm(
    retriever=store.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
    llm=ChatOpenAI(model=MODEL, temperature=0),
    include_original=True,
)
llm = ChatOpenAI(model=MODEL, temperature=0)


def chat(message, _history):
    hits = retriever.invoke(message)
    if not hits:
        return "None of the games on the shelf fit that."
    context = "\n\n".join(doc.page_content for doc in hits)
    answer = llm.invoke(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Games:\n{context}\n\nNeed: {message}"},
        ]
    ).content
    titles = ", ".join(dict.fromkeys(doc.metadata["title"] for doc in hits))
    return f"{answer}\n\nHits: {titles}"


if __name__ == "__main__":
    gr.ChatInterface(
        chat,
        type="messages",
        title="The Setup",
        examples=[
            QUERY,
            "a loud party game for a crowd",
        ],
    ).launch()
