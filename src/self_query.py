"""LLM splits the question into meaning plus a metadata filter."""

from dotenv import load_dotenv
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from binder import QUERY, GameStore, MemoryTranslator, binder_docs

load_dotenv()  # OPENAI_API_KEY from .env

store = GameStore.from_documents(
    binder_docs(), OpenAIEmbeddings(model="text-embedding-3-small")
)

# Names and types the query constructor is allowed to filter on.
# The descriptions are how gpt-4o-mini decides what "short" and "two" mean.
fields = [
    AttributeInfo(
        name="players_min",
        description="Smallest player count. Two people need players_min <= 2.",
        type="integer",
    ),
    AttributeInfo(
        name="players_max",
        description="Largest player count. Two people need players_max >= 2.",
        type="integer",
    ),
    AttributeInfo(
        name="minutes",
        description="Typical play time in minutes. Short after-work games are 45 or under.",
        type="integer",
    ),
    AttributeInfo(
        name="weight",
        description="Rules weight from 1 (light) to 4 (heavy).",
        type="integer",
    ),
    AttributeInfo(
        name="cooperative",
        description="True if players win together, false if they play against each other.",
        type="boolean",
    ),
    AttributeInfo(
        name="on_shelf",
        description="True if the box is on the open shelf tonight.",
        type="boolean",
    ),
]

base = store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# gpt-4o-mini keeps "calm" as the search text and turns "short" / "two" into filters.
selfq = SelfQueryRetriever.from_llm(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    vectorstore=store,
    document_contents="Staff binder notes for board games at The Setup cafe",
    metadata_field_info=fields,
    structured_query_translator=MemoryTranslator(),
    search_kwargs={"k": 3},
)


def show(label, hits):
    print(f"\n{label}")
    for doc in hits:
        meta = doc.metadata
        print(
            f"  {meta['title']}  {meta['players_min']}-{meta['players_max']}p  "
            f"{meta['minutes']} min"
        )


print(f"Query: {QUERY}")
show("Similarity only", base.invoke(QUERY))

# Print the split before the hits so we can see the filter the model built.
parsed = selfq.query_constructor.invoke({"query": QUERY})
print(f"\nMeaning: {parsed.query}")
print(f"Filter: {parsed.filter}")
show("Self-query (meaning + filter)", selfq.invoke(QUERY))
