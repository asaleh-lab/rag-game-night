"""LLM splits the question into meaning plus a metadata filter."""

import json
from operator import eq, ge, gt, le, lt, ne
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.documents import Document
from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)
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


# The in-memory store takes a callable filter, not Chroma's dict. This visitor
# turns the LLM's structured query into that callable.
COMPARE = {
    Comparator.EQ: eq,
    Comparator.NE: ne,
    Comparator.GT: gt,
    Comparator.GTE: ge,
    Comparator.LT: lt,
    Comparator.LTE: le,
}


class MemoryTranslator(Visitor):
    allowed_operators = [Operator.AND, Operator.OR]
    allowed_comparators = list(COMPARE)

    def visit_comparison(self, comparison: Comparison):
        op = COMPARE[comparison.comparator]
        attr, value = comparison.attribute, comparison.value

        def check(doc: Document) -> bool:
            return op(doc.metadata[attr], value)

        return check

    def visit_operation(self, operation: Operation):
        checks = [arg.accept(self) for arg in operation.arguments]
        if operation.operator is Operator.AND:
            return lambda doc: all(fn(doc) for fn in checks)
        return lambda doc: any(fn(doc) for fn in checks)

    def visit_structured_query(self, structured_query: StructuredQuery):
        if structured_query.filter is None:
            return structured_query.query, {}
        return structured_query.query, {
            "filter": structured_query.filter.accept(self)
        }


store = GameStore.from_documents(
    docs, OpenAIEmbeddings(model="text-embedding-3-small")
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
