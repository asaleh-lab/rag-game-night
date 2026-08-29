"""The Setup binder, plus the two classes the LangChain scripts share."""

import json
from operator import eq, ge, gt, le, lt, ne
from pathlib import Path

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

QUERY = "a short calm game for two people after work"


def load_games():
    return json.loads(Path("data/games.json").read_text(encoding="utf-8"))


def binder_docs():
    """Title plus how_it_plays is what we embed. Metadata rides along for filters."""
    return [
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
                "pairs_with": game["pairs_with"],
            },
        )
        for game in load_games()
    ]


class GameStore(InMemoryVectorStore):
    """In-memory store already returns cosine similarity; the retriever needs a 0 to 1 fn."""

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
