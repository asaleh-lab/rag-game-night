"""Same binder through six LlamaIndex retrievers."""

from dotenv import load_dotenv
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.indices.document_summary import DocumentSummaryIndex
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import (
    AutoMergingRetriever,
    QueryFusionRetriever,
    RecursiveRetriever,
)
from llama_index.core.schema import IndexNode, TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.retrievers.bm25 import BM25Retriever

from binder import QUERY, load_games

load_dotenv()  # OPENAI_API_KEY from .env

# Same models as the LangChain scripts. Settings is LlamaIndex's global config.
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

games = load_games()
docs = [
    Document(
        text=f"{game['title']}. {game['how_it_plays']}",
        doc_id=game["id"],
        metadata={"title": game["title"], "pairs_with": game["pairs_with"]},
    )
    for game in games
]
# One node per sheet so BM25, fusion, and recursive share the same units.
sheets = [
    TextNode(text=doc.text, id_=doc.doc_id, metadata=doc.metadata) for doc in docs
]


def show(label, hits):
    print(f"\n{label}")
    seen = set()
    for hit in hits:
        title = hit.node.metadata.get("title", hit.node.node_id)
        if title in seen:
            continue
        seen.add(title)
        print(f"  {title}  {len(hit.node.get_content())} chars")


print(f"Query: {QUERY}")

# Vector index: nearest neighbours, same idea as vector_retriever.py.
vector_index = VectorStoreIndex(sheets)
vector = vector_index.as_retriever(similarity_top_k=3)
show("Vector index", vector.retrieve(QUERY))

# BM25: keyword scores on the same nodes, no embeddings.
bm25 = BM25Retriever.from_defaults(nodes=sheets, similarity_top_k=3)
show("BM25", bm25.retrieve(QUERY))

# Document-summary: LLM writes a short card per sheet, then we embed those cards.
print("\nSummarising each sheet once...")
summary_index = DocumentSummaryIndex.from_documents(
    docs,
    transformations=[],
    summary_query="One sentence: the title, how the table feels, and who it is for.",
)
summary = summary_index.as_retriever(retriever_mode="embedding", similarity_top_k=3)
show("Document-summary index", summary.retrieve(QUERY))

# Auto-merging: embed the small leaves, hand back a parent if enough children hit.
parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[256, 80], chunk_overlap=0)
hierarchy = parser.get_nodes_from_documents(docs)
storage = StorageContext.from_defaults()
storage.docstore.add_documents(hierarchy)
leaf_index = VectorStoreIndex(get_leaf_nodes(hierarchy), storage_context=storage)
auto = AutoMergingRetriever(
    leaf_index.as_retriever(similarity_top_k=6),
    storage,
    simple_ratio_thresh=0.5,
)
show("Auto-merging", auto.retrieve(QUERY))

# Recursive: the node we embed is Patchwork, the link is the paired box (Azul).
root_nodes = [
    IndexNode.from_text_node(node, f"sheet-{node.metadata['pairs_with']}")
    for node in sheets
]
root = VectorStoreIndex(root_nodes)
retrievers = {"vector": root.as_retriever(similarity_top_k=3)}
for node in sheets:
    retrievers[f"sheet-{node.id_}"] = VectorStoreIndex([node]).as_retriever(
        similarity_top_k=1
    )
recursive = RecursiveRetriever("vector", retriever_dict=retrievers)
show("Recursive (follows pairs_with)", recursive.retrieve(QUERY))

# Query fusion: rewrite the question, run vector + BM25, fuse ranks with RRF.
fusion = QueryFusionRetriever(
    [vector, bm25],
    similarity_top_k=3,
    num_queries=4,
    mode="reciprocal_rerank",
    use_async=False,
)
show("Query fusion (RRF)", fusion.retrieve(QUERY))
