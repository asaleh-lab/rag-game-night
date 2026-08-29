# rag-game-night

This repo demonstrates how to retrieve from a small catalog with more than one LangChain retriever, then keep the same embeddings in FAISS. The example we will use is the shelf binder at The Setup, a one-room board game cafe. In our case we need a game for two tired people, even when they do not name a title.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Put your OpenAI API key in `.env`.

## Let's retrieve by similarity, then MMR, then a score threshold

```powershell
python src/vector_retriever.py
```
