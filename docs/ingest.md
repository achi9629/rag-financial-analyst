# Ingestion Pipeline — `src/ingest.py`

## Overview

The ingestion pipeline converts raw markdown documentation into a searchable vector database (ChromaDB). This enables the RAG system to retrieve relevant domain knowledge when generating pandas code for financial fraud analysis queries.

## Pipeline Flow

```
docs/*.md  →  Load  →  Split  →  Embed  →  Store (ChromaDB)
                │         │         │            │
           8 files   117 chunks  384-dim     data/chromadb/
                                vectors
```

## Configuration

All parameters are externalized in `configs/config.yaml` under the `ingestion` section:

| Parameter | Value | Purpose |
|---|---|---|
| `docs_dir` | `docs` | Directory containing markdown source files |
| `chroma_dir` | `data/chromadb` | Persist directory for ChromaDB |
| `collection_name` | `rag_financial_analyst` | ChromaDB collection identifier |
| `chunk_size` | `800` | Max characters per chunk (~200 tokens) |
| `chunk_overlap` | `120` | Overlap between chunks (~30 tokens) |
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model (384-dim, 256 token max) |
| `embedding_device` | `cpu` | Keeps GPUs free for LLMs |

## Functions

### `load_ingestion_config(config_path)`
Loads the `ingestion` section from `config.yaml`. Raises `FileNotFoundError` if the file is missing or `ValueError` if the `ingestion` key is absent.

### `load_documents(docs_dir)`
Uses LangChain's `DirectoryLoader` with `TextLoader` to recursively load all `**/*.md` files from the docs directory. Returns a list of LangChain `Document` objects with source metadata.

### `extract_section(text)`
Parses a chunk's text content to find the first markdown heading (`#`, `##`, `###`). Returns the heading text as the section name, or `"unknown"` if no heading is found. Used to enrich chunk metadata for filtered retrieval.

### `split_documents(docs, chunk_size, chunk_overlap)`
Splits loaded documents into smaller chunks using `RecursiveCharacterTextSplitter`. The splitter uses markdown-aware separators in priority order:

1. `\n## ` — H2 headings (top priority — keeps full sections intact)
2. `\n### ` — H3 headings
3. `\n---` — Horizontal rules
4. `\n\n` — Paragraph breaks
5. `\n` — Line breaks
6. ` ` — Word boundaries (last resort)

After splitting, each chunk's metadata is enriched with:
- `filename` — source file name (e.g., `fraud_rules.md`)
- `section` — extracted heading from chunk content

### `get_embeddings(embedding_model, device)`
Initializes a `HuggingFaceEmbeddings` instance. Decorated with `@lru_cache` to avoid reloading the model on repeated calls.

### `create_vectorstore(chunks, chroma_dir, collection_name, embedding_model, device)`
Creates a ChromaDB vector store from document chunks:
1. Initializes the embedding model
2. Creates the persist directory if needed
3. **Clears any existing collection** with the same name (prevents duplicates on re-ingestion)
4. Embeds all chunks and stores them in ChromaDB
5. Persists to disk at `chroma_dir`

### `load_vectorstore(chroma_dir, collection_name, embedding_model, device)`
Loads a previously persisted ChromaDB vector store. Used by downstream modules (chain, pipeline) to retrieve chunks without re-ingesting.

### `test_retrieval(vectorstore, query, k)`
Runs a similarity search against the vector store and logs the top-k results with scores, filenames, and sections. Used for sanity checking retrieval quality.

### `ingest(docs_dir, chroma_dir, chunk_size, chunk_overlap, collection_name, embedding_model, device)`
Orchestrator function that runs the full pipeline: `load_documents` → `split_documents` → `create_vectorstore`. Returns the populated vector store.

### `main()`
Entry point when running `python src/ingest.py`. Loads config, runs ingestion, and executes 5 test queries to verify retrieval quality.

## Current Stats

| Metric | Value |
|---|---|
| Source documents | 8 markdown files |
| Total chunks | 117 |
| Embedding dimensions | 384 |
| Embedding model max tokens | 256 |
| Chunk size (chars) | 800 (~200 tokens) |
| Persist location | `data/chromadb/` |

## Usage

```bash
# Run full ingestion + retrieval test
python src/ingest.py

# Use in other modules
from ingest import load_ingestion_config, load_vectorstore

config = load_ingestion_config("configs/config.yaml")
vectorstore = load_vectorstore(config["chroma_dir"], config["collection_name"],
                                config["embedding_model"], config["embedding_device"])
results = vectorstore.similarity_search("detect fraud", k=5)
```
