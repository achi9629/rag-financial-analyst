# Vector Database — ChromaDB

## How ChromaDB Scores Chunks

### Default Distance Metric: Squared L2 (Euclidean)

ChromaDB uses **squared L2 distance** by default when you call `similarity_search_with_score`:

$$d(a, b) = \sum_{i=1}^{n} (a_i - b_i)^2$$

Where:
- $a$ = query embedding vector (384-dim for all-MiniLM-L6-v2)
- $b$ = stored chunk embedding vector
- Lower score = more similar (0 = identical)

### Score Interpretation

| Score Range | Meaning |
|---|---|
| 0.0 – 0.5 | Very high similarity |
| 0.5 – 1.0 | High similarity |
| 1.0 – 1.5 | Moderate similarity |
| 1.5 – 2.0 | Low similarity |
| > 2.0 | Dissimilar |

### Available Distance Functions

You can change the metric when creating a collection:

```python
collection = client.create_collection(
    name="my_collection",
    metadata={"hnsw:space": "cosine"}  # or "l2" (default) or "ip"
)
```

| Metric | Formula | Score Meaning |
|---|---|---|
| `l2` (default) | Squared Euclidean distance | Lower = more similar |
| `cosine` | $1 - \cos(a, b)$ | Lower = more similar (0 = identical, 2 = opposite) |
| `ip` | Inner product (negative dot product) | Lower (more negative) = more similar |

### How Retrieval Works Internally

1. **Query embedding**: Your query string is embedded into a 384-dim vector using the same model
2. **HNSW index lookup**: ChromaDB uses HNSW (Hierarchical Navigable Small World) algorithm for approximate nearest neighbor search
3. **Distance computation**: Computes distance between query vector and candidate vectors
4. **Top-k selection**: Returns the k nearest chunks sorted by distance (ascending)

```
Query: "how to detect fraud"
    │
    ▼ embed with all-MiniLM-L6-v2
[0.023, -0.156, 0.089, ..., 0.045]  (384-dim)
    │
    ▼ HNSW index → approximate nearest neighbors
    │
    ▼ compute L2 distance to each candidate
    │
    ▼ sort by distance (ascending)
Results: [(chunk_1, 0.70), (chunk_2, 0.71), (chunk_3, 0.90), ...]
```

## ChromaDB Architecture

### Storage Model

```
data/chromadb/
├── chroma.sqlite3          # Metadata store (collection info, document text, metadata)
└── <collection_uuid>/
    ├── header.bin          # HNSW index header
    ├── data_level0.bin     # HNSW graph level 0 (all nodes)
    ├── length.bin          # Vector lengths
    └── link_lists.bin      # HNSW graph upper level links
```

### What's Stored Per Chunk

| Field | Storage | Purpose |
|---|---|---|
| `id` | SQLite | Unique identifier (auto-generated UUID) |
| `embedding` | HNSW binary files | 384-dim float vector |
| `document` | SQLite | Original text (page_content) |
| `metadata` | SQLite (JSON) | filename, section, source path |

### Key Properties

- **Persistent**: Data survives process restarts (stored on disk)
- **Exact + Approximate search**: HNSW gives approximate results; small collections may use brute-force exact search
- **No GPU required**: All indexing and search runs on CPU
- **Single-process**: Not designed for concurrent writes (fine for this project)

## ChromaDB vs FAISS

| Feature | ChromaDB | FAISS |
|---|---|---|
| **Primary use** | Application vector store | Research/production vector search library |
| **Built by** | Chroma (startup) | Meta AI Research |
| **Persistence** | Built-in (SQLite + files) | Manual (`faiss.write_index` / `faiss.read_index`) |
| **Metadata storage** | Built-in (stores text + metadata alongside vectors) | None — vectors only, metadata managed externally |
| **Filtering** | Native metadata filters (`where={"filename": "fraud_rules.md"}`) | No built-in filtering — must filter post-retrieval or pre-filter IDs |
| **API** | High-level Python (add, query, delete) | Low-level C++/Python (add vectors, search vectors) |
| **Index types** | HNSW only | Flat, IVF, HNSW, PQ, LSH, ScaNN-like, and combinations |
| **GPU support** | No | Yes — `faiss-gpu` for massive-scale search |
| **Scale** | ~1M vectors (comfortable) | Billions of vectors (production-tested) |
| **Speed (1M vectors)** | ~10-50ms per query | ~1-5ms per query |
| **Speed (10k vectors)** | <5ms | <1ms |
| **LangChain integration** | `Chroma` class (first-class) | `FAISS` class (first-class) |
| **Document management** | Add/update/delete by ID | Append-only (delete requires rebuild) |
| **Dependencies** | `pip install chromadb` (pure Python + SQLite) | `pip install faiss-cpu` (C++ compiled) |
| **Concurrency** | Single writer, multiple readers | Thread-safe reads, single writer |
| **Quantization** | No (full precision only) | Yes — PQ, SQ, OPQ for memory reduction |
| **Hosted option** | Chroma Cloud | None (library only) |

### When to Use ChromaDB

- Prototyping and small-to-medium RAG apps (< 1M vectors)
- Need metadata filtering alongside vector search
- Want persistence without extra infrastructure
- LangChain/LlamaIndex integration needed
- **Your case**: 117 chunks → ChromaDB is perfect, zero overhead

### When to Use FAISS

- Very large scale (millions to billions of vectors)
- Need GPU-accelerated search
- Need quantization for memory efficiency
- Need custom index configurations (IVF + PQ for billion-scale)
- Don't need metadata stored alongside vectors
- Batch offline processing where latency matters

### Code Comparison

**ChromaDB:**
```python
from langchain_community.vectorstores import Chroma

# Create
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="data/chromadb")

# Query with metadata filter
results = vectorstore.similarity_search(
    "detect fraud",
    k=5,
    filter={"filename": "fraud_rules.md"}
)
```

**FAISS:**
```python
from langchain_community.vectorstores import FAISS

# Create
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("data/faiss_index")

# Query (no native metadata filter)
results = vectorstore.similarity_search("detect fraud", k=5)

# Load later
vectorstore = FAISS.load_local("data/faiss_index", embeddings)
```

### Why ChromaDB for This Project

1. **117 chunks** — FAISS's speed advantage is invisible at this scale
2. **Metadata filtering** — can filter by `filename` or `section` before semantic search (useful when routing knows which doc type to search)
3. **Built-in persistence** — no extra save/load code needed
4. **Document storage** — retrieves full text without a separate document store
5. **Simpler API** — less boilerplate for a prototype

### Migration Path (If Needed Later)

Switching from ChromaDB to FAISS in LangChain is a one-line change:

```python
# Before (ChromaDB)
from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="data/chromadb")

# After (FAISS)
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("data/faiss_index")
```

The rest of the pipeline (retriever, chain, query) stays identical because LangChain abstracts the vector store interface.
