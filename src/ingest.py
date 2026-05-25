import yaml
import logging
import chromadb
from pathlib import Path
from functools import lru_cache
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

def load_ingestion_config(config_path: str) -> dict:
    
    '''
    Description:
        Load the ingestion configuration from a YAML file.
    Args:
        config_path (str): The path to the YAML configuration file.
    Returns:
        dict: A dictionary containing the ingestion configuration parameters.
    Raises:
        FileNotFoundError: If the configuration file is not found at the specified path.
        ValueError: If the 'ingestion' section is missing in the configuration file.
    '''
    
    if not Path(config_path).is_file():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if "ingestion" not in config:
        raise ValueError(f"Missing 'ingestion' section in config file: {config_path}")
    return config["ingestion"]    

def load_documents(docs_dir: str) -> list:
    
    '''
    Description:
        Load documents from the specified directory using DirectoryLoader.
    Args:
        docs_dir (str): The directory containing the documents to be loaded.
    Returns:
        list: A list of loaded documents.
    '''
    
    loader = DirectoryLoader(str(docs_dir),
                            glob = "**/*.md",
                            loader_cls = TextLoader,
                            loader_kwargs = {"encoding": "utf-8"},
                )
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} documents from {docs_dir}")
    return docs

def extract_section(text: str) -> str:
    
    '''
    Description:
        Extract the section title from the document content by looking for markdown headers.
    Args:
        text (str): The content of the document from which to extract the section title.
    Returns:
        str: The extracted section title or "unknown" if no header is found.
    '''
    
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return "unknown"

def split_documents(docs: list, 
                    chunk_size: int, 
                    chunk_overlap: int
        ) -> list:
    
    '''
    Description:
        Split the loaded documents into smaller chunks using RecursiveCharacterTextSplitter.
    Args:
        docs (list): The list of loaded documents to be split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.
    Returns:
        list: A list of split document chunks with enriched metadata.
    '''
    
    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size,
                                            chunk_overlap = chunk_overlap,
                                            separators = ["\n## ", "\n### ", "\n---", "\n\n", "\n", " "],
                )
    chunks = splitter.split_documents(docs)
    
    # Enrich metadata
    for chunk in chunks:
        source = Path(chunk.metadata.get("source", ""))
        chunk.metadata["filename"] = source.name
        chunk.metadata["section"] = extract_section(chunk.page_content)
    
    logger.info(f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks

@lru_cache(maxsize=128, typed=False)
def get_embeddings(embedding_model: str,
                   device: str = "cpu"
        ) -> HuggingFaceEmbeddings:
    
    '''
    Description:
        Initialize and return a HuggingFaceEmbeddings instance using the specified embedding model.
    Args:
        embedding_model (str): The name of the Hugging Face model to be used for generating embeddings.
    Returns:
        HuggingFaceEmbeddings: An instance of HuggingFaceEmbeddings initialized with the specified model.
    '''
    
    return HuggingFaceEmbeddings(model_name = embedding_model, 
                                 model_kwargs={"device": device},
            )
    
def create_vectorstore(chunks: list, 
                       chroma_dir: str, 
                       collection_name: str,
                       embedding_model: str,
                       device: str
        ) -> Chroma:
    
    '''
    Description:
        Create a Chroma vector store from the provided document chunks and persist it to the specified directory.
    Args:
        chunks (list): The list of document chunks to be stored in the vector store.
        chroma_dir (str): The directory where the Chroma vector store will be persisted.
        collection_name (str): The name of the collection within the Chroma vector store.
        embedding_model (str): The name of the Hugging Face model to be used for generating embeddings.
        device (str): The device to be used for generating embeddings (e.g., "cpu" or "cuda").
    Returns:
        Chroma: An instance of the Chroma vector store containing the document chunks and their embeddings.
    '''
    
    embeddings = get_embeddings(embedding_model, device)
    
    Path(chroma_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        client.delete_collection(collection_name)
        logger.info(f"Cleared existing collection '{collection_name}'")
    except (ValueError, chromadb.errors.NotFoundError):
        logger.info(f"No existing collection '{collection_name}' to clear")
    vectorstore = Chroma.from_documents(documents = chunks,
                                        embedding = embeddings,
                                        collection_name = collection_name,
                                        persist_directory = str(chroma_dir),
                    )
    logger.info(f"ChromaDB persisted at {chroma_dir} with {len(chunks)} chunks")
    return vectorstore

def load_vectorstore(chroma_dir: str, 
                     collection_name: str,
                     embedding_model: str,
                     device: str
        ) -> Chroma:
    
    '''
    Description:
        Load an existing Chroma vector store from the specified directory and collection name.
    Args:
        chroma_dir (str): The directory where the Chroma vector store is persisted.
        collection_name (str): The name of the collection to be loaded from the Chroma vector store.
    Returns:
        Chroma: An instance of the Chroma vector store loaded from the specified directory and collection name.
    '''
    
    embeddings = get_embeddings(embedding_model, device)
    
    return Chroma(collection_name=collection_name,
                persist_directory=str(chroma_dir),
                embedding_function=embeddings,
            )
    
def test_retrieval(vectorstore: Chroma, 
                   query: str, 
                   k: int = 5
        ) -> list:
    
    '''
    Description:
        Test the retrieval of relevant document chunks from the Chroma vector store based on a query.
    Args:
        vectorstore (Chroma): The Chroma vector store from which to retrieve relevant document chunks.
        query (str): The query string used to search for relevant document chunks in the vector store.
        k (int, optional): The number of top results to retrieve. Defaults to 5.
    Returns:
        list: A list of tuples containing the retrieved document chunks and their corresponding similarity scores.
    '''
    
    results = vectorstore.similarity_search_with_score(query, k=k)
    logger.info(f"\nQuery: '{query}'")
    logger.info(f"Top {k} results:")
    for i, (doc, score) in enumerate(results, 1):
        logger.info(f"  {i}. [score={score:.4f}] {doc.metadata['filename']} | {doc.metadata['section']}")
        logger.info(f"     {doc.page_content[:100]}...")
    return results

def ingest(docs_dir: str, 
            chroma_dir: str,
            chunk_size: int,
            chunk_overlap: int,
            collection_name: str,
            embedding_model: str,
            device: str
        ) -> Chroma:
    
    '''
    Description:
        The main ingestion function that orchestrates the loading, splitting, and vector store creation process.
    Args:
        docs_dir (str): The directory containing the documents to be ingested.
        chroma_dir (str): The directory where the Chroma vector store will be persisted.
        chunk_size (int): The maximum size of each document chunk.
        chunk_overlap (int): The number of characters to overlap between document chunks.
        collection_name (str): The name of the collection within the Chroma vector store.
        embedding_model (str): The name of the Hugging Face model to be used for generating
    Returns:
        Chroma: An instance of the Chroma vector store created from the ingested documents.
    '''
    
    docs = load_documents(docs_dir)
    chunks = split_documents(docs, 
                             chunk_size, 
                             chunk_overlap
                )
    vectorstore = create_vectorstore(chunks, 
                                     chroma_dir, 
                                     collection_name, 
                                     embedding_model,
                                     device
                    )
    return vectorstore

def main() -> None:
    
    project_root = Path(__file__).parent.parent
    
    config_path = project_root / "configs" / "config.yaml"
    config = load_ingestion_config(str(config_path))
    
    docs_dir = project_root / config['docs_dir']
    chroma_dir = project_root / config['chroma_dir']
    chunk_size = config['chunk_size']
    chunk_overlap = config['chunk_overlap']
    collection_name = config['collection_name']
    embedding_model = config['embedding_model']
    device = config["embedding_device"]
    
    vectorstore = ingest(str(docs_dir), 
                         str(chroma_dir), 
                         chunk_size, 
                         chunk_overlap, 
                         collection_name, 
                         embedding_model,
                         device
            )
    
    test_queries = ["how to detect fraud",
                    "balance mismatch detection",
                    "what is CASH_OUT",
                    "regulatory reporting threshold",
                    "transaction velocity",
                ]
    
    for q in test_queries:
        test_retrieval(vectorstore, q)
    
if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()