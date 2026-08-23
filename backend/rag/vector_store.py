import os
from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from backend.rag.embeddings import get_embeddings_model
from backend.rag.loaders import load_and_split_contracts

class VectorStoreManager:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = get_embeddings_model()
        self.collection_name = "synthetic_contracts"
        
        # Initialize the Chroma store
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def ingest_documents(self, documents_dir: str = None) -> int:
        chunks = load_and_split_contracts(documents_dir)
        if not chunks:
            return 0
            
        # We clear existing elements to prevent duplicates during multiple runs
        # (For production, you'd use a more robust checking mechanism)
        try:
            self.vector_store.delete_collection()
        except:
            pass
            
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )
        return len(chunks)

    def search_vector(self, query: str, metadata_filter: Optional[Dict[str, Any]] = None, top_k: int = 4) -> List[dict]:
        """
        Embeds query, searches ChromaDB with optional metadata filter,
        and returns results formatted as dictionaries.
        """
        # Chroma expects filter to be a dict, e.g. {"contract_id": "CNT-001"}
        results = self.vector_store.similarity_search_with_score(
            query,
            k=top_k,
            filter=metadata_filter
        )
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)  # Typically L2 distance in default Chroma
            })
            
        return formatted_results
