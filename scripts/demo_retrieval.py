import os
import sys

# Add root to sys path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from backend.rag.vector_store import VectorStoreManager
from backend.rag.keyword_search import KeywordSearchManager
from backend.rag.hybrid_search import HybridSearcher
from backend.database.repository import DatabaseRepository

def print_result(title, results):
    print(f"\n{'='*50}\n{title}\n{'='*50}")
    for i, r in enumerate(results):
        print(f"\nRank: {i+1}")
        print(f"Contract ID: {r['metadata'].get('contract_id', 'N/A')}")
        print(f"Tender Ref: {r['metadata'].get('tender_reference', 'N/A')}")
        print(f"Road Name: {r['metadata'].get('road_name', 'N/A')}")
        print(f"Score: {r.get('score', r.get('rrf_score', 'N/A'))}")
        # Print a snippet of content
        print(f"Content Snippet: {r['page_content'][:150]}...")

def main():
    print("Initializing Database Layer...")
    db = DatabaseRepository()
    
    print("\nInitializing Vector Store (Embedding PDFs)...")
    v_store = VectorStoreManager(persist_directory="./chroma_demo_db")
    count = v_store.ingest_documents()
    print(f"Ingested {count} document chunks into ChromaDB.")
    
    print("\nInitializing Keyword Store (BM25)...")
    k_store = KeywordSearchManager()
    
    print("\nInitializing Hybrid Searcher...")
    h_store = HybridSearcher(v_store, k_store)
    
    query = "Find the maintenance tender associated with road RD-007"
    print(f"\nExecuting Query: '{query}'")
    
    # Get Road Name for RD-007 to show structure retrieval
    road = db.get_road("RD-007")
    if road:
        print(f"\n[Structured DB] Resolved RD-007 to Road Name: {road.road_name}")
        # Enhance query with structured data knowledge
        enhanced_query = f"{query} {road.road_name}"
    else:
        enhanced_query = query
        
    v_results = v_store.search_vector(enhanced_query, top_k=3)
    print_result("VECTOR SEARCH RESULTS", v_results)
    
    k_results = k_store.search_keyword(enhanced_query, top_k=3)
    print_result("KEYWORD SEARCH RESULTS (BM25)", k_results)
    
    h_results = h_store.hybrid_search(enhanced_query, top_k=3)
    print_result("HYBRID SEARCH RESULTS (RRF)", h_results)

if __name__ == "__main__":
    main()
