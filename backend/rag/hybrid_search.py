from typing import List, Dict, Any, Optional
from backend.rag.vector_store import VectorStoreManager
from backend.rag.keyword_search import KeywordSearchManager

def reciprocal_rank_fusion(vector_results: List[dict], keyword_results: List[dict], k=60) -> List[dict]:
    """
    Combines vector and keyword search results using Reciprocal Rank Fusion (RRF).
    score = 1 / (k + rank)
    """
    fused_scores = {}
    doc_mapping = {}
    
    # Process vector results
    for rank, res in enumerate(vector_results):
        # Use page_content + source as a unique key for the chunk
        # (Using a chunk hash would be better in production)
        doc_key = f"{res['metadata'].get('document_id', '')}_{res['page_content'][:50]}"
        
        if doc_key not in fused_scores:
            fused_scores[doc_key] = 0
            doc_mapping[doc_key] = res
            
        fused_scores[doc_key] += 1 / (k + rank + 1)
        
    # Process keyword results
    for rank, res in enumerate(keyword_results):
        doc_key = f"{res['metadata'].get('document_id', '')}_{res['page_content'][:50]}"
        
        if doc_key not in fused_scores:
            fused_scores[doc_key] = 0
            doc_mapping[doc_key] = res
            
        fused_scores[doc_key] += 1 / (k + rank + 1)
        
    # Sort fused results descending
    sorted_fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    final_results = []
    for doc_key, rrf_score in sorted_fused:
        doc = doc_mapping[doc_key].copy()
        doc["rrf_score"] = rrf_score
        final_results.append(doc)
        
    return final_results

class HybridSearcher:
    def __init__(self, vector_store: VectorStoreManager, keyword_store: KeywordSearchManager):
        self.vector_store = vector_store
        self.keyword_store = keyword_store

    def hybrid_search(self, query: str, metadata_filter: Optional[Dict[str, Any]] = None, top_k: int = 4) -> List[dict]:
        # We fetch slightly more results from each source to get a good fusion candidate list
        vector_results = self.vector_store.search_vector(query, metadata_filter=metadata_filter, top_k=top_k * 2)
        keyword_results = self.keyword_store.search_keyword(query, top_k=top_k * 2)
        
        fused = reciprocal_rank_fusion(vector_results, keyword_results)
        return fused[:top_k]
