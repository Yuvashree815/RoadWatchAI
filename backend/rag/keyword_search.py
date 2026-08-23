import os
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from backend.rag.loaders import load_and_split_contracts

class KeywordSearchManager:
    def __init__(self, documents_dir: str = None):
        self.chunks = load_and_split_contracts(documents_dir)
        self.tokenized_corpus = [self._tokenize(chunk.page_content) for chunk in self.chunks]
        
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def _tokenize(self, text: str) -> List[str]:
        # Simple whitespace tokenization, lowercased
        # Suitable for exact matches like "CNT-001"
        return text.lower().split()

    def search_keyword(self, query: str, top_k: int = 4) -> List[dict]:
        """
        Performs BM25 keyword search over the loaded corpus.
        Returns formatted dictionaries similar to vector search.
        """
        if not self.bm25 or not self.chunks:
            return []
            
        tokenized_query = self._tokenize(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Zip documents with scores and sort by score descending
        doc_score_pairs = zip(self.chunks, doc_scores)
        sorted_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        
        # Filter out zero scores if query didn't match anything
        results = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)  # BM25 score
            }
            for doc, score in sorted_pairs[:top_k] if score > 0
        ]
        
        return results
