import os
import sys
import json
import pytest

# Add root to sys path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from backend.rag.loaders import load_and_split_contracts
from backend.rag.vector_store import VectorStoreManager
from backend.rag.keyword_search import KeywordSearchManager
from backend.rag.hybrid_search import HybridSearcher
from backend.database.repository import DatabaseRepository

@pytest.fixture(scope="module")
def vector_manager():
    manager = VectorStoreManager(persist_directory="./chroma_test_db")
    # Ingest the 12 generated synthetic PDFs
    count = manager.ingest_documents()
    assert count > 0, "PDF ingestion failed"
    return manager

@pytest.fixture(scope="module")
def keyword_manager():
    return KeywordSearchManager()

@pytest.fixture(scope="module")
def hybrid_searcher(vector_manager, keyword_manager):
    return HybridSearcher(vector_manager, keyword_manager)

@pytest.fixture(scope="module")
def db_repo():
    return DatabaseRepository()

def test_pdf_ingestion_and_chunk_creation(vector_manager):
    """Test that PDFs are loaded, split into chunks, and metadata is preserved"""
    # A search with empty string should return top 4 chunks
    results = vector_manager.search_vector("test", top_k=1)
    assert len(results) > 0
    metadata = results[0]["metadata"]
    # Check that required metadata fields are extracted
    assert "contract_id" in metadata
    assert "tender_reference" in metadata
    assert "road_name" in metadata
    assert "document_type" in metadata
    
def test_metadata_filtering(vector_manager):
    """Test filtering ChromaDB results by metadata"""
    # Ask a generic question but strictly filter to CNT-001
    results = vector_manager.search_vector("Maintenance", metadata_filter={"contract_id": "CNT-001"}, top_k=5)
    assert len(results) > 0
    for r in results:
        assert r["metadata"]["contract_id"] == "CNT-001"

def test_exact_tender_reference_retrieval_keyword(keyword_manager):
    """Test exact BM25 keyword search for a specific tender reference"""
    results = keyword_manager.search_keyword("TN-2026-002", top_k=3)
    assert len(results) > 0
    # BM25 is good at exact matches, the top result should contain it
    found = any("TN-2026-002" in r["page_content"] for r in results)
    assert found, "BM25 failed to find exact tender reference"

def test_vector_search(vector_manager):
    """Test vector similarity search"""
    # Semantic query that doesn't rely on exact keywords
    results = vector_manager.search_vector("Who is in charge of synthetic roads?", top_k=3)
    assert len(results) > 0
    assert "score" in results[0]

def test_hybrid_search(hybrid_searcher):
    """Test hybrid search using RRF.
    
    The contract ID is reliably stored in chunk metadata (extracted during ingestion).
    Checking page_content is fragile for small/short PDFs where the contract ID may
    land in a different chunk than the top-k results. Metadata is the correct assertion.
    """
    results = hybrid_searcher.hybrid_search("CNT-003 pothole maintenance", top_k=5)
    assert len(results) > 0
    assert "rrf_score" in results[0]
    # CNT-003 should appear in the metadata of at least one top result
    found = any(r["metadata"].get("contract_id") == "CNT-003" for r in results)
    assert found, f"Hybrid search failed to surface CNT-003 in top results. Got: {[r['metadata'].get('contract_id') for r in results]}"

def test_ground_truth_cases(db_repo, hybrid_searcher):
    """Test that we can retrieve the correct structured and unstructured data using ground truth"""
    gt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/ground_truth.json'))
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)
        
    for case in ground_truth[:3]:  # Test first 3 cases
        demo_id = case["demo_case_id"]
        expected_road_id = case["expected_road_id"]
        expected_contract_id = case["expected_contract_id"]
        
        # 1. Test Structured Retrieval: Get road from demo location
        road = db_repo.get_road_from_demo_location(demo_id)
        assert road is not None
        assert road.road_id == expected_road_id
        
        # 2. Test Structured Retrieval: Get full relationship
        rel = db_repo.get_complete_relationship(road.road_id)
        assert rel is not None
        assert rel.contract.contract_id == expected_contract_id
        
        # 3. Test Unstructured Retrieval: Hybrid search for the road name
        results = hybrid_searcher.hybrid_search(road.road_name, top_k=5)
        
        # At least one of the top results should map to the correct contract ID
        found_expected = any(r["metadata"].get("contract_id") == expected_contract_id for r in results)
        assert found_expected, f"Hybrid search failed to find expected contract {expected_contract_id} for road {road.road_name}"
