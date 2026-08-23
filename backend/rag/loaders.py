import os
import re
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_and_split_contracts(documents_dir: str = None) -> List[Document]:
    if documents_dir is None:
        documents_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../documents/contracts'))
        
    all_chunks = []
    
    # We use a modest chunk size to keep things simple and ensure we don't chop up short contracts too much
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    if not os.path.exists(documents_dir):
        print(f"Directory not found: {documents_dir}")
        return []
        
    for filename in os.listdir(documents_dir):
        if not filename.endswith(".pdf"):
            continue
            
        file_path = os.path.join(documents_dir, filename)
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        # Combine pages to extract metadata
        full_text = " ".join([page.page_content for page in pages])
        
        # Extract basic metadata using regex
        tender_match = re.search(r"Tender Reference:\s*(TN-\d+-\d+)", full_text)
        contract_match = re.search(r"Contract ID:\s*(CNT-\d+)", full_text)
        road_match = re.search(r"Road Name:\s*(.+?)(?=\s*District:|\n)", full_text)
        district_match = re.search(r"District:\s*(.+?)(?=\s*Area:|\n)", full_text)
        
        metadata = {
            "document_id": filename,
            "contract_id": contract_match.group(1) if contract_match else "UNKNOWN",
            "tender_reference": tender_match.group(1) if tender_match else "UNKNOWN",
            "road_name": road_match.group(1).strip() if road_match else "UNKNOWN",
            "district": district_match.group(1).strip() if district_match else "UNKNOWN",
            "document_type": "contract",
            "year": "2026"
        }
        
        chunks = text_splitter.split_documents(pages)
        for chunk in chunks:
            # Update the chunk's metadata with our extracted metadata
            chunk.metadata.update(metadata)
            all_chunks.append(chunk)
            
    return all_chunks
