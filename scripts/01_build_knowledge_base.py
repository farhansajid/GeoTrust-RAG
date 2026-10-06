"""
Script 01: Build and Persist FAISS Index for Geospatial Knowledge Base
"""

import sys
import os

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval import GeospatialKnowledgeRetriever

def main():
    print("=" * 70)
    print("GeoTrust-RAG: Stage 1 - Building Geospatial Knowledge Vector Store")
    print("=" * 70)
    
    retriever = GeospatialKnowledgeRetriever(
        corpus_path="data/knowledge_base/geospatial_corpus.json",
        model_name="all-MiniLM-L6-v2",
        device="cpu"
    )
    
    retriever.build_index()
    retriever.save_index("data/knowledge_base/faiss_index.bin")
    
    # Test sample retrieval
    sample_queries = [
        "Is the road safe for emergency ambulance access?",
        "How is severe structural collapse classified by FEMA?",
        "Submerged vehicle hazard depth"
    ]
    
    print("\n[Sanity Check] Testing Retrieval for Sample Queries:")
    for q in sample_queries:
        docs = retriever.retrieve(q, top_k=2)
        print(f"\nQuery: '{q}'")
        for d in docs:
            print(f"  -> Top-{d['rank']} [{d['doc_id']}] (Score: {d['score']:.4f}): {d['topic']}")
            
    print("\n[SUCCESS] Knowledge base indexed and verified successfully!")

if __name__ == "__main__":
    main()
