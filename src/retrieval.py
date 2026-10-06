"""
Geospatial Knowledge Retrieval (Geo-RAG) Engine.
Offloaded to CPU / System RAM to preserve GPU VRAM for Vision-Language Models.
"""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class GeospatialKnowledgeRetriever:
    def __init__(self, corpus_path="data/knowledge_base/geospatial_corpus.json", 
                 model_name="all-MiniLM-L6-v2", device="cpu"):
        self.corpus_path = corpus_path
        self.device = device
        print(f"[Geo-RAG] Loading embedding model '{model_name}' on {self.device}...")
        self.encoder = SentenceTransformer(model_name, device=self.device)
        self.passages = []
        self.index = None
        self.embeddings = None
        self._load_corpus()

    def _load_corpus(self):
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            self.passages = json.load(f)
        print(f"[Geo-RAG] Loaded {len(self.passages)} authoritative geospatial knowledge passages.")

    def build_index(self):
        """Builds a FAISS FlatIP (inner product / cosine) index on CPU."""
        texts = [f"{p['topic']}: {p['content']}" for p in self.passages]
        print(f"[Geo-RAG] Encoding {len(texts)} passages on {self.device}...")
        embeddings = self.encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self.embeddings = embeddings.astype(np.float32)
        
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        print(f"[Geo-RAG] FAISS Index built successfully with dimension {dim}.")

    def save_index(self, index_path="data/knowledge_base/faiss_index.bin"):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        print(f"[Geo-RAG] Index persisted to {index_path}")

    def load_index(self, index_path="data/knowledge_base/faiss_index.bin"):
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print(f"[Geo-RAG] Index loaded from {index_path}")
            return True
        return False

    def retrieve(self, query_text, image_context_cues=None, top_k=3):
        """
        Multimodal Context-Aware Retrieval.
        Combines textual query with visual semantic context tags (e.g. flooded road, roof damage).
        """
        if self.index is None:
            self.build_index()

        # Context-aware query enrichment
        if image_context_cues:
            augmented_query = f"{query_text} [Geospatial Visual Context: {image_context_cues}]"
        else:
            augmented_query = query_text

        q_emb = self.encoder.encode([augmented_query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(q_emb, top_k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < len(self.passages):
                doc = self.passages[idx]
                results.append({
                    "rank": rank + 1,
                    "doc_id": doc["doc_id"],
                    "source": doc["source"],
                    "topic": doc["topic"],
                    "content": doc["content"],
                    "score": float(score)
                })
        return results

    def format_retrieved_context(self, retrieved_docs):
        """Formats retrieved documents for prompt injection."""
        context_str = ""
        for doc in retrieved_docs:
            context_str += f"[{doc['doc_id']}] ({doc['source']} - {doc['topic']}): {doc['content']}\n"
        return context_str.strip()
