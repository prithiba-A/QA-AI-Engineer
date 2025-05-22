from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.IndexFlatL2(384)
        self.text_chunks = []
        self.sources = []
        self.vectors = None  # Store all vectors here

    def build_index(self, data):
        all_vectors = []
        for url, content in data.items():
            chunks = [content[i:i+500] for i in range(0, len(content), 500)]
            vectors = self.model.encode(chunks)
            all_vectors.append(vectors)
            self.text_chunks.extend(chunks)
            self.sources.extend([url]*len(chunks))

        all_vectors = np.vstack(all_vectors).astype('float32')
        self.index.add(all_vectors)
        self.vectors = all_vectors  # Keep a copy for retrieval

    def search(self, query, top_k=3):
        q_vector = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(q_vector, top_k)
        results = [(self.text_chunks[i], self.sources[i]) for i in indices[0]]
        return results
