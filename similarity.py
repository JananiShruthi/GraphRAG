import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# Simple in-memory DB
class DocumentDB:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.tok_docs = []
        self.bm25 = None

    def add_document(self, content, embedding):
        self.documents.append(content)
        self.embeddings.append(embedding)
        self.tok_docs.append(content.lower().split())
        self.bm25 = BM25Okapi(self.tok_docs)

    def get_best_match(self, query_embedding, query_text):
        # Cosine similarity
        cos_sims = cosine_similarity([query_embedding], self.embeddings)[0]
        cos_idx = np.argmax(cos_sims)

        # BM25
        query_tokens = query_text.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_idx = int(np.argmax(bm25_scores))

        return {
            "cosine_best_doc": self.documents[cos_idx],
            "cosine_score": cos_sims[cos_idx],
            "bm25_best_doc": self.documents[bm25_idx],
            "bm25_score": bm25_scores[bm25_idx]
        }

# Dummy embedding using TF-IDF (replace with real model as needed)
class SimpleEmbedder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed(self, doc):
        return self.model.encode(doc)

if __name__ == "__main__":
    db = DocumentDB()
    embedder = SimpleEmbedder()

    while True:
        no_of_documents = int(input("How many documents do you want to add? "))
        for _ in range(no_of_documents):
            doc_content = input("Enter document content: ")
            doc_embedding = embedder.embed(doc_content)
            db.add_document(doc_content, doc_embedding)

        # Get query from user
        query = input("Enter your query: ")
        query_embedding = embedder.embed(query)

        # Retrieve best match
        result = db.get_best_match(query_embedding, query)
        print("\nCosine Similarity Best Match:")
        print(result["cosine_best_doc"])
        print("Score:", result["cosine_score"])

        print("\nBM25 Best Match:")
        print(result["bm25_best_doc"])
        print("Score:", result["bm25_score"])
